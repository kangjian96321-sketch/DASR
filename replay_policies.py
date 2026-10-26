"""Historical-task selection, independent of PyTorch and global random state.

All policies choose ONE historical task per active replay epoch. Random replay
samples a task with replacement; uniform replay cycles through the task list.
The activation schedule is independent of task selection in matched-budget runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

POLICIES = ('dtr', 'random', 'uniform', 'lowest_accuracy', 'none')


@dataclass(frozen=True)
class ReplayDecision:
    active: bool
    target: Optional[str]
    maximum_drop_pp: Optional[float]
    reason: str

    def to_dict(self):
        return asdict(self)


class ReplaySelector:
    """Select only from historical tasks; ties follow their supplied order.

    ``gate='native'``: DTR activates only for a drop strictly above threshold;
    the other selectors activate at every eligible epoch.
    ``gate='matched'``: use an externally recorded DTR activation mask for ALL
    selectors. This matches replay epochs/examples, not measured GPU time.
    Epoch zero is always replay-free, matching the supplied training loop.
    """

    def __init__(self, policy: str, seed: int, drop_threshold_pp: float = 1.0):
        if policy not in POLICIES:
            raise ValueError(f'Unknown replay policy: {policy}')
        if not math.isfinite(drop_threshold_pp) or drop_threshold_pp < 0:
            raise ValueError('drop_threshold_pp must be finite and nonnegative')
        self.policy = policy
        self.drop_threshold_pp = float(drop_threshold_pp)
        self.rng = random.Random(seed)
        self.uniform_cursor = 0

    def choose(self, history: Sequence[str], reference: Mapping[str, float],
               accuracy: Mapping[str, float], epoch: int, gate: str = 'native',
               matched_active: Optional[bool] = None) -> ReplayDecision:
        history = list(history)
        if len(set(history)) != len(history):
            raise ValueError('Historical tasks must be unique')
        if epoch < 0 or gate not in ('native', 'matched'):
            raise ValueError('Invalid epoch or replay gate')
        if gate == 'matched' and type(matched_active) is not bool:
            raise ValueError('A matched schedule entry must be a boolean')
        if self.policy == 'none':
            if gate == 'matched' and matched_active:
                raise ValueError('A no-replay ablation cannot match an active mask')
            return ReplayDecision(False, None, None, 'no_replay_ablation')
        if not history or epoch == 0:
            if matched_active:
                raise ValueError('Matched mask requests replay before it is eligible')
            return ReplayDecision(False, None, None, 'no_history_or_first_epoch')
        for task in history:
            if task not in reference or task not in accuracy:
                raise ValueError(f'Missing previous-epoch accuracy/reference: {task}')
            if not all(math.isfinite(float(x)) and 0 <= float(x) <= 100
                       for x in (reference[task], accuracy[task])):
                raise ValueError('Accuracies must be finite percentages in [0, 100]')
        drops = {task: float(reference[task]) - float(accuracy[task]) for task in history}
        maximum_drop = max(drops.values())
        active = (matched_active if gate == 'matched' else
                  maximum_drop > self.drop_threshold_pp if self.policy == 'dtr' else True)
        if not active:
            return ReplayDecision(False, None, maximum_drop, 'gate_inactive')
        if self.policy == 'dtr':
            target = max(history, key=drops.__getitem__)
        elif self.policy == 'lowest_accuracy':
            target = min(history, key=lambda task: float(accuracy[task]))
        elif self.policy == 'random':
            target = self.rng.choice(history)
        else:
            target = history[self.uniform_cursor % len(history)]
            self.uniform_cursor += 1
        return ReplayDecision(True, target, maximum_drop, gate)

    def state_dict(self):
        return {'policy': self.policy, 'drop_threshold_pp': self.drop_threshold_pp,
                'uniform_cursor': self.uniform_cursor, 'rng_state': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['policy'] != self.policy or state['drop_threshold_pp'] != self.drop_threshold_pp:
            raise ValueError('Cannot resume a different replay policy/configuration')
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
        self.rng.setstate(tuples(state['rng_state']))
        self.uniform_cursor = int(state['uniform_cursor'])


def historical_tasks(task_sequence, stage, include_clean=True):
    if not 0 <= stage < len(task_sequence):
        raise ValueError('Invalid stage')
    return [task for task in task_sequence[:stage] if include_clean or task != 'clean']
