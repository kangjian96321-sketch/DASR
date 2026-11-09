"""A global reservoir over (source stage, training sample) observations.

The buffer is frozen during a stage and receives one deterministic training-set
pass after the stage. This prevents multi-epoch current-task writes from erasing
old threats. Sampling and insertion have independent, checkpointed RNG streams.
"""
import random
import torch
from dasr_aligned.baseline_memory import ReplayMemory
from dasr_aligned.common import isolated_random_seed


class HistoricalMemory(ReplayMemory):
    def __init__(self, capacity, input_shape, num_classes, seed):
        super().__init__(capacity, input_shape, num_classes, False, seed)
        self.sampling_rng = random.Random(seed + 1)
        self.last_committed_stage = -1

    def sample(self, count, device):
        if count <= 0:
            raise ValueError("Replay sample count must be positive")
        if not len(self):
            return None
        ids = self.sampling_rng.sample(range(len(self)), min(count, len(self)))
        return {"images": self.images[ids].to(device), "labels": self.labels[ids].to(device),
                "tasks": self.tasks[ids].clone()}

    def assert_history(self, current_stage):
        if self.last_committed_stage != current_stage - 1:
            raise ValueError("Historical memory must contain exactly the completed stages")
        if len(self) and (bool((self.tasks[:len(self)] < 0).any()) or
                         bool((self.tasks[:len(self)] >= current_stage).any())):
            raise ValueError("Replay contains a current or future source stage")

    def commit_stage(self, loader, stage, cfg, device):
        """Called only at the stage boundary; last.pt makes this exactly-once."""
        self.assert_history(stage)
        before = self.num_seen
        with isolated_random_seed(cfg.seed + 8257 + stage * 1009, device):
            for batch_index, batch in enumerate(loader):
                if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
                    break  # explicit smoke-test setting, recorded in the run config
                self.add(batch[0], batch[1], stage)
        if self.num_seen == before:
            raise ValueError("Cannot commit an empty training stage")
        self.last_committed_stage = stage
        return self.num_seen - before

    def state_dict(self):
        return {**super().state_dict(), "sampling_rng": self.sampling_rng.getstate(),
                "last_committed_stage": self.last_committed_stage}

    def load_state_dict(self, state):
        stage = int(state["last_committed_stage"])
        tasks = state["tasks"]
        if stage < -1 or (len(tasks) and (bool((tasks < 0).any()) or bool((tasks > stage).any()))):
            raise ValueError("Invalid memory source-stage state")
        super().load_state_dict(state)
        self.sampling_rng.setstate(state["sampling_rng"])
        self.last_committed_stage = stage

    def summary(self):
        result = super().summary()
        result["seen_stage_sample_pairs"] = result.pop("seen_training_exposures")
        result.update(last_committed_stage=self.last_committed_stage,
                      update_policy="one_training_eval_pass_per_completed_stage",
                      replay_policy="uniform_from_frozen_global_historical_reservoir")
        return result
