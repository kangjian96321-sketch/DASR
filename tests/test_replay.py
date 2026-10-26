import copy
import json
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_policies import ReplaySelector, historical_tasks
from dasr_components import TeacherFeatureQueue, AdaptiveDecoupledLACDLoss
from train_DASR import ADAPTIVE_TASKS, load_matched_schedule


H = ['clean', 'patch', 'pgd_l2']
REF = dict(zip(H, [95.0, 90.0, 40.0]))
ACC = dict(zip(H, [94.0, 70.0, 35.0]))


def test_drop_and_difficulty_are_distinct():
    assert ReplaySelector('dtr', 42).choose(H, REF, ACC, 1).target == 'patch'
    assert ReplaySelector('lowest_accuracy', 42).choose(H, REF, ACC, 1).target == 'pgd_l2'
    assert historical_tasks(['clean', 'patch', 'pgd_l2', 'pgd'], 3) == H


def test_strict_threshold_first_epoch_and_ties():
    selector = ReplaySelector('dtr', 42)
    assert not selector.choose(H, REF, ACC, 0).active
    assert not selector.choose(['clean'], {'clean': 95}, {'clean': 94}, 1).active
    assert selector.choose(['clean'], {'clean': 95}, {'clean': 93.9}, 1).active
    assert selector.choose(H, {h: 90 for h in H}, {h: 80 for h in H}, 1).target == 'clean'


def test_random_is_private_and_resumable():
    state = random.getstate()
    selector = ReplaySelector('random', 43)
    for epoch in range(1, 6):
        selector.choose(H, REF, ACC, epoch)
    saved = json.loads(json.dumps(selector.state_dict()))
    expected = [selector.choose(H, REF, ACC, e).target for e in range(6, 30)]
    restored = ReplaySelector('random', 0)
    restored.load_state_dict(saved)
    assert expected == [restored.choose(H, REF, ACC, e).target for e in range(6, 30)]
    assert random.getstate() == state


def test_uniform_balances_active_epochs_only():
    selector = ReplaySelector('uniform', 43)
    targets = []
    for epoch in range(1, 12):
        decision = selector.choose(H, REF, ACC, epoch, 'matched', epoch % 3 != 0)
        if decision.active:
            targets.append(decision.target)
    assert targets == (H * 3)[:len(targets)]
    assert max(targets.count(h) for h in H) - min(targets.count(h) for h in H) <= 1


@pytest.mark.parametrize('policy', ['dtr', 'random', 'uniform', 'lowest_accuracy'])
def test_identical_external_gate(policy):
    selector = ReplaySelector(policy, 42)
    mask = [False, True, False, True]
    assert [selector.choose(H, REF, ACC, e, 'matched', active).active
            for e, active in enumerate(mask)] == mask


def test_missing_measurement_and_invalid_gate_fail_closed():
    with pytest.raises(ValueError):
        ReplaySelector('dtr', 42).choose(H, REF, {}, 1)
    with pytest.raises(ValueError):
        ReplaySelector('random', 42).choose(H, REF, ACC, 1, 'matched', None)
    with pytest.raises(ValueError):
        ReplaySelector('random', 42).choose([], {}, {}, 0, 'matched', True)


def test_class_quotas_and_resume():
    queue = TeacherFeatureQueue(capacity=8, num_classes=3, device='cpu')
    queue.enqueue(torch.randn(9, 4), torch.tensor([0] * 5 + [1] * 3 + [2]))
    assert queue.class_capacities == [3, 3, 2]
    assert queue.class_counts == [3, 3, 1]
    restored = TeacherFeatureQueue(8, 3, 'cpu')
    restored.load_state_dict(copy.deepcopy(queue.state_dict()))
    keys, labels = torch.randn(4, 4), torch.tensor([0, 1, 2, 2])
    queue.enqueue(keys, labels)
    restored.enqueue(keys, labels)
    assert torch.equal(queue.features, restored.features)
    assert torch.equal(queue.labels, restored.labels)
    assert queue.class_ptrs == restored.class_ptrs


def test_adaptive_loss_matches_multiset_formula_and_detached_temperature():
    torch.manual_seed(20)
    assert 'patch' in ADAPTIVE_TASKS
    q = torch.randn(6, 4, requires_grad=True)
    k = torch.randn(3, 4).repeat(2, 1)
    labels = torch.tensor([0, 1, 2]).repeat(2)
    loss = AdaptiveDecoupledLACDLoss()(q, k, labels)
    qn, kn = F.normalize(q, dim=1, eps=1e-8), F.normalize(k, dim=1, eps=1e-8)
    distances = (qn - kn).norm(dim=1)
    temp = (0.05 + 0.1 * (distances - distances.min()) /
            (distances.max() - distances.min() + 1e-4)).detach()
    sim = qn @ kn.T
    positive = sim.diag() / temp
    negative = (sim / 0.1).masked_fill(labels[:, None] == labels[None, :], -1e9)
    expected = (torch.logsumexp(torch.cat([positive[:, None], negative], 1), 1) - positive).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(loss, q, retain_graph=True)[0],
                          torch.autograd.grad(expected, q)[0], atol=1e-6)


def test_gate_schedule_rejects_seed_or_missing_epochs(tmp_path):
    path = tmp_path / 'schedule.json'
    payload = {'policy': 'dtr', 'gate': 'native', 'complete': True,
               'protocol_signature': 'seed42', 'epochs': [{'stage': 0, 'epoch': 0, 'active': False}]}
    path.write_text(json.dumps(payload))
    assert load_matched_schedule(path, 'seed42', [1]) == {(0, 0): False}
    with pytest.raises(ValueError):
        load_matched_schedule(path, 'seed43', [1])
    with pytest.raises(ValueError):
        load_matched_schedule(path, 'seed42', [2])
