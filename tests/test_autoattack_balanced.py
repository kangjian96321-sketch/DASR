"""Deterministic class-balanced AA sampling and legacy-cache regression tests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_aligned import common, feature_plots
import train_replay_comparison as pace


class LabelsOnly(Dataset):
    def __init__(self, counts):
        self.targets = [label for label, count in enumerate(counts) for _ in range(count)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        raise AssertionError("Sampling must not decode images when targets are available")


@pytest.mark.parametrize("dataset_name,classes", [("imagenette", 10), ("cifar10", 10), ("cifar100", 100)])
def test_exact_1000_balanced_no_replacement_from_imbalanced_pool(dataset_name, classes):
    dataset = LabelsOnly([120 + label for label in range(classes)])
    cfg = common.ExperimentConfig(method="pace", dataset=dataset_name, autoattack_samples=1000)
    ids = common.autoattack_subset_indices(dataset, cfg)
    assert len(ids) == len(set(ids)) == 1000
    assert Counter(dataset.targets[index] for index in ids) == dict.fromkeys(range(classes), 1000 // classes)
    assert ids == common.autoattack_subset_indices(dataset, cfg)
    cfg.autoattack_subset_seed += 1
    assert set(ids) != set(common.autoattack_subset_indices(dataset, cfg))


def test_nondivisible_size_and_fixed_remainder_allocation():
    dataset = LabelsOnly([200] * 10)
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", autoattack_samples=1003)
    ids = common.autoattack_subset_indices(dataset, cfg)
    counts = Counter(dataset.targets[index] for index in ids)
    assert sorted(counts.values()) == [100] * 7 + [101] * 3
    assert ids == common.autoattack_subset_indices(dataset, cfg)


@pytest.mark.parametrize("requested", [0, 210, 999])
def test_full_set_keeps_natural_imbalance_and_original_order(requested):
    dataset = LabelsOnly([1] + [label * 5 + 1 for label in range(1, 10)])
    # Use its exact size, a larger request, and explicit 0.
    if requested == 210:
        requested = len(dataset)
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", autoattack_samples=requested)
    ids = common.autoattack_subset_indices(dataset, cfg)
    assert sorted(ids) == list(range(len(dataset)))
    assert Counter(dataset.targets[index] for index in ids) == Counter(dataset.targets)
    assert ids == torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.autoattack_subset_seed)).tolist()


def test_shortage_never_silently_oversamples_or_changes_requested_count():
    dataset = LabelsOnly([1] + [20] * 9)
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", autoattack_samples=100)
    with pytest.raises(ValueError, match="Class 0 has 1 samples, but 10 are required"):
        common.autoattack_subset_indices(dataset, cfg)
    cfg.autoattack_samples = 0
    assert len(common.autoattack_subset_indices(dataset, cfg)) == len(dataset)


def test_missing_class_and_invalid_label_are_explicit_errors():
    dataset = LabelsOnly([0] + [20] * 9)
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", autoattack_samples=100)
    with pytest.raises(ValueError, match="missing classes"):
        common.autoattack_subset_indices(dataset, cfg)
    dataset.targets[-1] = 10
    with pytest.raises(ValueError, match="Invalid class label"):
        common.autoattack_subset_indices(dataset, cfg)


def test_subset_mapping_method_independence_and_global_rng_isolation():
    dataset = Subset(LabelsOnly([20] * 10), list(reversed(range(200))))
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", autoattack_samples=100)
    py_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    ids = common.autoattack_subset_indices(dataset, cfg)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    assert Counter(common.dataset_target(dataset, index) for index in ids) == dict.fromkeys(range(10), 10)
    cfg.method, cfg.seed, cfg.evaluation_seed = "sseat", 123, 567
    assert ids == common.autoattack_subset_indices(dataset, cfg)


def test_old_random_subset_feature_cache_cannot_be_reused(tmp_path):
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", output_root=str(tmp_path))
    model = torch.nn.Linear(3, 10)
    loader = DataLoader(LabelsOnly([200] * 10))
    current = feature_plots.make_feature_store(model, loader, cfg)
    old_spec = dict(current.manifest["spec"])
    old_spec.pop("autoattack_sampling_policy")
    old_signature = hashlib.sha256(json.dumps(old_spec, sort_keys=True).encode()).hexdigest()
    legacy = feature_plots.FeatureStore(cfg.run_dir, old_signature, old_spec, ["autoattack"])
    collector = legacy.collector("autoattack")
    collector.add([0, 1], torch.tensor([0, 0]), torch.ones(2, 3), torch.tensor([0, 1]))
    collector.finish(50.)
    assert legacy.has("autoattack")
    current = feature_plots.make_feature_store(model, loader, cfg)
    assert not current.has("autoattack") and current.signature != old_signature
    assert current.manifest["spec"]["autoattack_sampling_policy"] == common.AUTOATTACK_SAMPLING_POLICY


def test_legacy_pace_accuracy_cache_is_invalidated_without_retraining(tmp_path, monkeypatch):
    from ablation_config import method_name
    cfg = common.ExperimentConfig(
        method=method_name("uniform", "full"), dataset="cifar10", device="cpu", workers=0,
        output_root=str(tmp_path), run_name="aa_sampling_resume", depth=10, widen_factor=1,
        batch_size=4, train_attack_steps=1, eval_attack_steps=1,
        task_epochs=dict.fromkeys(common.TASKS, 1), max_train_batches=1, max_eval_batches=1,
        final_test_methods=("clean",))
    current_digest = pace.digest

    def legacy_digest(value):
        if isinstance(value, dict) and "model" in value and "autoattack_samples" in value:
            value = dict(value)
            value.pop("autoattack_sampling_policy", None)
        return current_digest(value)

    monkeypatch.setattr(pace, "digest", legacy_digest)
    pace.run_experiment(cfg, policy="uniform", smoke=True)
    old_meta = json.loads((cfg.run_dir / "final_test_meta.json").read_text())
    old_identity = json.loads((cfg.run_dir / "run_complete.json").read_text())["identity"]
    monkeypatch.setattr(pace, "digest", current_digest)
    monkeypatch.setattr(pace, "train_epoch", lambda *a, **kw: pytest.fail("Unexpected retraining"))
    measured = []
    original_evaluate = pace.evaluate_seen

    def track_evaluation(*args, **kwargs):
        measured.append(args[2])
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(pace, "evaluate_seen", track_evaluation)
    cfg.resume = True
    pace.run_experiment(cfg, policy="uniform", smoke=True, eval_only=True)
    new_meta = json.loads((cfg.run_dir / "final_test_meta.json").read_text())
    assert new_meta["fingerprint"] != old_meta["fingerprint"]
    assert new_meta["evaluation_settings"]["autoattack_sampling_policy"] == common.AUTOATTACK_SAMPLING_POLICY
    assert json.loads((cfg.run_dir / "run_complete.json").read_text())["identity"] == old_identity
    assert measured == [["clean"]]
    pace.run_experiment(cfg, policy="uniform", smoke=True, eval_only=True)
    assert measured == [["clean"]]  # Once updated, results are reusable.
