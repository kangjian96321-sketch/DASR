import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import far_replay
import train_replay_comparison as training
from far_replay import SOURCE_MODES, choose_source, decompose
from pace_aligned.common import ExperimentConfig, class_balanced_indices
from pace_components import AdaptiveDecoupledLACDLoss


def tiny_config(output: Path, mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        method=f"pace_uniform_{mode}",
        dataset="cifar10",
        output_root=str(output),
        run_name="four_sources",
        device="cpu",
        depth=10,
        widen_factor=1,
        batch_size=4,
        workers=0,
        task_sequence=("clean", "patch"),
        task_epochs={"clean": 2, "patch": 2},
        train_attack_steps=1,
        eval_attack_steps=1,
        max_train_batches=1,
        max_eval_batches=1,
        final_test_methods=("clean",),
    )


def test_decomposition_and_source_choices():
    values = decompose(80, 60, 50)
    assert values["historical_forgetting_pp"] == 20
    assert values["attack_reoptimization_gap_pp"] == 10
    assert values["degradation_pp"] == 30
    assert values["identity_residual_pp"] == 0
    assert values["historical_probability"] == pytest.approx(2 / 3)
    assert choose_source("historical", values, 1) == "historical"
    assert choose_source("regenerated", values, 1) == "regenerated"
    assert choose_source("fg_adaptive", {**values, "historical_probability": 1.0}, 1) == "historical"
    assert choose_source("fg_adaptive_lazy", {**values, "historical_probability": 0.0}, 1) == "regenerated"


def test_class_balanced_indices_are_exact_and_deterministic():
    labels = torch.arange(100).repeat_interleave(3)
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(len(labels), 1), labels
    )
    first = class_balanced_indices(dataset, 200, 100, seed=42)
    second = class_balanced_indices(dataset, 200, 100, seed=42)
    selected_labels = labels[first]
    counts = torch.bincount(selected_labels, minlength=100)
    assert first == second
    assert counts.tolist() == [2] * 100
    assert len(first) == len(set(first)) == 200


def test_groupwise_temperature_preserves_joint_negative_formula():
    torch.manual_seed(20)
    query = torch.randn(6, 4, requires_grad=True)
    keys = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    criterion = AdaptiveDecoupledLACDLoss()
    actual = criterion(
        query, keys, labels, temperature_groups=(3, 3)
    )
    qn = F.normalize(query, dim=1, eps=1e-8)
    kn = F.normalize(keys, dim=1, eps=1e-8)
    distance = (qn - kn).norm(dim=1)
    normalized = []
    for group in distance.split(3):
        normalized.append(
            (group - group.min()) / (group.max() - group.min() + 1e-4)
        )
    temperature = (0.05 + 0.1 * torch.cat(normalized)).detach()
    similarity = qn @ kn.T
    positive = similarity.diag() / temperature
    negative = (similarity / 0.1).masked_fill(
        labels[:, None] == labels[None, :], -1e9
    )
    expected = (
        torch.logsumexp(torch.cat([positive[:, None], negative], dim=1), dim=1)
        - positive
    ).mean()
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("mode", SOURCE_MODES)
def test_four_source_modes_execute_and_cache_as_specified(tmp_path, monkeypatch, mode):
    torch.set_num_threads(2)
    if mode == "fg_adaptive_lazy":
        original = far_replay.decompose

        def force_history(*args, **kwargs):
            result = original(*args, **kwargs)
            result["historical_probability"] = 1.0
            result["regenerated_probability"] = 0.0
            return result

        monkeypatch.setattr(far_replay, "decompose", force_history)
    cfg = tiny_config(tmp_path, mode)
    training.run_experiment(
        cfg,
        policy="uniform",
        smoke=True,
        replay_source_mode=mode,
        far_monitor_size=4,
        far_history_size=4,
    )
    resources = json.loads((cfg.run_dir / "resource_metrics.json").read_text())
    replay_stage = resources[1]
    bank = torch.load(
        cfg.run_dir / "far_history_bank.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert bank["protocol"]["sampling"] == "deterministic_class_balanced"
    assert bank["protocol"]["fresh_monitor_source"] == "reused_full_validation"
    monitor_counts = torch.bincount(
        bank["panels"]["monitor"]["labels"], minlength=cfg.num_classes
    )
    history_counts = torch.bincount(
        bank["panels"]["training"]["labels"], minlength=cfg.num_classes
    )
    assert int(monitor_counts.max() - monitor_counts.min()) <= 1
    assert int(history_counts.max() - history_counts.min()) <= 1
    assert replay_stage["generated_replay_monitor_adversarial_samples"] == 0
    clean_ready = "training_adv" in bank["tasks"]["clean"]
    if mode in ("historical", "fg_adaptive"):
        assert clean_ready
        assert replay_stage["lazy_cache_builds"] == 0
    elif mode == "regenerated":
        assert not clean_ready
        assert replay_stage["historical_replay_examples"] == 0
        assert replay_stage["regenerated_replay_examples"] > 0
    else:
        assert clean_ready
        assert replay_stage["lazy_cache_builds"] == 1
        assert replay_stage["historical_replay_examples"] > 0
        assert replay_stage["regenerated_replay_examples"] == 0
    assert (cfg.run_dir / "fg_decomposition.json").exists()
    assert (cfg.run_dir / "far_history_summary.json").exists()
    epoch_rows = json.loads((cfg.run_dir / "epoch_metrics.json").read_text())
    monitored = next(row for row in epoch_rows if row["fg_monitor"])
    clean_metrics = monitored["fg_monitor"]["clean"]
    assert clean_metrics["routing_basis"] == "stage_decomposition"
    assert "stage_degradation_pp" in clean_metrics
    assert "cumulative_degradation_pp" in clean_metrics
    assert monitored["reference_mode"] == "previous_stage_best_checkpoint"


@pytest.mark.parametrize("mode", SOURCE_MODES)
def test_command_line_accepts_every_source_mode(mode):
    parsed = training.parse_args(["--replay-source-mode", mode])
    assert parsed.replay_source_mode == mode
    assert parsed.far_monitor_size == 500
    assert parsed.far_history_size == 500
