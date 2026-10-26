"""Joint training regression; synthetic CPU data, never downloads Imagenette."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_aligned import joint
from pace_aligned.common import DataBundle, EMA, ExperimentConfig, make_backbone


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(4, 10)

    def forward(self, images, return_feature=False):
        features = torch.relu(self.bn(self.conv(images))).mean((2, 3))
        logits = self.fc(self.drop(features))
        return (logits, features, features, features) if return_feature else logits


def bundle():
    generator = torch.Generator().manual_seed(312)
    data = TensorDataset(torch.rand(6, 3, 8, 8, generator=generator), torch.arange(6))
    loader = DataLoader(data, batch_size=2, shuffle=False)
    return DataBundle(train=loader, train_eval=loader, val=loader, selection_val=loader,
                      test=loader, train_indices=list(range(6)), val_indices=list(range(6, 12)))


def config(tmp_path, clean=True):
    return ExperimentConfig(
        method="joint_mean", dataset="cifar10", device="cpu", workers=0,
        output_root=str(tmp_path), run_name="synthetic_only", batch_size=2,
        backbone="resnet18", patch_size=2, train_attack_steps=1, eval_attack_steps=1,
        cw_eval_steps=1, max_train_batches=1, max_eval_batches=1,
        task_sequence=("clean", "joint") if clean else ("joint",),
        task_epochs={"clean": 2 if clean else 0, "joint": 3},
        task_lrs={"clean": 0.1, "joint": 0.01 if clean else 0.1},
        final_test_methods=("clean", "patch", "pgd_l2", "pgd"))


def patch_data(monkeypatch):
    monkeypatch.setattr(joint, "build_dataloaders", lambda cfg: bundle())
    monkeypatch.setattr(joint, "make_backbone", lambda cfg: TinyModel())


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_defaults_and_actual_entry_not_common_old_batch(monkeypatch):
    args = joint.build_parser().parse_args([])
    cfg, options = joint.resolve_config(args)
    assert cfg.dataset == "imagenette" and cfg.backbone == "resnet18"
    assert cfg.image_size == 160 and cfg.batch_size == 32
    assert cfg.autoattack_batch_size == 4 and cfg.autoattack_samples == 0
    assert cfg.patch_size == 40 and cfg.l2_eps == 2.5
    assert cfg.task_epochs == {"clean": 100, "joint": 90}
    assert options.attacks == ("clean", "patch", "pgd_l2", "pgd")
    observed = []
    monkeypatch.setattr(joint, "run", lambda cfg, *a, **k: observed.append(cfg.batch_size))
    joint.main([])
    joint.main(["--batch-size", "16"])
    assert observed == [32, 16]


def test_from_scratch_and_cli_overrides(capsys):
    args = joint.build_parser().parse_args([
        "--clean-epochs", "0", "--joint-epochs", "190", "--batch-size", "16",
        "--joint-attacks", "patch", "pgd_l2", "pgd", "--l2-eps", "1",
        "--patch-size", "20", "--linf-alpha", "0.001", "--autoattack-samples", "1000",
        "--autoattack-batch-size", "2", "--plot-final-features", "--skip-autoattack"])
    cfg, options = joint.resolve_config(args)
    assert cfg.task_sequence == ("joint",) and cfg.task_lrs["joint"] == 0.1
    assert cfg.batch_size == 16 and cfg.l2_eps == 1 and cfg.patch_size == 20
    assert cfg.autoattack_batch_size == 2 and cfg.autoattack_samples == 1000
    assert options.attacks == ("patch", "pgd_l2", "pgd")
    assert cfg.plot_final_features and cfg.tsne_dimensions == "both"
    assert not any(name.startswith("autoattack") for name in cfg.final_test_methods)
    joint.main(["--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["config"]["batch_size"] == 32


@pytest.mark.parametrize("flags", [
    ["--batch-size", "0"], ["--batch-size", "-1"], ["--workers", "-1"],
    ["--clean-epochs", "-1"], ["--joint-epochs", "0"],
    ["--joint-attacks", "patch", "patch"], ["--grad-clip", "-1"],
    ["--l2-eps", "-1"], ["--linf-alpha", "nan"], ["--joint-lr", "0"],
    ["--patch-size", "161"], ["--train-attack-steps", "0"],
    ["--max-train-batches", "0"], ["--max-eval-batches", "0"],
    ["--autoattack-batch-size", "0"], ["--autoattack-samples", "-1"],
])
def test_invalid_options(flags):
    with pytest.raises(ValueError):
        joint.resolve_config(joint.build_parser().parse_args(flags))


def test_training_signature_rejects_training_not_plot_or_aa_changes():
    cfg, options = joint.resolve_config(joint.build_parser().parse_args([]))
    original = joint.digest(joint.training_spec(cfg, options))
    cfg.autoattack_samples = 1000
    cfg.autoattack_batch_size = 2
    cfg.plot_final_features = True
    cfg.tsne_dimensions = "2"
    cfg.resume = True
    assert original == joint.digest(joint.training_spec(cfg, options))
    cfg.batch_size = 16
    assert original != joint.digest(joint.training_spec(cfg, options))


def test_every_condition_contributes_before_one_optimizer_step(monkeypatch):
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
    expected = copy.deepcopy(model)
    images, labels = torch.rand(2, 3, 2, 2), torch.tensor([1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    cfg = ExperimentConfig(method="joint_mean", device="cpu", batch_size=2)
    conditions = ("clean", "patch", "pgd_l2", "pgd")
    calls = []
    before = copy.deepcopy(model.state_dict())
    def fake_attack(current, x, y, name, *args, **kwargs):
        assert torch.equal(x, images) and torch.equal(y, labels)
        assert_same(before, current.state_dict())
        calls.append(name)
        return x + 0.01 * conditions.index(name)
    monkeypatch.setattr(joint, "generate_attack", fake_attack)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    expected_opt = torch.optim.SGD(expected.parameters(), lr=0.05)
    expected_loss = sum(
        torch.nn.functional.cross_entropy(expected(images + 0.01 * index), labels)
        for index in range(4)) / 4
    expected_loss.backward()
    expected_opt.step()
    stats = joint.train_epoch(model, EMA(model, 0.9), loader, optimizer, conditions,
                              1, 0, cfg, joint.JointOptions(), torch.device("cpu"))
    assert calls == list(conditions)
    assert stats["optimizer_updates"] == 1
    assert stats["generated_adversarial_samples"] == 6
    assert stats["condition_examples"] == 8 and stats["clean_source_examples"] == 2
    assert stats["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
    for actual, target in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-7)


def test_attacks_do_not_update_batchnorm_before_training():
    torch.manual_seed(22)
    model = TinyModel()
    cfg = config(Path("unused"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    stats = joint.train_epoch(model, EMA(model, 0.9), bundle().train, optimizer,
                              joint.TASKS, 1, 0, cfg, joint.JointOptions(), torch.device("cpu"))
    assert int(model.bn.num_batches_tracked) == 4
    assert stats["generated_by_attack"] == {"patch": 2, "pgd_l2": 2, "pgd": 2}


def test_real_imagenette_resnet18_joint_step():
    cfg, options = joint.resolve_config(joint.build_parser().parse_args([
        "--device", "cpu", "--workers", "0", "--batch-size", "2",
        "--train-attack-steps", "1", "--max-train-batches", "1"]))
    torch.manual_seed(42)
    model = make_backbone(cfg)
    loader = DataLoader(TensorDataset(torch.rand(2, 3, 160, 160), torch.tensor([0, 1])),
                        batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    stats = joint.train_epoch(model, EMA(model, 0.999), loader, optimizer,
                              options.attacks, 1, 0, cfg, options, torch.device("cpu"))
    assert set(stats["condition_ce"]) == set(joint.TASKS)
    assert stats["optimizer_updates"] == 1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("phase_index,interrupt_epoch", [(0, 1), (0, 2), (1, 1), (1, 3)])
def test_epoch_resume_exact_with_bn_dropout_and_native_attacks(
        tmp_path, monkeypatch, phase_index, interrupt_epoch):
    patch_data(monkeypatch)
    baseline, resumed = config(tmp_path / "continuous"), config(tmp_path / "interrupted")
    expected = joint.run(baseline, joint.JointOptions())
    real_save = joint.atomic_checkpoint
    def interrupt(path, payload):
        real_save(path, payload)
        if (Path(path).name == "last.pt" and payload["phase_index"] == phase_index
                and payload["next_epoch"] == interrupt_epoch):
            raise KeyboardInterrupt("Simulated interruption after durable epoch checkpoint")
    monkeypatch.setattr(joint, "atomic_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        joint.run(resumed, joint.JointOptions())
    monkeypatch.setattr(joint, "atomic_checkpoint", real_save)
    resumed.resume = True
    actual = joint.run(resumed, joint.JointOptions())
    assert actual == expected
    left = torch.load(baseline.run_dir / "last.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed.run_dir / "last.pt", map_location="cpu", weights_only=False)
    for name in ("model", "ema", "optimizer", "scheduler", "best_snapshot", "epoch_rows",
                 "stage_summaries", "best_score", "next_epoch", "signature"):
        assert_same(left[name], right[name])
    assert left["resource_state"]["generated"] == right["resource_state"]["generated"]
    rows = json.loads((resumed.run_dir / "epoch_metrics.json").read_text())
    assert len(rows) == 5
    assert all(row["generated_adversarial_samples"] == 6 for row in rows if row["task"] == "joint")
    assert len(json.loads((resumed.run_dir / "stage_summaries.json").read_text())) == 2
    # Completed resume / eval-only must not call the training loop at all.
    monkeypatch.setattr(joint, "train_epoch", lambda *a, **k: pytest.fail("Unexpected retraining"))
    assert joint.run(resumed, joint.JointOptions(), eval_only=True) == expected


def test_fresh_joint_no_clean_phase_and_mismatch_protection(tmp_path, monkeypatch):
    patch_data(monkeypatch)
    cfg = config(tmp_path, clean=False)
    joint.run(cfg, joint.JointOptions())
    assert (cfg.run_dir / "joint_best.pt").exists()
    assert not (cfg.run_dir / "clean_best.pt").exists()
    with pytest.raises(FileExistsError):
        joint.run(cfg, joint.JointOptions())
    before = (cfg.run_dir / "config.json").read_bytes()
    cfg.resume = True
    cfg.batch_size = 4
    with pytest.raises(ValueError, match="configuration mismatch"):
        joint.run(cfg, joint.JointOptions())
    assert (cfg.run_dir / "config.json").read_bytes() == before


def test_zero_accuracy_included_in_best_selection(tmp_path, monkeypatch):
    patch_data(monkeypatch)
    cfg = config(tmp_path, clean=False)
    cfg.task_epochs["joint"] = 1
    monkeypatch.setattr(joint, "evaluate_seen",
                        lambda *a, **k: {"clean": 80., "patch": 0., "pgd_l2": 40., "pgd": 0.})
    joint.run(cfg, joint.JointOptions())
    saved = torch.load(cfg.run_dir / "joint_best.pt", map_location="cpu", weights_only=False)
    assert saved["combined_metric"] == 30.


def test_partial_final_evaluation_resumes_and_invalidates(tmp_path, monkeypatch):
    cfg = config(tmp_path, clean=False)
    cfg.run_dir.mkdir(parents=True)
    model, data = TinyModel().eval(), bundle()
    called = []
    def failing_accuracy(model, loader, method, *a, **k):
        called.append(method)
        if method == "pgd_l2":
            raise RuntimeError("Simulated OOM")
        return 50.
    monkeypatch.setattr(joint, "evaluate_accuracy", failing_accuracy)
    with pytest.raises(RuntimeError, match="OOM"):
        joint.final_test(model, data, cfg, torch.device("cpu"), "sig")
    assert called == ["clean", "patch", "pgd_l2"]
    called.clear()
    def success_accuracy(model, loader, method, *a, **k):
        called.append(method)
        return 50.
    monkeypatch.setattr(joint, "evaluate_accuracy", success_accuracy)
    joint.final_test(model, data, cfg, torch.device("cpu"), "sig")
    assert called == ["pgd_l2", "pgd"]
    called.clear()
    cfg.eval_attack_steps += 1
    joint.final_test(model, data, cfg, torch.device("cpu"), "sig")
    assert called == list(cfg.final_test_methods)


def test_feature_collection_reuses_real_evaluation_features(tmp_path, monkeypatch):
    cfg = config(tmp_path, clean=False)
    cfg.run_dir.mkdir(parents=True)
    cfg.plot_final_features = True
    cfg.max_eval_batches = None
    rendered = []
    from pace_aligned.feature_plots import FeatureStore
    monkeypatch.setattr(FeatureStore, "render", lambda self, cfg: rendered.append(tuple(self.methods)))
    model, data = TinyModel().eval(), bundle()
    expected = joint.final_test(model, data, cfg, torch.device("cpu"), "sig")
    for method in cfg.final_test_methods:
        assert (cfg.run_dir / "final_features" / f"{method}.npz").exists()
    monkeypatch.setattr(joint, "evaluate_accuracy", lambda *a, **k: pytest.fail("Cache not reused"))
    assert joint.final_test(model, data, cfg, torch.device("cpu"), "sig") == expected
    assert len(rendered) == 2


def test_autoattack_uses_independent_batch_and_both_norms(tmp_path, monkeypatch):
    cfg = config(tmp_path, clean=False)
    cfg.run_dir.mkdir(parents=True)
    cfg.final_test_methods = ("clean", "autoattack_l2", "autoattack")
    cfg.autoattack_samples, cfg.autoattack_batch_size = 0, 1
    seen = []
    def fake_aa(model, loader, norm, eps, config, device, **kwargs):
        seen.append((norm, eps, config.autoattack_samples, config.autoattack_batch_size))
        return 25.
    monkeypatch.setattr(joint, "evaluate_autoattack", fake_aa)
    joint.final_test(TinyModel().eval(), bundle(), cfg, torch.device("cpu"), "sig")
    assert seen == [("L2", cfg.l2_eps, 0, 1), ("Linf", cfg.linf_eps, 0, 1)]
