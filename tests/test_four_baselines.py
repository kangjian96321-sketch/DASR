"""CPU regressions for the four protocol-aligned legacy methods."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_aligned import continual_baselines as runner
from pace_aligned import baseline_losses as losses
from pace_aligned.baseline_memory import ReplayMemory
from pace_aligned.common import DataBundle, EMA, ExperimentConfig, make_backbone


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(4, 10)

    def forward(self, x, return_feature=False):
        features = torch.relu(self.bn(self.conv(x))).mean((2, 3))
        logits = self.fc(self.drop(features))
        return (logits, features, features, features) if return_feature else logits


def data_bundle():
    generator = torch.Generator().manual_seed(72)
    dataset = TensorDataset(torch.rand(6, 3, 32, 32, generator=generator), torch.arange(6))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return DataBundle(train=loader, train_eval=loader, val=loader, selection_val=loader,
                      test=loader, train_indices=list(range(6)), val_indices=list(range(6, 12)))


def config(path, method):
    return ExperimentConfig(
        method=method, dataset="cifar10", device="cpu", workers=0, backbone="resnet18",
        batch_size=2, output_root=str(path), run_name="test_only",
        task_epochs=dict.fromkeys(runner.TASKS, 2), train_attack_steps=1,
        eval_attack_steps=1, cw_eval_steps=1, patch_size=2,
        max_train_batches=1, max_eval_batches=1,
        final_test_methods=("clean", "patch", "pgd_l2", "pgd"))


def patch_runtime(monkeypatch):
    monkeypatch.setattr(runner, "build_dataloaders", lambda _: data_bundle())
    monkeypatch.setattr(runner, "make_backbone", lambda _: TinyModel())


def same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", losses.METHODS)
@pytest.mark.parametrize("dataset,batch,side,eps", [
    ("cifar10", 64, 8, 0.5), ("cifar100", 64, 8, 0.5),
    ("tiny_imagenet", 16, 16, 1.), ("imagenette", 32, 40, 2.5)])
def test_defaults_per_entry(method, dataset, batch, side, eps):
    args = runner.build_parser(method).parse_args(["--dataset", dataset, "--backbone", "resnet18"])
    cfg, options = runner.resolve_config(method, args)
    assert cfg.batch_size == batch
    assert cfg.patch_size == side and cfg.l2_eps == eps
    assert cfg.task_epochs == {"clean": 100, "patch": 30, "pgd_l2": 30, "pgd": 30}
    assert cfg.autoattack_samples == 0
    assert cfg.autoattack_batch_size == (4 if dataset == "imagenette" else min(32, batch))
    assert cfg.ema_decay == 0.999 and options.grad_clip == 1.
    assert options.trades_inner_loss == "kl"


@pytest.mark.parametrize("script,method", [
    ("train_derpp_baseline.py", "derpp"), ("train_gsa_baseline.py", "gsa"),
    ("train_trades.py", "trades"), ("train_mart.py", "mart")])
def test_wrappers_forward_and_do_not_override_gpu(script, method, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    called = []
    monkeypatch.setattr(runner, "main", lambda name: called.append(name))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / script), run_name="__main__")
    assert called == [method]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_cli_overrides_and_dry_run(capsys):
    args = runner.build_parser("trades").parse_args([
        "--dataset", "imagenette", "--backbone", "resnet18", "--batch-size", "16",
        "--beta", "4", "--trades-inner-loss", "ce", "--l2-eps", "1", "--l2-alpha", "0.15",
        "--autoattack-samples", "1000", "--autoattack-batch-size", "2", "--grad-clip", "0",
        "--plot-final-features", "--skip-autoattack"])
    cfg, options = runner.resolve_config("trades", args)
    assert cfg.batch_size == 16 and cfg.l2_eps == 1. and cfg.l2_alpha == 0.15
    assert cfg.autoattack_samples == 1000 and cfg.autoattack_batch_size == 2
    assert options.beta("patch") == options.beta("pgd_l2") == options.beta("pgd") == 4.
    assert options.trades_inner_loss == "ce" and options.grad_clip == 0.
    assert cfg.plot_final_features and cfg.tsne_dimensions == "both"
    assert not any(name.startswith("autoattack") for name in cfg.final_test_methods)
    runner.main("mart", ["--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["config"]["dataset"] == "cifar100"


@pytest.mark.parametrize("flags", [
    ["--batch-size", "0"], ["--batch-size", "-1"], ["--clean-epochs", "0"],
    ["--attack-epochs", "-1"], ["--ema-decay", "1"], ["--grad-clip", "-1"],
    ["--selection-probe-size", "6000"], ["--buffer-size", "0"],
    ["--replay-batch-size", "0"], ["--der-alpha", "-1"], ["--l2-eps", "-1"],
    ["--linf-alpha", "nan"], ["--train-attack-steps", "0"],
    ["--max-train-batches", "0"], ["--patch-size", "1000"],
    ["--autoattack-batch-size", "0"], ["--depth", "12"]])
def test_bad_options(flags):
    with pytest.raises(ValueError):
        runner.resolve_config("derpp", runner.build_parser("derpp").parse_args(flags))


@pytest.mark.parametrize("store_logits", [False, True])
def test_memory_content_capacity_serialization_rng(store_logits):
    memory = ReplayMemory(3, (3, 2, 2), 10, store_logits, seed=23)
    assert memory.sample(2, "cpu") is None
    images = torch.arange(60).float().reshape(5, 3, 2, 2) / 100
    labels = torch.arange(5)
    logits = torch.arange(50).float().reshape(5, 10)
    memory.add(images[:2], labels[:2], 0, logits[:2] if store_logits else None)
    partial = memory.state_dict()
    assert partial["images"].shape[0] == 2
    memory.add(images[2:], labels[2:], 1, logits[2:] if store_logits else None)
    assert len(memory) == 3 and memory.num_seen == 5
    saved = memory.state_dict()
    restored = ReplayMemory(3, (3, 2, 2), 10, store_logits, seed=999)
    restored.load_state_dict(saved)
    same(memory.sample(99, "cpu"), restored.sample(99, "cpu"))
    memory.add(images, labels, 2, logits if store_logits else None)
    restored.add(images, labels, 2, logits if store_logits else None)
    same(memory.state_dict(), restored.state_dict())
    assert memory.summary()["valid_storage_bytes"] == memory.summary()["cpu_allocated_bytes"]
    sampled = memory.sample(3, "cpu")
    for image, label in zip(sampled["images"], sampled["labels"]):
        assert torch.equal(image, images[label])


def test_reservoir_new_exposure_can_be_rejected(monkeypatch):
    memory = ReplayMemory(1, (1, 1, 1), 2, False, 1)
    memory.add(torch.zeros(1, 1, 1, 1), torch.tensor([0]), 0)
    seen_upper_bounds = []
    def select_last(stop):
        seen_upper_bounds.append(stop)
        return stop - 1
    monkeypatch.setattr(memory.rng, "randrange", select_last)
    memory.add(torch.ones(1, 1, 1, 1), torch.tensor([1]), 1)
    assert seen_upper_bounds == [2]  # fixed off-by-one; the old upper bound was 1.
    assert memory.labels[0].item() == 0


def test_trades_formula_and_clean_branch_gradient():
    natural = torch.tensor([[2., -1., 0.], [0., 1., 2.]], requires_grad=True)
    adversarial = torch.tensor([[0., 1., -1.], [2., 0., 1.]], requires_grad=True)
    labels = torch.tensor([0, 2])
    loss, ce, kl = losses.trades_objective(natural, adversarial, labels, 6.)
    p, q = natural.softmax(1), adversarial.softmax(1)
    manual = -(p[torch.arange(2), labels].log()).mean() + 6. * (p * (p.log() - q.log())).sum(1).mean()
    torch.testing.assert_close(loss, manual)
    expected_grad = torch.autograd.grad(manual, (natural, adversarial), retain_graph=True)
    actual_grad = torch.autograd.grad(loss, (natural, adversarial))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_mart_formula_including_singleton_batch(batch_size):
    torch.manual_seed(10)
    natural, adversarial = torch.randn(batch_size, 4), torch.randn(batch_size, 4)
    labels = torch.arange(batch_size)
    loss, boosted, robust = losses.mart_objective(natural, adversarial, labels, 6.)
    p, q = natural.softmax(1), adversarial.softmax(1)
    wrong = q.clone()
    wrong[torch.arange(batch_size), labels] = -1.
    alternative = wrong.argmax(1)
    manual_adv = -q[torch.arange(batch_size), labels].log().mean()
    manual_adv -= (1.0001 - q[torch.arange(batch_size), alternative] + 1e-12).log().mean()
    manual_kl = (p * (p.log() - (q + 1e-12).log())).sum(1)
    manual_robust = (manual_kl * (1.0000001 - p[torch.arange(batch_size), labels])).mean()
    torch.testing.assert_close(boosted, manual_adv)
    torch.testing.assert_close(robust, manual_robust)
    torch.testing.assert_close(loss, manual_adv + 6 * manual_robust)


@pytest.mark.parametrize("task", ("clean", "patch", "pgd_l2", "pgd"))
def test_kl_attack_constraints_labels_independence_modes_and_rng(task):
    torch.manual_seed(31)
    model = TinyModel().train()
    x, y = torch.rand(2, 3, 8, 8), torch.tensor([1, 2])
    cfg = ExperimentConfig(method="trades", device="cpu", patch_size=2, train_attack_steps=3)
    rng = torch.get_rng_state().clone()
    output = losses.generate_kl_attack(model, x, y, task, cfg, 19)
    other_labels = losses.generate_kl_attack(model, x, 9-y, task, cfg, 19)
    assert torch.equal(output, other_labels)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training and int(model.bn.num_batches_tracked) == 0
    assert output.min() >= 0 and output.max() <= 1 and not output.requires_grad
    delta = output - x
    if task == "pgd":
        assert delta.abs().max() <= cfg.linf_eps + 1e-6
    elif task == "pgd_l2":
        assert (delta.flatten(1).norm(dim=1) <= cfg.l2_eps + 1e-6).all()
    elif task == "patch":
        assert ((delta.abs().sum(1) > 0).flatten(1).sum(1) <= 4).all()
    else:
        assert torch.equal(output, x)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("new,reference,expected,conflict", [
    ([-1., 1.], [1., 0.], [1., 1.], True),
    ([1., 1.], [1., 0.], [2., 1.], False),
    ([1., -1.], [0., 0.], [1., -1.], False)])
def test_gsa_projection_and_frozen_parameter_offsets(new, reference, expected, conflict):
    model = torch.nn.Module()
    model.frozen = torch.nn.Parameter(torch.ones(5), requires_grad=False)
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.unused = torch.nn.Parameter(torch.ones(3))
    a = (model.weight * torch.tensor(new)).sum()
    b = (model.weight * torch.tensor(reference)).sum()
    stats = losses.backward_gsa(a, b, model)
    torch.testing.assert_close(model.weight.grad, torch.tensor(expected))
    assert model.frozen.grad is None and model.unused.grad is None
    assert stats["gradient_conflict"] is conflict


@pytest.mark.parametrize("method", losses.METHODS)
def test_one_training_step_and_memory_semantics(method):
    torch.manual_seed(32)
    cfg = config(Path("unused"), method)
    options = losses.BaselineOptions(method, buffer_size=5)
    model = TinyModel()
    ema = EMA(model, cfg.ema_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    memory = runner.make_memory(cfg, options)
    loader = data_bundle().train
    stats = runner.train_epoch(model, ema, loader, optimizer, "pgd", 3, 0,
                               cfg, options, memory, torch.device("cpu"))
    assert stats["optimizer_updates"] == 1 and stats["generated_current"] == 2
    if memory is not None:
        assert len(memory) == 2 and memory.num_seen == 2
        images, labels = next(iter(loader))
        if method == "gsa":
            assert torch.equal(memory.images[:2], images)
        else:
            assert (memory.images[:2] - images).abs().max() <= cfg.linf_eps + 1e-6
            assert not torch.equal(memory.images[:2], images)
            assert memory.logits.shape == (5, 10)
    stats2 = runner.train_epoch(model, ema, loader, optimizer, "pgd", 3, 1,
                                cfg, options, memory, torch.device("cpu"))
    if method == "gsa":
        assert stats2["generated_replay"] == 2 and stats2["gradient_projections"] == 1
    elif method == "derpp":
        assert stats2["generated_replay"] == 0 and stats2["replay_examples"] == 2
    else:
        assert stats2["replay_examples"] == 0 and memory is None


def test_trades_ce_compatibility_dispatch(monkeypatch):
    cfg = config(Path("unused"), "trades")
    model = TinyModel()
    calls = []
    actual = runner.generate_attack
    def ce_attack(*args, **kwargs):
        calls.append(args[3])
        return actual(*args, **kwargs)
    monkeypatch.setattr(runner, "generate_attack", ce_attack)
    monkeypatch.setattr(runner, "generate_kl_attack", lambda *a, **k: pytest.fail("Unexpected KL attack"))
    runner.train_epoch(model, EMA(model, .9), data_bundle().train,
                       torch.optim.SGD(model.parameters(), lr=.01), "patch", 1, 0,
                       cfg, losses.BaselineOptions("trades", trades_inner_loss="ce"),
                       None, torch.device("cpu"))
    assert calls == ["patch"]


@pytest.mark.parametrize("method", losses.METHODS)
@pytest.mark.parametrize("interrupt_stage,interrupt_epoch", [(0, 1), (2, 2)])
def test_four_stage_exact_resume_with_method_memory(
        method, interrupt_stage, interrupt_epoch, tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    options = losses.BaselineOptions(method, buffer_size=5)
    continuous = config(tmp_path / "continuous", method)
    interrupted = config(tmp_path / "interrupted", method)
    expected = runner.run(continuous, options)
    actual_save = runner.atomic_checkpoint
    def interrupt(path, value):
        actual_save(path, value)
        if (Path(path).name == "last.pt" and value["stage"] == interrupt_stage
                and value["next_epoch"] == interrupt_epoch):
            raise KeyboardInterrupt("Simulated epoch-boundary interruption")
    monkeypatch.setattr(runner, "atomic_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(interrupted, options)
    monkeypatch.setattr(runner, "atomic_checkpoint", actual_save)
    interrupted.resume = True
    actual = runner.run(interrupted, options)
    assert actual == expected
    left = torch.load(continuous.run_dir / "last.pt", map_location="cpu", weights_only=False)
    right = torch.load(interrupted.run_dir / "last.pt", map_location="cpu", weights_only=False)
    for name in ("model", "ema", "optimizer", "scheduler", "memory", "best_snapshot",
                 "epoch_rows", "best_score", "stage_summaries"):
        same(left[name], right[name])
    assert len(json.loads((interrupted.run_dir / "epoch_metrics.json").read_text())) == 8
    matrix = json.loads((interrupted.run_dir / "accuracy_matrix.json").read_text())
    assert [set(row) for row in matrix] == [set(runner.TASKS[:i+1]) for i in range(4)]
    monkeypatch.setattr(runner, "train_epoch", lambda *a, **k: pytest.fail("Unexpected retraining"))
    assert runner.run(interrupted, options, eval_only=True) == expected


def test_resume_rejects_batch_and_method_change_before_overwrite(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    cfg = config(tmp_path, "derpp")
    options = losses.BaselineOptions("derpp", buffer_size=5)
    runner.run(cfg, options)
    before = (cfg.run_dir / "baseline_config.json").read_bytes()
    cfg.resume = True
    cfg.batch_size = 4
    with pytest.raises(ValueError, match="mismatch"):
        runner.run(cfg, options)
    cfg.batch_size = 2
    with pytest.raises(ValueError, match="mismatch"):
        runner.run(cfg, losses.BaselineOptions("gsa", buffer_size=5))
    assert (cfg.run_dir / "baseline_config.json").read_bytes() == before


def test_selection_zero_values_and_memory_not_rewound(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    cfg = config(tmp_path, "derpp")
    options = losses.BaselineOptions("derpp", buffer_size=5)
    monkeypatch.setattr(runner, "evaluate_seen", lambda m,l,t,*a: {
        name: (80. if name == "clean" else 0.) for name in t})
    runner.run(cfg, options)
    best = torch.load(cfg.run_dir / "stage3_pgd_best.pt", map_location="cpu", weights_only=False)
    assert best["combined_metric"] == 20. and best["epoch"] == 2
    last = torch.load(cfg.run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["memory"]["num_seen"] == 4 * 2 * 2  # exposure stream is not rewound to best epoch.


@pytest.mark.parametrize("method", losses.METHODS)
def test_real_resnet18_imagenette_shape_step(method):
    args = runner.build_parser(method).parse_args([
        "--dataset", "imagenette", "--backbone", "resnet18", "--device", "cpu",
        "--batch-size", "2", "--workers", "0", "--train-attack-steps", "1",
        "--max-train-batches", "1"] + (["--buffer-size", "4"] if method in ("gsa", "derpp") else []))
    cfg, options = runner.resolve_config(method, args)
    torch.manual_seed(5)
    model = make_backbone(cfg)
    memory = runner.make_memory(cfg, options)
    images, labels = torch.rand(2, 3, 160, 160), torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    if memory is not None:
        memory.add(images, labels, 0, torch.zeros(2, 10) if method == "derpp" else None)
    stats = runner.train_epoch(model, EMA(model, .999), loader,
                               torch.optim.SGD(model.parameters(), lr=.01), "pgd_l2", 2, 0,
                               cfg, options, memory, torch.device("cpu"))
    assert stats["optimizer_updates"] == 1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
