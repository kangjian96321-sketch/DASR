"""CPU regression tests for final-evaluation batching and checkpoint lifetime.

The AutoAttack test double checks our adapter contract, not attack strength.
Real AutoAttack/CUDA peak memory must additionally be checked on a GPU host.
"""
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import types
import weakref

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pace_aligned import common, sseat
import train_replay_comparison as pace


class BatchGuard(torch.nn.Module):
    def __init__(self, maximum):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.maximum = maximum
        self.calls = []

    def forward(self, x):
        assert len(x) <= self.maximum, "Unbounded final forward would exhaust CUDA"
        self.calls.append((len(x), torch.is_grad_enabled()))
        return torch.cat((x * self.scale, -torch.ones(len(x), 8, device=x.device)), dim=1)


@pytest.fixture
def fake_autoattack(monkeypatch):
    calls = []

    class FakeAutoAttack:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def run_standard_evaluation(self, x, y, bs):
            assert x.device.type == y.device.type == "cpu"
            assert torch.is_grad_enabled()
            assert all(not p.requires_grad for p in self.model.parameters())
            # check_dynamic uses a forward without input gradients. It should
            # no longer build a training-sized graph for parameter gradients.
            output = self.model(x[:bs].to(self.kwargs["device"]))
            assert not output.requires_grad
            calls.append({**self.kwargs, "x": x.clone(), "y": y.clone(), "bs": bs})
            # Mimic official AutoAttack: move only the current batch to device.
            for start in range(0, len(x), bs):
                batch = x[start:start + bs].to(self.kwargs["device"]).clone().requires_grad_()
                logits = self.model(batch)
                grad, = torch.autograd.grad(logits[:, 0].sum(), batch)
                assert grad.abs().sum() > 0  # Attacks still have input gradients.
            return x.clone()

    monkeypatch.setitem(sys.modules, "autoattack", types.SimpleNamespace(AutoAttack=FakeAutoAttack))
    return calls


@pytest.mark.parametrize("norm,eps", [("L2", 2.5), ("Linf", 8 / 255)])
@pytest.mark.parametrize("requested", [0, 7, 100])
def test_autoattack_full_cpu_subset_and_bounded_final_forward(fake_autoattack, norm, eps, requested, tmp_path):
    predicted = torch.arange(11) % 2
    x = torch.nn.functional.one_hot(predicted, 2).float()
    y = predicted.clone()
    y[::3] = 1 - y[::3]
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    cfg = common.ExperimentConfig(method="pace", autoattack_samples=requested,
                                  autoattack_batch_size=3, autoattack_subset_seed=17,
                                  output_root=str(tmp_path))
    model = BatchGuard(3)
    model.scale.grad = torch.ones_like(model.scale)
    weights = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        accuracy = common.evaluate_autoattack(model, loader, norm, eps, cfg, torch.device("cpu"))
    count = len(x) if requested == 0 else min(requested, len(x))
    indices = common.autoattack_subset_indices(loader.dataset, cfg)
    expected = 100.0 * int(predicted[indices].eq(y[indices]).sum()) / count
    assert accuracy == pytest.approx(expected)
    assert len(fake_autoattack) == 1  # Not one independent AA invocation per batch.
    call = fake_autoattack[0]
    assert torch.equal(call["x"], x[indices])
    assert torch.equal(call["y"], y[indices])
    audit = json.loads((cfg.run_dir / "autoattack_subset.json").read_text())
    assert audit["indices"] == indices and audit["effective_samples"] == count
    assert audit["class_counts"]["0"] == int((y[indices] == 0).sum())
    assert audit["class_counts"]["1"] == int((y[indices] == 1).sum())
    assert audit["mode"] == ("full_test_set" if count == len(x) else "class_balanced_subset")
    assert (call["norm"], call["eps"], call["version"], call["bs"], call["device"]) == (
        norm, eps, "standard", 3, "cpu")
    assert call["seed"] == cfg.evaluation_seed
    final_calls = [size for size, grad in model.calls if not grad]
    assert sum(final_calls) == count and max(final_calls) <= 3
    assert final_calls[-1] == count % 3  # Both cases have a partial tail batch.
    assert model.scale.grad is None
    assert model.scale.requires_grad
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("field,value", [("autoattack_batch_size", 0),
                                        ("autoattack_batch_size", -1),
                                        ("autoattack_samples", -1)])
def test_autoattack_rejects_invalid_limits(field, value):
    cfg = common.ExperimentConfig(method="pace")
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="must be"):
        common.evaluate_autoattack(BatchGuard(2), None, "L2", .5, cfg, torch.device("cpu"))


def test_autoattack_rejects_empty_dataset(fake_autoattack):
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="dataset is empty"):
        common.evaluate_autoattack(BatchGuard(2), loader, "L2", .5,
                                   common.ExperimentConfig(method="pace"), torch.device("cpu"))


@pytest.mark.parametrize("method", ["sseat", "crt"])
def test_baseline_aa_batch_override_leaves_training_batch_unchanged(method):
    parser = importlib.import_module(f"pace_aligned.{method}").build_parser()
    args = parser.parse_args(["--dataset", "imagenette", "--autoattack-batch-size", "1",
                              "--autoattack-samples", "0"])
    cfg = common.config_from_args(method, args)
    assert cfg.batch_size == 32 and cfg.autoattack_batch_size == 1
    assert cfg.autoattack_samples == 0
    args.autoattack_batch_size = 0
    with pytest.raises(ValueError, match="must be positive"):
        common.config_from_args(method, args)


def test_pace_aa_override_preserves_training_identity(monkeypatch):
    captured = []
    monkeypatch.setattr(pace, "run_experiment", lambda cfg, *a, **kw: captured.append(cfg))
    base = ["--dataset", "imagenette", "--resume", "--eval-only",
            "--far-monitor-size", "500", "--far-history-size", "500"]
    pace.main(base)
    pace.main(base + ["--autoattack-batch-size", "1", "--autoattack-samples", "0"])
    old, new = captured
    assert old.batch_size == new.batch_size == 64
    assert old.autoattack_batch_size == 32 and new.autoattack_batch_size == 1
    assert old.autoattack_samples == 1000 and new.autoattack_samples == 0
    assert pace.protocol_settings(old, True, 1.0, False) == pace.protocol_settings(new, True, 1.0, False)
    with pytest.raises(ValueError, match="must be positive"):
        pace.main(base + ["--autoattack-batch-size", "0"])
    with pytest.raises(ValueError, match="must be nonnegative"):
        pace.main(base + ["--autoattack-samples", "-1"])


def test_frozen_flags_restored_after_attack_exception(monkeypatch, tmp_path):
    model = BatchGuard(3)
    model.register_parameter("already_frozen", torch.nn.Parameter(torch.ones(1), requires_grad=False))

    class FailingAttack:
        def __init__(self, attack_model, **kwargs):
            self.model = attack_model

        def run_standard_evaluation(self, x, y, bs):
            assert all(not p.requires_grad for p in self.model.parameters())
            raise RuntimeError("simulated interrupted attack")

    monkeypatch.setitem(sys.modules, "autoattack", types.SimpleNamespace(AutoAttack=FailingAttack))
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.ones(2, dtype=torch.long)))
    with pytest.raises(RuntimeError, match="simulated interrupted attack"):
        common.evaluate_autoattack(model, loader, "L2", .5,
                                   common.ExperimentConfig(method="pace", output_root=str(tmp_path)), torch.device("cpu"))
    assert model.scale.requires_grad and not model.already_frozen.requires_grad


def test_sseat_four_stage_cleanup_and_completed_resume(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(52)
        train = DataLoader(TensorDataset(torch.rand(4, 3, 32, 32),
                          torch.tensor([0, 1, 2, 3]), torch.arange(4)), batch_size=2)
        val = DataLoader(TensorDataset(torch.rand(4, 3, 32, 32),
                        torch.tensor([0, 1, 2, 3])), batch_size=2)
        bundle = common.DataBundle(train=train, train_eval=val, val=val, selection_val=val,
                                   test=val, train_indices=[0, 1, 2, 3], val_indices=[4, 5, 6, 7])
        monkeypatch.setattr(sseat, "build_dataloaders", lambda _: bundle)
        refs = []

        def track_factory(factory):
            def wrapped(*args, **kwargs):
                value = factory(*args, **kwargs)
                refs.append(weakref.ref(value))
                return value
            return wrapped

        for name in ("make_optimizer", "clone_frozen", "EMA"):
            monkeypatch.setattr(sseat, name, track_factory(getattr(sseat, name)))
        original_payload = sseat.checkpoint_payload
        momentum_refs = []

        def tracking_payload(*args, **kwargs):
            payload = original_payload(*args, **kwargs)
            momentum_refs.extend(weakref.ref(state["momentum_buffer"])
                                 for state in payload["optimizer"]["state"].values()
                                 if "momentum_buffer" in state)
            return payload

        monkeypatch.setattr(sseat, "checkpoint_payload", tracking_payload)
        original_final = sseat.final_evaluation

        def checked_final(*args, **kwargs):
            gc.collect()
            assert refs and all(ref() is None for ref in refs)
            assert momentum_refs and all(ref() is None for ref in momentum_refs)
            return original_final(*args, **kwargs)

        monkeypatch.setattr(sseat, "final_evaluation", checked_final)
        original_load = torch.load
        locations = []

        def tracking_load(*args, **kwargs):
            locations.append(kwargs.get("map_location"))
            return original_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", tracking_load)
        cfg = common.ExperimentConfig(
            method="sseat", dataset="cifar10", device="cpu", workers=0,
            output_root=str(tmp_path), run_name="sseat_oom_regression", batch_size=2,
            depth=10, widen_factor=1, dropout=0., task_epochs=dict.fromkeys(common.TASKS, 1),
            train_attack_steps=1, eval_attack_steps=1, max_train_batches=1, max_eval_batches=1,
            sseat_memory_size=8, sseat_replay_batch_size=2, sseat_uncertainty_views=1,
            final_test_methods=("clean",))
        results = sseat.run(cfg)
        assert set(results) == {"clean"}
        assert len(locations) == 4 and all(location == "cpu" for location in locations)
        cfg.resume = True
        monkeypatch.setattr(sseat, "train_epoch", lambda *a, **kw: pytest.fail("Unexpected retraining"))
        resumed = sseat.run(cfg)
        assert resumed == results
        assert len(locations) == 6 and all(location == "cpu" for location in locations)
    finally:
        torch.set_num_threads(previous_threads)
