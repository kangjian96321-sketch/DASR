"""Compatibility tests for the restored CRT and existing baseline entries."""

import importlib
import json
from pathlib import Path
import random
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pace_aligned import crt
from pace_aligned.common import DataBundle, ExperimentConfig, config_from_args


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", ["sseat", "crt"])
def test_baseline_import_and_imagenette_configuration(method):
    module = importlib.import_module(f"pace_aligned.{method}")
    parser = module.build_parser()
    args = parser.parse_args(["--dataset", "imagenette", "--dry-run"])
    cfg = config_from_args(method, args)
    assert cfg.image_size == 160
    assert cfg.val_size == 1000
    assert cfg.patch_size == 40
    assert cfg.l2_eps == 2.5
    assert cfg.l2_alpha == 0.5
    assert cfg.linf_eps == pytest.approx(8 / 255)
    assert cfg.selection_probe_size == 500  # Existing baseline protocol.
    assert cfg.autoattack_samples == 1000
    entry = importlib.import_module(f"train_{method}_pace_aligned")
    assert entry.main is module.main


def test_crt_attack_sampling_and_options():
    tasks = ("patch", "pgd_l2", "pgd")
    assert crt._choose_attack((), "clean", "croce", {}, {}) == "clean"
    assert crt._choose_attack(tasks, "pgd", "single", {}, {}) == "pgd"
    previous = random.getstate()
    try:
        random.seed(52)
        picks = [crt._choose_attack(tasks, "pgd", "croce",
                                   dict(zip(tasks, [99, 50, 0])),
                                   dict.fromkeys(tasks, 100)) for _ in range(1000)]
    finally:
        random.setstate(previous)
    assert picks.count("pgd") > picks.count("pgd_l2") > picks.count("patch")
    args = crt.build_parser().parse_args(["--no-crt-use-alr", "--crt-strategy", "single"])
    assert not args.crt_use_alr
    assert args.crt_strategy == "single"


@pytest.mark.parametrize("task", ["clean", "patch", "pgd_l2", "pgd"])
def test_alr_constraint_and_repeatability(task):
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 10))
    model.train()
    x = torch.rand(2, 3, 8, 8)
    cfg = ExperimentConfig(method="crt", device="cpu", patch_size=2)
    a = crt.generate_alr_view(model, x, task, cfg, steps=2, rng_seed=123)
    b = crt.generate_alr_view(model, x, task, cfg, steps=2, rng_seed=123)
    assert torch.equal(a, b)
    assert model.training
    assert not a.requires_grad
    assert torch.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1
    delta = a - x
    if task == "clean":
        assert torch.equal(a, x)
    elif task == "pgd":
        assert delta.abs().max() <= cfg.linf_eps + 1e-6
    elif task == "pgd_l2":
        assert (delta.flatten(1).norm(dim=1) <= cfg.l2_eps + 1e-6).all()
    else:
        assert ((delta.abs().sum(1) > 0).flatten(1).sum(1) <= 4).all()


def test_crt_four_stage_training_and_completed_resume(tmp_path, monkeypatch):
    torch.manual_seed(13)
    train = DataLoader(TensorDataset(torch.rand(4, 3, 32, 32),
                                    torch.tensor([0, 1, 2, 3]), torch.arange(4)), batch_size=2)
    val = DataLoader(TensorDataset(torch.rand(4, 3, 32, 32),
                                  torch.tensor([0, 1, 2, 3])), batch_size=2)
    bundle = DataBundle(train=train, train_eval=val, val=val, selection_val=val,
                        test=val, train_indices=[0, 1, 2, 3], val_indices=[4, 5, 6, 7])
    monkeypatch.setattr(crt, "build_dataloaders", lambda _: bundle)
    cfg = ExperimentConfig(method="crt", dataset="cifar10", device="cpu", workers=0,
                           output_root=str(tmp_path), run_name="synthetic_only", batch_size=2,
                           depth=10, widen_factor=1, dropout=0.0,
                           task_epochs=dict.fromkeys(("clean", "patch", "pgd_l2", "pgd"), 1),
                           train_attack_steps=1, eval_attack_steps=1,
                           max_train_batches=1, max_eval_batches=1,
                           final_test_methods=("clean", "patch", "pgd_l2", "pgd"))
    results = crt.run(cfg, crt.CRTOptions())
    assert set(results) == set(cfg.final_test_methods)
    assert all(0 <= v <= 100 for v in results.values())
    summaries = json.loads((cfg.run_dir / "stage_summaries.json").read_text())
    assert len(summaries) == 4
    assert summaries[-1]["known_attacks"] == ["patch", "pgd_l2", "pgd"]
    saved = torch.load(cfg.run_dir / "completed_stage.pt", map_location="cpu", weights_only=False)
    assert saved["stage"] == 3
    cfg.resume = True
    resumed = crt.run(cfg, crt.CRTOptions())
    assert results == resumed
    assert len(json.loads((cfg.run_dir / "stage_summaries.json").read_text())) == 4
