"""Real CPU ResNet-18 forwards/training; synthetic data are not experiments."""
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_aligned import common, crt, sseat, feature_plots
from my_functions.shared_base import UnifiedWideResNet, CIFAR10_MEAN, CIFAR10_STD
from my_functions.backbone_options import backbone_spec
from my_functions.resnet_base import UnifiedResNet18
import checkpoint_compat
import train_replay_comparison as pace


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", ["sseat", "crt"])
@pytest.mark.parametrize("argument", ["resnet18", "resnet-18", "wrn", "wrn-34-10"])
def test_shared_cli_selection(method, argument):
    module = importlib.import_module("pace_aligned." + method)
    args = module.build_parser().parse_args(["--dataset", "imagenette", "--backbone", argument])
    cfg = common.config_from_args(method, args)
    assert cfg.backbone == ("resnet18" if argument.startswith("resnet") else "wrn")
    assert cfg.autoattack_samples == 1000 and cfg.image_size == 160


def test_pace_cli_default_alias_and_invalid_choice(monkeypatch):
    captured = []
    monkeypatch.setattr(pace, "run_experiment", lambda cfg, *a, **kw: captured.append(cfg))
    pace.main([])
    pace.main(["--backbone", "resnet-18"])
    assert captured[0].backbone == "wrn" and (captured[0].depth, captured[0].widen_factor) == (34, 10)
    assert captured[1].backbone == "resnet18"
    with pytest.raises(SystemExit):
        pace.parse_args(["--backbone", "resnet50"])


def test_legacy_wrn_weights_forward_rng_unchanged_and_protocol_versioned():
    cfg = common.ExperimentConfig(method="pace", dataset="cifar10")
    # Selection/defaults changed intentionally; old runs must not match.
    assert pace.digest(pace.protocol_settings(cfg, True, 1., False)) != (
        "4ab213e893b3ffcbbffd86ecf746fac1b33fd0c187acc8ea9d7cc7154ac465d9")
    cfg.depth, cfg.widen_factor = 10, 1
    torch.manual_seed(71)
    actual = common.make_backbone(cfg).eval()
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(71)
    expected = UnifiedWideResNet(depth=10, widen_factor=1, num_classes=10, dropout_rate=.3,
                                dataset_mean=CIFAR10_MEAN, dataset_std=CIFAR10_STD).eval()
    assert torch.equal(torch.get_rng_state(), rng_after)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.state_dict().items())
    with torch.no_grad():
        x = torch.rand(2, 3, 32, 32)
        assert torch.equal(actual(x), expected(x))
    assert cfg.run_dir.name == "pace_cifar10_seed42_three_seed_resources"


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "tiny_imagenet", "imagenette"])
def test_resnet_shapes_stem_normalization_and_input_gradients(dataset):
    cfg = common.ExperimentConfig(method="pace", dataset=dataset, backbone="resnet18")
    model = common.make_backbone(cfg).eval()
    assert isinstance(model, UnifiedResNet18)
    small = dataset.startswith("cifar")
    assert model.conv1.kernel_size == ((3, 3) if small else (7, 7))
    assert isinstance(model.maxpool, torch.nn.Identity) == small
    assert [len(getattr(model, f"layer{i}")) for i in range(1, 5)] == [2, 2, 2, 2]
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
    x = torch.rand(1, 3, cfg.image_size, cfg.image_size, requires_grad=True)
    seen = []
    hook = model.conv1.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].detach().clone()))
    with common._frozen_model_parameters(model):
        logits, features, queries, extra = model(x, return_feature=True)
        gradient, = torch.autograd.grad(logits.sum(), x)
    hook.remove()
    assert torch.equal(seen[0], (x.detach() - model.mean) / model.std)
    assert logits.shape == (1, cfg.num_classes) and features.shape == (1, 512)
    assert features is queries and extra is None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_standard_imagenette_resnet_matches_torchvision_exactly():
    from torchvision.models import resnet18
    torch.manual_seed(73)
    cfg = common.ExperimentConfig(method="pace", dataset="imagenette", backbone="resnet18")
    wrapped = common.make_backbone(cfg).eval()
    torch.manual_seed(73)
    reference = resnet18(weights=None, num_classes=10).eval()
    with torch.no_grad():
        x = torch.rand(2, 3, 160, 160)
        assert torch.equal(wrapped(x), reference((x - wrapped.mean) / wrapped.std))


def test_backbone_identity_paths_metadata_and_checkpoint_rejection(tmp_path):
    wrn = common.ExperimentConfig(method="pace", dataset="cifar10", output_root=str(tmp_path), depth=10, widen_factor=1)
    resnet = copy.deepcopy(wrn)
    resnet.backbone = "resnet18"
    assert resnet.run_dir != wrn.run_dir and "_resnet18_" in resnet.run_dir.name
    assert pace.protocol_settings(wrn, True, 1., False) != pace.protocol_settings(resnet, True, 1., False)
    protocol = pace.protocol_settings(resnet, True, 1., False)
    resnet.depth, resnet.widen_factor, resnet.dropout = 28, 2, .5
    assert pace.protocol_settings(resnet, True, 1., False) == protocol
    assert common.config_to_dict(resnet)["resolved_backbone"]["feature_dim"] == 512
    with pytest.raises(ValueError, match="backbone"):
        checkpoint_compat.validated_state({"model": common.make_backbone(wrn).state_dict()}, common.make_backbone(resnet))
    common.prepare_run_directory(wrn)
    common.write_environment_manifest(wrn)
    path = wrn.run_dir / "config.json"
    before = path.read_bytes()
    resnet.run_directory, resnet.resume = str(wrn.run_dir), True
    with pytest.raises(ValueError, match="configuration mismatch|architecture/dataset mismatch"):
        common.prepare_run_directory(resnet)
    assert path.read_bytes() == before
    # New-protocol resumes require complete identity metadata.
    legacy_config = json.loads(before)
    legacy_config.pop("backbone")
    legacy_config.pop("resolved_backbone")
    common.save_json(path, legacy_config)
    wrn.resume = True
    with pytest.raises(ValueError, match="configuration mismatch"):
        common.prepare_run_directory(wrn)


def make_config(tmp_path, method):
    return common.ExperimentConfig(
        method=method, dataset="cifar10", backbone="resnet18", device="cpu", workers=0,
        output_root=str(tmp_path), run_name="synthetic_only", batch_size=2,
        task_epochs=dict.fromkeys(common.TASKS, 1), train_attack_steps=1, eval_attack_steps=1,
        cw_eval_steps=1, max_train_batches=1, max_eval_batches=1,
        sseat_memory_size=8, sseat_replay_batch_size=2, sseat_uncertainty_views=2,
        final_test_methods=("clean", "patch", "pgd_l2", "pgd"))


@pytest.mark.parametrize("method", ["sseat", "crt"])
def test_resnet_baselines_four_stage_training_and_resume(tmp_path, monkeypatch, method):
    import shutil
    if shutil.disk_usage(tmp_path).free < 2 * 1024**3:
        pytest.skip('Full ResNet-18 multi-stage checkpoint test requires 2 GiB free disk')
    module = {"sseat": sseat, "crt": crt}[method]
    cfg = make_config(tmp_path, method)
    bundle = pace.synthetic_loaders(cfg)
    monkeypatch.setattr(module, "build_dataloaders", lambda _: bundle)
    invoke = (lambda: sseat.run(cfg)) if method == "sseat" else (lambda: crt.run(cfg, crt.CRTOptions()))
    results = invoke()
    assert set(results) == set(cfg.final_test_methods)
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in results.values())
    checkpoint = torch.load(cfg.run_dir / "completed_stage.pt", weights_only=False, map_location="cpu")
    assert checkpoint["stage"] == 3 and "layer4.0.conv1.weight" in checkpoint["model"]
    cfg.resume = True
    monkeypatch.setattr(module, "train_epoch", lambda *a, **kw: pytest.fail("Unexpected retraining"))
    assert invoke() == results


def test_resnet_pace_fg_lazy_four_stages_queue_and_resume(tmp_path, monkeypatch):
    import shutil
    if shutil.disk_usage(tmp_path).free < 2 * 1024**3:
        pytest.skip('Full ResNet-18 multi-stage checkpoint test requires 2 GiB free disk')
    import far_replay
    from ablation_config import method_name
    cfg = make_config(tmp_path, method_name("uniform", "full"))
    cfg.task_epochs = dict.fromkeys(common.TASKS, 2)
    original_decompose = far_replay.decompose

    def force_cache_branch(*args, **kwargs):
        metrics = original_decompose(*args, **kwargs)
        metrics.update(historical_probability=1., regenerated_probability=0.)
        return metrics

    monkeypatch.setattr(far_replay, "decompose", force_cache_branch)
    args = dict(policy="uniform", smoke=True, replay_source_mode="fg_adaptive_lazy",
                far_monitor_size=4, far_history_size=4)
    pace.run_experiment(cfg, **args)
    checkpoint = torch.load(cfg.run_dir / "stage3_pgd_resume.pt", weights_only=False, map_location="cpu")
    assert checkpoint["queue"]["features"].shape[1] == 512
    assert "layer4.0.conv1.weight" in checkpoint["teacher"]
    rows = json.loads((cfg.run_dir / "epoch_metrics.json").read_text())
    assert any(row["historical_replay_examples"] > 0 for row in rows)
    assert any(row["lazy_cache_builds"] > 0 for row in rows)
    assert all(math.isfinite(row["loss"]) for row in rows)
    before = (cfg.run_dir / "final_test.json").read_bytes()
    cfg.resume = True
    monkeypatch.setattr(pace, "train_epoch", lambda *a, **kw: pytest.fail("Unexpected retraining"))
    pace.run_experiment(cfg, eval_only=True, **args)
    assert (cfg.run_dir / "final_test.json").read_bytes() == before


def test_resnet_aa_balance_feature_cache_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, "pace")
    cfg.max_eval_batches = None
    cfg.final_test_methods = ("clean", "autoattack_l2", "autoattack")
    cfg.plot_final_features, cfg.autoattack_samples, cfg.autoattack_batch_size = True, 10, 2
    model = common.make_backbone(cfg).eval()
    labels = torch.arange(20) % 10
    loader = DataLoader(TensorDataset(torch.rand(20, 3, 32, 32), labels), batch_size=2)
    bundle = common.DataBundle(loader, loader, loader, loader, loader, [], [])
    calls = []

    class FakeAA:
        def __init__(self, net, **kwargs):
            self.net, self.norm = net, kwargs["norm"]

        def run_standard_evaluation(self, x, y, bs):
            assert x.device.type == "cpu" and bs == 2
            assert torch.bincount(y, minlength=10).tolist() == [1] * 10
            inputs = x[:bs].clone().requires_grad_()
            gradient, = torch.autograd.grad(self.net(inputs).sum(), inputs)
            assert gradient.abs().sum() > 0
            calls.append((self.norm, x.clone()))
            return (x + .001).clamp(0, 1)

    monkeypatch.setitem(sys.modules, "autoattack", types.SimpleNamespace(AutoAttack=FakeAA))
    monkeypatch.setattr(feature_plots.FeatureStore, "render", lambda *a: None)
    results = common.final_evaluation(model, bundle, cfg, torch.device("cpu"))
    assert [entry[0] for entry in calls] == ["L2", "Linf"]
    assert torch.equal(calls[0][1], calls[1][1])
    _, data = feature_plots.load_feature_cache(cfg.run_dir)
    assert all(item["features"].shape[1] == 512 for item in data.values())
    assert data["autoattack"]["ids"].tolist() == data["autoattack_l2"]["ids"].tolist()
    assert common.final_evaluation(model, bundle, cfg, torch.device("cpu")) == results
    assert len(calls) == 2
    restored = common.make_backbone(cfg).eval()
    restored.load_state_dict(checkpoint_compat.validated_state({"model": model.state_dict()}, restored))
    with torch.no_grad():
        x = loader.dataset.tensors[0][:2]
        assert torch.equal(restored(x), model(x))


@pytest.mark.parametrize("launcher", ["run_ablations", "run_far_source_modes"])
def test_batch_launcher_resnet_resume_paths(tmp_path, launcher):
    from argparse import Namespace
    module = importlib.import_module(launcher)
    args = Namespace(seeds=[42], ablations=["full"], ablation="full", checkpoint_map=None,
                     output_root=str(tmp_path), dataset="cifar10", run_name="", data_root="./data",
                     policy="dtr", resume=True, modes=["fg_adaptive_lazy"], resume_mode="epoch",
                     backbone="resnet18", replay_source_mode="fg_adaptive_lazy")
    cfg = common.ExperimentConfig(method="pace_dtr_fg_adaptive_lazy", dataset="cifar10",
                                  backbone="resnet18", output_root=str(tmp_path), run_name="")
    cfg.run_dir.mkdir()
    (cfg.run_dir / "stub.pt").touch()
    commands = module.build_commands(args, [])
    assert len(commands) == 1 and "--resume" in commands[0]
    assert commands[0][commands[0].index("--backbone") + 1] == "resnet18"


def test_matched_study_launcher_resnet_anchor_and_resume(tmp_path, monkeypatch):
    import run_replay_study
    cfg = common.ExperimentConfig(method="pace_dtr_fg_adaptive_lazy", dataset="cifar10",
                                  backbone="resnet18", output_root=str(tmp_path), run_name="study")
    cfg.run_dir.mkdir()
    (cfg.run_dir / "stub.pt").touch()
    commands = []
    monkeypatch.setattr(run_replay_study.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(sys, "argv", ["run_replay_study.py", "--dataset", "cifar10", "--seeds", "42",
                                     "--output-root", str(tmp_path), "--run-name", "study",
                                     "--backbone", "resnet18", "--resume"])
    run_replay_study.main()
    assert len(commands) == 4 and "--resume" in commands[0]
    for command in commands:
        assert command[command.index("--backbone") + 1] == "resnet18"
    for command in commands[1:]:
        assert Path(command[command.index("--match-schedule") + 1]) == cfg.run_dir / "replay_schedule.json"
