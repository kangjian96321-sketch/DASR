"""Actual CPU feature collection/plotting; AutoAttack is stubbed only in its test."""
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_aligned import common
from pace_aligned import feature_plots as plots
import train_replay_comparison as pace


class TinyFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 10)

    def forward(self, x, return_feature=False):
        features = x.flatten(1)
        logits = self.linear(features)
        return (logits, features, features, None) if return_feature else logits


def config(tmp_path):
    return common.ExperimentConfig(method="pace", dataset="cifar10", device="cpu", workers=0,
                                   output_root=str(tmp_path), run_name="features", batch_size=3,
                                   autoattack_batch_size=2, autoattack_samples=5,
                                   final_test_methods=("clean", "pgd"), plot_final_features=True,
                                   tsne_samples=0, tsne_iterations=300)


def test_paired_distance_and_permuted_attack_ids():
    clean = {"ids": np.array([0, 1, 2]), "labels": np.array([0, 1, 2])}
    attack = {"ids": np.array([2, 0]), "labels": np.array([2, 0])}
    ids, ci, ai = plots.pair_features(clean, attack)
    assert ids.tolist() == [0, 2] and ci.tolist() == [0, 2] and ai.tolist() == [1, 0]
    a, b = np.array([[3., 4.], [1., 0.]]), np.array([[0., 0.], [0., 1.]])
    assert plots.feature_distances(a, b, "l2") == pytest.approx([5., 2. ** .5])
    assert plots.feature_distances(a[1:], b[1:], "normalized_l2") == pytest.approx([2. ** .5])
    assert plots.feature_distances(a[1:], b[1:], "cosine") == pytest.approx([1.])
    with pytest.raises(ValueError, match="undefined"):
        plots.feature_distances(a, b, "cosine")
    attack["labels"][0] = 3
    with pytest.raises(ValueError, match="Mismatched labels"):
        plots.pair_features(clean, attack)


def test_balanced_sample_and_all_mode():
    ids, labels = np.arange(15), np.repeat([0, 1, 2], [3, 5, 7])
    selected = plots.balanced_ids(ids, labels, 9, 42)
    assert np.bincount(labels[selected]).tolist() == [3, 3, 3]
    assert np.array_equal(selected, plots.balanced_ids(ids, labels, 9, 42))
    assert np.array_equal(plots.balanced_ids(ids, labels, 0, 42), ids)
    with pytest.raises(ValueError, match="all 3"):
        plots.balanced_ids(ids, labels, 2, 42)


def test_regular_attack_reused_and_cached(tmp_path, monkeypatch):
    torch.manual_seed(4)
    cfg = config(tmp_path)
    model = TinyFeatureModel()
    loader = DataLoader(TensorDataset(torch.rand(7, 3, 2, 2), torch.arange(7)), batch_size=3)
    bundle = common.DataBundle(loader, loader, loader, loader, loader, [], [])
    calls = []

    def attack(model, images, labels, task, cfg, training):
        calls.append(task)
        return images if task == "clean" else (images + .05).clamp(0, 1)

    monkeypatch.setattr(common, "generate_attack", attack)
    monkeypatch.setattr(plots.FeatureStore, "render", lambda *args: None)
    results = common.final_evaluation(model, bundle, cfg, torch.device("cpu"))
    assert calls == ["clean"] * 3 + ["pgd"] * 3
    manifest, data = plots.load_feature_cache(cfg.run_dir)
    assert data["pgd"]["ids"].tolist() == list(range(7))
    assert np.all(data["pgd"]["features"] >= data["clean"]["features"])
    assert manifest["conditions"]["pgd"]["accuracy_percent"] == results["pgd"]
    monkeypatch.setattr(common, "generate_attack", lambda *a, **kw: pytest.fail("Attack regenerated despite valid cache"))
    assert common.final_evaluation(model, bundle, cfg, torch.device("cpu")) == results
    with torch.no_grad():
        model.linear.weight.add_(1.)
    changed = plots.make_feature_store(model, loader, cfg)
    assert not changed.has("clean") and not changed.has("pgd")


def test_feature_path_does_not_change_accuracy(tmp_path, monkeypatch):
    model = TinyFeatureModel()
    loader = DataLoader(TensorDataset(torch.rand(7, 3, 2, 2), torch.arange(7)), batch_size=3)
    cfg = config(tmp_path)
    monkeypatch.setattr(common, "generate_attack", lambda model, x, *a, **kw: x + .02)
    original = common.evaluate_accuracy(model, loader, "pgd", cfg, torch.device("cpu"))
    store = plots.make_feature_store(model, loader, cfg)
    captured = common.evaluate_accuracy(model, loader, "pgd", cfg, torch.device("cpu"), store.collector("pgd"))
    assert original == captured
    shuffled = DataLoader(loader.dataset, batch_size=3, shuffle=True)
    with pytest.raises(ValueError, match="sequential"):
        common.evaluate_accuracy(model, shuffled, "pgd", cfg, torch.device("cpu"), store.collector("pgd"))


def test_aa_subset_ids_match_clean_without_second_attack(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.final_test_methods = ("clean", "autoattack_l2", "autoattack")
    model = TinyFeatureModel()
    loader = DataLoader(TensorDataset(torch.rand(9, 3, 2, 2), torch.arange(9)), batch_size=3)
    bundle = common.DataBundle(loader, loader, loader, loader, loader, [], [])
    calls = []

    class FakeAA:
        def __init__(self, model, **kwargs):
            self.norm = kwargs["norm"]

        def run_standard_evaluation(self, x, y, bs):
            calls.append(self.norm)
            assert bs == 2 and x.device.type == "cpu"
            return (x + .01).clamp(0, 1)

    monkeypatch.setitem(sys.modules, "autoattack", types.SimpleNamespace(AutoAttack=FakeAA))
    monkeypatch.setattr(plots.FeatureStore, "render", lambda *a: None)
    common.final_evaluation(model, bundle, cfg, torch.device("cpu"))
    assert calls == ["L2", "Linf"]
    _, data = plots.load_feature_cache(cfg.run_dir)
    expected = sorted(common.autoattack_subset_indices(loader.dataset, cfg))
    assert data["autoattack"]["ids"].tolist() == expected
    assert data["autoattack_l2"]["ids"].tolist() == expected
    ids, ci, ai = plots.pair_features(data["clean"], data["autoattack"])
    assert np.allclose(data["autoattack"]["features"][ai], np.minimum(data["clean"]["features"][ci] + .01, 1.))
    common.final_evaluation(model, bundle, cfg, torch.device("cpu"))
    assert calls == ["L2", "Linf"]


def test_plot_options_do_not_change_training_identity(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    first = pace.protocol_settings(cfg, True, 1., False)
    cfg.plot_final_features = False
    cfg.tsne_samples = 99
    cfg.feature_distance = "cosine"
    assert pace.protocol_settings(cfg, True, 1., False) == first
    captured = []
    monkeypatch.setattr(pace, "run_experiment", lambda cfg, *a, **kw: captured.append(cfg))
    pace.main(["--dataset", "imagenette", "--plot-final-features", "--tsne-samples", "0"])
    assert captured[0].plot_final_features and captured[0].tsne_samples == 0
    from pace_aligned import sseat
    args = sseat.build_parser().parse_args(["--plot-final-features", "--feature-distance", "normalized_l2"])
    assert common.config_from_args("sseat", args).feature_distance == "normalized_l2"


def make_synthetic_cache(path, classes=3, per_class=12):
    """Only for plot QA; never report these as experiment measurements."""
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(classes), per_class)
    ids = np.arange(len(labels))
    centers = rng.normal(0, 4, (classes, 12))
    clean = centers[labels] + rng.normal(0, .3, (len(labels), 12))
    methods = list(plots.DISPLAY)
    store = plots.FeatureStore(path, "synthetic-test-only", {
        "evaluation_seed": 42, "feature_layer": "synthetic_12d_features_NOT_EXPERIMENT_RESULTS"}, methods)
    for j, method in enumerate(methods):
        features = clean + rng.normal(0, .08 * j, clean.shape)
        collector = store.collector(method)
        predictions = labels.copy()
        predictions[:j] = (predictions[:j] + 1) % classes
        collector.add(ids, torch.tensor(labels), torch.tensor(features), torch.tensor(predictions))
        collector.finish(100. * np.mean(labels == predictions))
    return store


def test_real_plotting_all_conditions_and_coordinate_reuse(tmp_path, monkeypatch):
    make_synthetic_cache(tmp_path)
    calls = []
    real_fit = plots._fit_tsne

    def fit(*a, **kw):
        calls.append(1)
        return real_fit(*a, **kw)

    monkeypatch.setattr(plots, "_fit_tsne", fit)
    output = plots.plot_cached_features(tmp_path, samples=12, iterations=300, bins=15, boundary_overlays=False)
    metadata = json.loads((output / "plot_manifest.json").read_text())
    assert calls == [1, 1]
    assert metadata["class_counts_per_condition"] == {"0": 4, "1": 4, "2": 4}
    assert metadata["tsne"]["2d"]["total_points"] == 12 * 7
    assert metadata["tsne"]["3d"]["total_points"] == 12 * 7
    assert len(metadata["markers"]) == len(set(metadata["markers"].values()))
    assert len(metadata["figures"]) == 48  # 7 distance + 16 scatter + 1 legend, PNG/PDF.
    for name in metadata["figures"]:
        assert (output / name).stat().st_size > 500
    with np.load(output / "tsne_2d_coordinates.npz") as cache:
        assert cache["coordinates"].shape == (84, 2)
    with np.load(output / "tsne_3d_coordinates.npz") as cache:
        assert cache["coordinates"].shape == (84, 3)
    plots.plot_cached_features(tmp_path, samples=12, iterations=300, bins=20, metric="normalized_l2", boundary_overlays=False)
    assert calls == [1, 1]  # Only bins/distance changed: no additional fits or attacks.


def test_joint_sampling_uses_all_condition_intersection():
    clean = {"ids": np.arange(6), "labels": np.array([0, 0, 1, 1, 2, 2]), "features": np.eye(6)}
    attacked = {key: value[[1, 3, 5]] for key, value in clean.items()}
    methods, ids, labels, features, settings = plots.joint_embedding(
        {"clean": clean, "autoattack": attacked}, 0, 42, 30, 300)
    assert ids.tolist() == [1, 3, 5]
    assert features.shape == (6, 6) and settings["perplexity"] == 5.


def test_partial_cache_is_not_plotted(tmp_path):
    plots.FeatureStore(tmp_path, "test", {}, ["clean", "pgd"])
    with pytest.raises(ValueError, match="Incomplete"):
        plots.plot_cached_features(tmp_path)


def test_pace_accuracy_only_run_can_add_features_without_retraining(tmp_path, monkeypatch):
    from ablation_config import method_name
    cfg = common.ExperimentConfig(
        method=method_name("uniform", "full"), dataset="cifar10", device="cpu", workers=0,
        output_root=str(tmp_path), run_name="resume_feature_test", depth=10, widen_factor=1,
        batch_size=4, train_attack_steps=1, eval_attack_steps=1, cw_eval_steps=1,
        task_epochs=dict.fromkeys(common.TASKS, 1), max_train_batches=1, max_eval_batches=1,
        final_test_methods=("clean", "pgd"))
    pace.run_experiment(cfg, policy="uniform", smoke=True)
    initial = json.loads((cfg.run_dir / "final_test.json").read_text())
    cfg.resume = True
    cfg.plot_final_features = True
    monkeypatch.setattr(pace, "train_epoch", lambda *a, **kw: pytest.fail("Unexpected retraining"))
    monkeypatch.setattr(plots.FeatureStore, "render", lambda *a: None)
    pace.run_experiment(cfg, policy="uniform", smoke=True, eval_only=True)
    _, data = plots.load_feature_cache(cfg.run_dir)
    assert set(data) == {"clean", "pgd"}
    assert json.loads((cfg.run_dir / "final_test.json").read_text()) == initial
    timestamps = {p.name: p.stat().st_mtime_ns for p in (cfg.run_dir / "final_features").glob("*.npz")}
    pace.run_experiment(cfg, policy="uniform", smoke=True, eval_only=True)
    assert timestamps == {p.name: p.stat().st_mtime_ns for p in (cfg.run_dir / "final_features").glob("*.npz")}


@pytest.mark.parametrize("ndim", [2, 3])
def test_projected_surrogate_targets_and_grouped_holdout(ndim):
    rng = np.random.default_rng(8)
    points = rng.normal(size=(40, ndim))
    predictions = (points[:, 0] > 0).astype(int)
    ids = np.tile(np.arange(20), 2)
    bounds = [(-3., 3.)] * ndim
    regions = plots.projected_regions(points, predictions, ids, bounds, 42, grid_size=12)
    assert regions["codes"].shape == (12,) * ndim
    assert regions["report"]["held_out_image_ids"] == 4
    assert 0. <= regions["report"]["held_out_image_id_agreement"] <= 1.
    assert regions["report"]["warning"].startswith("NOT")
    if ndim == 3:
        assert len(regions["faces"]) > 0
