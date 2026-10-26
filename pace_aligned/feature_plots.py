"""Paired final-test feature caches and CPU-only scientific plots.

No adversarial image cache and no attack generation live in this module.
Every attack feature must come from the actual final-accuracy evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch


PLOT_FIELDS = ("plot_final_features", "tsne_samples", "tsne_perplexity",
               "tsne_iterations", "feature_distance", "feature_bins", "tsne_dimensions",
               "feature_boundary_overlays")
DISPLAY = {"clean": "Clean", "patch": "Patch", "cw_l2": "CW-L2",
           "pgd_l2": "PGD-L2", "pgd": "PGD-Linf",
           "autoattack_l2": "AA-L2", "autoattack": "AA-Linf"}
MARKERS = {"clean": ".", "patch": "^", "cw_l2": "s", "pgd_l2": "v",
           "pgd": "x", "autoattack_l2": "D", "autoattack": "P"}
ATTACK_COLORS = {"clean": "#737373", "patch": "#0072B2", "cw_l2": "#E69F00",
                 "pgd_l2": "#009E73", "pgd": "#D55E00",
                 "autoattack_l2": "#CC79A7", "autoattack": "#6F4E9C"}


def add_feature_arguments(parser, include_enable=True):
    if include_enable:
        parser.add_argument("--plot-final-features", action="store_true",
                            help="Cache final-test features and draw all/per-attack plots")
    parser.add_argument("--tsne-samples", type=int, default=1000,
                        help="Class-balanced matched clean IDs for t-SNE; 0 = all common IDs")
    parser.add_argument("--tsne-perplexity", type=float, default=30.)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--tsne-dimensions", choices=("both", "2", "3"), default="both")
    parser.add_argument("--feature-boundary-overlays", action=argparse.BooleanOptionalAction, default=True,
                        help="Extra projected-surrogate boundary and correct/error distribution copies (not true model boundaries)")
    parser.add_argument("--feature-distance", choices=("l2", "normalized_l2", "cosine"), default="l2")
    parser.add_argument("--feature-bins", type=int, default=60)


def apply_feature_arguments(cfg, args):
    for field in PLOT_FIELDS:
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))
    validate_options(cfg.tsne_samples, cfg.tsne_perplexity, cfg.tsne_iterations,
                     cfg.feature_distance, cfg.feature_bins)


def validate_options(samples, perplexity, iterations, metric, bins):
    if samples < 0 or not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("t-SNE samples must be nonnegative and perplexity finite/positive")
    if iterations < 250 or bins < 2:
        raise ValueError("t-SNE iterations must be >= 250 and feature bins >= 2")
    if metric not in ("l2", "normalized_l2", "cosine"):
        raise ValueError(f"Unknown feature distance: {metric}")


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _npz_write(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def forward_with_features(model, images):
    """WRN/ResNet: pooled backbone feature before the classifier (not logits)."""
    output = model(images, return_feature=True)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("Feature plotting needs model(x, return_feature=True) -> (logits, features, ...)")
    logits, features = output[:2]
    if features.ndim != 2 or features.shape[0] != images.shape[0]:
        raise ValueError("Expected one 2D pooled feature vector per image")
    return logits, features


def _model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_feature_store(model, loader, cfg):
    """Strict evaluation fingerprint; plotting style never changes training identity."""
    from .common import AUTOATTACK_SAMPLING_POLICY
    fields = ("dataset", "evaluation_seed", "autoattack_subset_seed", "autoattack_samples",
              "autoattack_batch_size", "batch_size", "eval_attack_steps", "cw_eval_steps",
              "linf_eps", "linf_alpha", "l2_eps", "l2_alpha", "patch_size", "patch_alpha",
              "max_eval_batches", "num_classes")
    spec = {field: getattr(cfg, field) for field in fields}
    spec["autoattack_sampling_policy"] = AUTOATTACK_SAMPLING_POLICY
    spec.update(schema=1, model_sha256=_model_hash(model),
                data_root=str(Path(cfg.data_root).resolve()), test_size=len(loader.dataset),
                feature_layer="pooled_backbone_before_classifier", feature_dtype="float32",
                model_type=type(model).__qualname__,
                seen_stages=getattr(model, "seen_stages", None),
                preprocessing=repr(getattr(loader.dataset, "transform", None)))
    # Include test ordering when the dataset exposes its file list / labels.
    samples = getattr(loader.dataset, "samples", None)
    targets = getattr(loader.dataset, "targets", None)
    ordering = samples if samples is not None else (list(map(int, targets)) if targets is not None else None)
    if ordering is not None:
        spec["ordering_sha256"] = hashlib.sha256(json.dumps(ordering, default=str).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    methods = list(dict.fromkeys(("clean", *cfg.final_test_methods)))
    return FeatureStore(cfg.run_dir, signature, spec, methods)


class FeatureStore:
    def __init__(self, run_dir, signature, spec, methods):
        self.directory = Path(run_dir) / "final_features"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "manifest.json"
        self.signature = signature
        self.methods = list(methods)
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.manifest = old if old.get("signature") == signature else {
            "signature": signature, "spec": spec, "conditions": {}}
        self.manifest["methods"] = self.methods
        _json_write(self.path, self.manifest)

    def _path(self, method):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", method):
            raise ValueError("Invalid feature-cache method name")
        return self.directory / f"{method}.npz"

    def has(self, method):
        info = self.manifest["conditions"].get(method)
        if not info or not self._path(method).is_file():
            return False
        try:
            with np.load(self._path(method), allow_pickle=False) as cache:
                return str(cache["signature"].item()) == self.signature and len(cache["ids"]) == info["count"]
        except (ValueError, OSError, KeyError, EOFError):
            return False

    def accuracy(self, method):
        return float(self.manifest["conditions"][method]["accuracy_percent"])

    def collector(self, method):
        return FeatureCollector(self, method)

    def render(self, cfg):
        return plot_cached_features(self.directory.parent, samples=cfg.tsne_samples,
                                    perplexity=cfg.tsne_perplexity, iterations=cfg.tsne_iterations,
                                    metric=cfg.feature_distance, bins=cfg.feature_bins,
                                    seed=cfg.evaluation_seed, dimensions=cfg.tsne_dimensions,
                                    boundary_overlays=cfg.feature_boundary_overlays)


class FeatureCollector:
    def __init__(self, store, method):
        self.store, self.method = store, method
        self.parts = []

    def add(self, ids, labels, features, predictions):
        self.parts.append((np.asarray(ids, dtype=np.int64), labels.detach().cpu().numpy().copy(),
                           features.detach().float().cpu().numpy().copy(),
                           predictions.detach().cpu().numpy().copy()))

    def finish(self, accuracy):
        if not self.parts:
            raise ValueError(f"No evaluated features for {self.method}")
        ids, labels, features, predictions = [np.concatenate(items) for items in zip(*self.parts)]
        order = np.argsort(ids)
        ids, labels, features, predictions = [x[order] for x in (ids, labels, features, predictions)]
        if len(np.unique(ids)) != len(ids) or not np.isfinite(features).all():
            raise ValueError("Duplicate sample IDs or non-finite features")
        measured = 100. * np.mean(labels == predictions)
        if not math.isclose(measured, accuracy, abs_tol=1e-6):
            raise ValueError("Feature predictions do not match final-test accuracy")
        _npz_write(self.store._path(self.method), ids=ids, labels=labels, features=features,
                   predictions=predictions, signature=np.asarray(self.store.signature))
        self.store.manifest["conditions"][self.method] = {
            "count": len(ids), "dimension": int(features.shape[1]),
            "accuracy_percent": float(accuracy), "file": f"{self.method}.npz"}
        _json_write(self.store.path, self.store.manifest)
        self.parts.clear()


def load_feature_cache(run_dir):
    directory = Path(run_dir) / "final_features"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No final_features/manifest.json; first run final evaluation with --plot-final-features")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = {}
    for method in manifest["methods"]:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", method):
            raise ValueError("Invalid cached method name")
        path = directory / f"{method}.npz"
        if method not in manifest["conditions"] or not path.exists():
            raise ValueError(f"Incomplete feature cache: {method}; resume final evaluation first")
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != manifest["signature"]:
                raise ValueError(f"Stale feature cache: {method}")
            data[method] = {key: cache[key] for key in ("ids", "labels", "features", "predictions")}
        item = data[method]
        if len(np.unique(item["ids"])) != len(item["ids"]) or not np.isfinite(item["features"]).all():
            raise ValueError(f"Invalid feature arrays: {method}")
    if "clean" not in data:
        raise ValueError("Clean features are required for paired analysis")
    return manifest, data


def pair_features(clean, attacked):
    ids, ci, ai = np.intersect1d(clean["ids"], attacked["ids"], return_indices=True)
    if len(ids) != len(attacked["ids"]):
        raise ValueError("Some attacked IDs have no clean feature; cannot pair by batch position")
    if not np.array_equal(clean["labels"][ci], attacked["labels"][ai]):
        raise ValueError("Mismatched labels for the same clean/attack sample ID")
    return ids, ci, ai


def feature_distances(clean, attacked, metric):
    # Float64 accumulation avoids cancellation in the cosine computation.
    a, b = np.asarray(clean, dtype=np.float64), np.asarray(attacked, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired feature shapes differ")
    if metric == "l2":
        return np.linalg.norm(a - b, axis=1)
    norms_a, norms_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    a = a / np.maximum(norms_a[:, None], 1e-12)
    b = b / np.maximum(norms_b[:, None], 1e-12)
    if metric == "normalized_l2":
        return np.linalg.norm(a - b, axis=1)
    if metric == "cosine":
        if np.any(norms_a <= 1e-12) or np.any(norms_b <= 1e-12):
            raise ValueError("Cosine distance is undefined for zero-norm features; use l2 instead")
        return np.clip(1. - np.sum(a * b, axis=1), 0., 2.)
    raise ValueError(metric)


def balanced_ids(ids, labels, samples, seed):
    if samples == 0 or samples >= len(ids):
        return np.sort(ids)
    classes = np.unique(labels)
    if samples < len(classes):
        raise ValueError(f"--tsne-samples must cover all {len(classes)} present classes (or use 0)")
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(ids[labels == label])) for label in classes]
    chosen = []
    while len(chosen) < samples:
        for group in groups:
            if group and len(chosen) < samples:
                chosen.append(group.pop())
    return np.sort(chosen)


def joint_embedding(data, samples, seed, perplexity, iterations):
    """Fit once to all conditions, never refit individual attack panels."""
    methods = list(data)
    common = data["clean"]["ids"]
    for method in methods:
        common = np.intersect1d(common, data[method]["ids"])
    if not len(common):
        raise ValueError("No test IDs are shared by all evaluated attacks")
    lookup = {int(v): i for i, v in enumerate(data["clean"]["ids"])}
    labels = data["clean"]["labels"][[lookup[int(v)] for v in common]]
    ids = balanced_ids(common, labels, samples, seed)
    labels = data["clean"]["labels"][[lookup[int(v)] for v in ids]]
    matrices = []
    for method in methods:
        index = {int(v): i for i, v in enumerate(data[method]["ids"])}
        positions = [index[int(v)] for v in ids]
        if not np.array_equal(data[method]["labels"][positions], labels):
            raise ValueError("Inconsistent labels in t-SNE alignment")
        matrices.append(data[method]["features"][positions])
    stacked = np.concatenate(matrices).astype(np.float32)
    if min(stacked.shape) < 2 or len(stacked) < 3:
        raise ValueError("t-SNE needs >= 3 observations and >= 2 feature dimensions")
    return methods, ids, labels, stacked, {
        "perplexity": min(float(perplexity), float(len(stacked) - 1)),
        "iterations": iterations, "seed": seed, "common_available": len(common),
        "images_per_condition": len(ids), "total_points": len(stacked),
        "feature_space": "raw pooled backbone features; PCA at most 50 dimensions",
    }


def _fit_tsne(features, settings):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    components = settings.get("dimensions", 2)
    options = dict(n_components=components, perplexity=settings["perplexity"], init="pca",
                   learning_rate="auto", random_state=settings["seed"], n_jobs=1)
    # sklearn 1.5 renamed n_iter to max_iter; support the existing >=1.3 env.
    key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    options[key] = settings["iterations"]
    with threadpool_limits(limits=1):
        if min(features.shape[1], len(features) - 1) < components:
            raise ValueError(f"{components}D t-SNE needs >= {components} feature dimensions and > {components} observations")
        dimensions = min(50, features.shape[1], len(features) - 1)
        if dimensions < features.shape[1]:
            features = PCA(n_components=dimensions, random_state=settings["seed"]).fit_transform(features)
        # A constant feature matrix has no meaningful t-SNE; report, not invent a plot.
        if not np.any(np.var(features, axis=0) > 0):
            raise ValueError("All cached features are identical; t-SNE is not informative")
        return TSNE(**options).fit_transform(features)


def projected_regions(coordinates, predictions, anchor_ids, bounds, seed, grid_size=None):
    """Visualization-only kNN surrogate of model predictions in a t-SNE map.

    Held-out agreement uses disjoint ORIGINAL IMAGE IDs, not random clean/adv
    rows of the same image. The embedding itself is transductive. This is not
    an estimate of robust accuracy or the model's high-dimensional boundary.
    """
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from threadpoolctl import threadpool_limits
    ndim = coordinates.shape[1]
    grid_size = grid_size or (140 if ndim == 2 else 26)
    unique_ids = np.unique(anchor_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    n_test = max(1, len(unique_ids) // 5) if len(unique_ids) >= 3 else 0
    test_mask = np.isin(anchor_ids, shuffled[:n_test])
    agreement = None
    with threadpool_limits(limits=1):
        if test_mask.any():
            probe = KNeighborsClassifier(n_neighbors=min(15, int((~test_mask).sum())), weights="distance", n_jobs=1)
            probe.fit(coordinates[~test_mask], predictions[~test_mask])
            agreement = float(np.mean(probe.predict(coordinates[test_mask]) == predictions[test_mask]))
        model = KNeighborsClassifier(n_neighbors=min(15, len(coordinates)), weights="distance", n_jobs=1)
        model.fit(coordinates, predictions)
        axes = [np.linspace(*bound, grid_size) for bound in bounds]
        meshes = np.meshgrid(*axes, indexing="ij")
        grid = np.column_stack([mesh.ravel() for mesh in meshes])
        codes = model.predict(grid).reshape([grid_size] * ndim)
        # Avoid filling distant, unsupported gaps with invented prediction regions.
        nearby = NearestNeighbors(n_neighbors=min(6, len(coordinates)), n_jobs=1).fit(coordinates)
        radius = float(np.quantile(nearby.kneighbors(coordinates)[0][:, -1], .95) * 1.5)
        radius = max(radius, max(hi - lo for lo, hi in bounds) * 1e-6)
        supported = (nearby.kneighbors(grid, n_neighbors=1)[0][:, 0] <= radius).reshape(codes.shape)
    faces = []
    if ndim == 3:
        steps = [axis[1] - axis[0] for axis in axes]
        for dimension in range(3):
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[dimension], right[dimension] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            changed = (codes[left] != codes[right]) & supported[left] & supported[right]
            indices = np.argwhere(changed)
            if not len(indices):
                continue
            centers = np.column_stack([axes[d][indices[:, d]] for d in range(3)])
            centers[:, dimension] += steps[dimension] * .5
            other = [d for d in range(3) if d != dimension]
            corners = np.zeros((4, 3))
            corners[:, other[0]] = np.array([-.5, .5, .5, -.5]) * steps[other[0]]
            corners[:, other[1]] = np.array([-.5, -.5, .5, .5]) * steps[other[1]]
            faces.extend(centers[:, None, :] + corners[None, :, :])
    return {"axes": axes, "meshes": meshes, "codes": codes, "supported": supported,
            "faces": faces, "report": {
                "type": "distance-weighted 15-nearest-neighbor projected prediction surrogate",
                "target": "original model predicted labels, not ground-truth labels",
                "held_out_image_id_agreement": agreement, "held_out_image_ids": int(n_test),
                "agreement_scope": "transductive embedding; diagnostic only, not test accuracy",
                "grid_size_per_axis": grid_size, "supported_grid_fraction": float(supported.mean()),
                "support_radius": radius, "boundary_faces_3d": len(faces),
                "warning": "NOT the original model's high-dimensional decision boundary"}}


def plot_cached_features(run_dir, samples=1000, perplexity=30., iterations=1000,
                         metric="l2", bins=60, seed=None, dimensions="both", boundary_overlays=True):
    validate_options(samples, perplexity, iterations, metric, bins)
    manifest, data = load_feature_cache(run_dir)
    seed = int(manifest["spec"].get("evaluation_seed", 42) if seed is None else seed)
    output = Path(run_dir) / "final_feature_plots"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_hex, hsv_to_rgb
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    files = []

    def save(fig, name, extra_artists=()):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=220, bbox_inches="tight", bbox_extra_artists=extra_artists)
            files.append(path.name)
        plt.close(fig)

    methods = manifest["methods"]
    attacks = [m for m in methods if m != "clean"]
    distances, stats, errors = {}, {}, {}
    distance_label = {"l2": "L2 feature distance", "normalized_l2": "L2 distance of unit-normalized features",
                      "cosine": "Cosine feature distance"}[metric]
    with (output / f"paired_distances_{metric}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("attack", "test_id", "true_class", "clean_prediction", "attack_prediction", "distance"))
        for method in attacks:
            ids, ci, ai = pair_features(data["clean"], data[method])
            values = feature_distances(data["clean"]["features"][ci], data[method]["features"][ai], metric)
            distances[method] = values
            errors[method] = data[method]["predictions"][ai] != data[method]["labels"][ai]
            stats[method] = {"count": len(values), "mean": float(values.mean()),
                             "std": float(values.std()), "median": float(np.median(values)),
                             "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99)),
                             "zero_fraction": float(np.mean(values == 0)),
                             "correct_count": int((~errors[method]).sum()),
                             "wrong_count": int(errors[method].sum())}
            writer.writerows((method, int(v), int(data["clean"]["labels"][c]),
                              int(data["clean"]["predictions"][c]), int(data[method]["predictions"][a]), float(d))
                             for v, c, a, d in zip(ids, ci, ai, values))
    upper = max((float(v.max()) for v in distances.values()), default=1.)
    edges = np.linspace(0., upper * 1.025 if upper > 0 else 1., bins + 1)
    centers = (edges[:-1] + edges[1:]) * .5
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method, values in distances.items():
        color = ATTACK_COLORS.get(method)
        density, _ = np.histogram(values, bins=edges, density=True)
        ax.plot(centers, density, color=color, linewidth=1.8,
                label=f"{DISPLAY.get(method, method)} (n={len(values):,})")
    ax.axvline(0., color=ATTACK_COLORS["clean"], linestyle="--", label="Clean vs Clean = 0")
    ax.set(xlabel=distance_label, ylabel="Probability density (shared bins)",
           title="Final test: paired clean-adversarial feature distances", xlim=(-edges[-1] * .025, edges[-1]))
    ax.grid(alpha=.2); ax.legend(fontsize=9); fig.tight_layout()
    save(fig, f"distance_{metric}_all")
    for method, values in distances.items():
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.hist(values, bins=edges, color=ATTACK_COLORS.get(method, "#0072B2"), alpha=.65,
                edgecolor="#243746", linewidth=.55)
        ax.axvline(stats[method]["p90"], color="#E69F00", linestyle="--",
                   label=f"90th: {stats[method]['p90']:.4g}")
        ax.axvline(stats[method]["p99"], color="#D55E00", linestyle="--",
                   label=f"99th: {stats[method]['p99']:.4g}")
        ax.set(xlabel=distance_label, ylabel="Frequency (number of images)", xlim=(0., edges[-1]),
               title=f"Clean vs {DISPLAY.get(method, method)} | final test | n={len(values):,}")
        ax.grid(axis="y", alpha=.2); ax.legend(); fig.tight_layout()
        save(fig, f"distance_{metric}_{method}")
    if boundary_overlays:
        # A scalar distance has no universal classification threshold. Show
        # empirical correct/wrong components instead of drawing a fake boundary.
        fig, ax = plt.subplots(figsize=(10, 5.4))
        for method, values in distances.items():
            for mask, style in ((~errors[method], "-"), (errors[method], "--")):
                counts, _ = np.histogram(values[mask], bins=edges)
                density = counts / (len(values) * np.diff(edges))
                ax.plot(centers, density, color=ATTACK_COLORS.get(method), linestyle=style, linewidth=1.6)
        attack_handles = [Line2D([], [], color=ATTACK_COLORS.get(m), label=DISPLAY.get(m, m)) for m in attacks]
        status_handles = [Line2D([], [], color="#505050", linestyle=s, label=t)
                          for s, t in (("-", "Predicted correctly"), ("--", "Predicted incorrectly"))]
        ax.axvline(0., color="#737373", linestyle=":", linewidth=1.)
        ax.legend(handles=attack_handles + status_handles, ncol=2, fontsize=9)
        ax.set(xlabel=distance_label, ylabel="Density contribution (area = group proportion)",
               title="Final test: correct / incorrect prediction components\nEmpirical distributions, NOT a classification boundary",
               xlim=(-edges[-1] * .025, edges[-1]))
        ax.grid(alpha=.2); fig.tight_layout()
        save(fig, f"distance_{metric}_all_correct_wrong")
        for method, values in distances.items():
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.hist([values[~errors[method]], values[errors[method]]], bins=edges, stacked=True,
                    color=[ATTACK_COLORS.get(method, "#0072B2"), "#E25C52"], alpha=.72,
                    edgecolor="#243746", linewidth=.5,
                    label=[f"Correct (n={stats[method]['correct_count']:,})", f"Wrong (n={stats[method]['wrong_count']:,})"])
            for quantile, color in (("p90", "#E69F00"), ("p99", "#7B3294")):
                ax.axvline(stats[method][quantile], color=color, linestyle="--", linewidth=1.3,
                           label=f"{quantile}: {stats[method][quantile]:.4g}")
            ax.set(xlabel=distance_label, ylabel="Frequency (number of images)", xlim=(0., edges[-1]),
                   title=f"Clean vs {DISPLAY.get(method, method)}: prediction status\nEmpirical distributions, NOT a classification boundary")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=.2); fig.tight_layout()
            save(fig, f"distance_{metric}_{method}_correct_wrong")
    _json_write(output / f"distance_summary_{metric}.json", {
        "metric": metric, "feature_layer": manifest["spec"].get("feature_layer"),
        "conditions": stats, "shared_bin_edges": edges.tolist(),
        "population": "all evaluated pairs per attack, including unsuccessful attacks",
        "aa_note": "Standard AutoAttack may return the original clean image when no replacement is found; zero distance is retained."})

    methods, ids, labels, stacked, settings = joint_embedding(data, samples, seed, perplexity, iterations)
    prediction_parts = []
    for method in methods:
        lookup = {int(v): i for i, v in enumerate(data[method]["ids"])}
        prediction_parts.append(data[method]["predictions"][[lookup[int(v)] for v in ids]])
    projected_predictions = np.concatenate(prediction_parts)
    classes = np.unique(np.concatenate((labels, projected_predictions))) if boundary_overlays else np.unique(labels)
    if len(classes) <= 20:
        cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
        colors = {int(label): to_hex(cmap(i)) for i, label in enumerate(classes)}
    else:
        colors = {int(label): to_hex(hsv_to_rgb([(i * .61803398875) % 1., .55 + .15 * (i % 3), .72 + .13 * (i % 2)]))
                  for i, label in enumerate(classes)}
    point_colors = [colors[int(label)] for label in labels]
    class_handles = [Line2D([], [], marker="o", color="none", markerfacecolor=colors[int(label)],
                            markeredgecolor="none", label=str(int(label)), markersize=6) for label in classes]
    fig, ax = plt.subplots(figsize=(max(5., min(12., len(classes) * .35)), max(1.8, math.ceil(len(classes) / 8) * .3 + .8)))
    ax.axis("off")
    ax.legend(handles=class_handles, title="Class colors (points: true class; surrogate regions: predicted class)",
              loc="center", ncol=min(8, len(classes)), frameon=False)
    save(fig, "class_legend")
    def scatter_panel(shown, name, title, offsets, bounds, ndim, regions=None):
        fig = plt.figure(figsize=(12.5, 7.8))
        ax = fig.add_subplot(111, projection="3d" if ndim == 3 else None)
        fig.subplots_adjust(right=.78)
        if regions is not None:
            if ndim == 2:
                from matplotlib.colors import ListedColormap
                region_classes = np.unique(regions["codes"])
                mapped = np.searchsorted(region_classes, regions["codes"])
                masked = np.ma.masked_where(~regions["supported"], mapped)
                ax.pcolormesh(*regions["meshes"], masked,
                              cmap=ListedColormap([colors[int(v)] for v in region_classes]),
                              vmin=-.5, vmax=len(region_classes) - .5, alpha=.12, shading="nearest", rasterized=True)
                for label in region_classes:
                    inside = regions["codes"] == label
                    if inside[regions["supported"]].any() and not inside[regions["supported"]].all():
                        ax.contour(*regions["meshes"], np.ma.masked_where(~regions["supported"], inside.astype(float)),
                                   levels=[.5], colors="#5A6872", linewidths=.6, alpha=.55)
            elif regions["faces"]:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                surface = Poly3DCollection(regions["faces"], facecolors="#748A9B", edgecolors="none", alpha=.085)
                surface.set_rasterized(True)
                ax.add_collection3d(surface)
        for method in shown:
            xy = offsets[method]
            options = {"depthshade": False} if ndim == 3 else {}
            ax.scatter(*[xy[:, d] for d in range(ndim)], c=point_colors, marker=MARKERS.get(method, "*"),
                       s=18 if method == "clean" else 24, alpha=.68 if method == "clean" else .48,
                       linewidths=.65, rasterized=True, **options)
        condition_handles = [Line2D([], [], marker=MARKERS.get(m, "*"), linestyle="none", color="#505050",
                                    label=DISPLAY.get(m, m), markersize=7) for m in shown]
        type_legend = ax.legend(handles=condition_handles, title="Condition", loc="upper left", bbox_to_anchor=(1.02, 1.))
        ax.add_artist(type_legend)
        legend_artists = [type_legend]
        if len(classes) <= 20:
            legend_artists.append(ax.legend(handles=class_handles, title="Class", loc="lower left", bbox_to_anchor=(1.02, 0.)))
        else:
            text = ax.text2D if ndim == 3 else ax.text
            text(1.03, .45, f"{len(classes)} class colors\nSee class_legend.png", transform=ax.transAxes)
        subtitle = ("Projected prediction surrogate - NOT the original decision boundary" if regions is not None
                    else f"Shared {ndim}D embedding | {len(ids):,} matched images per condition")
        ax.set(xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2", xlim=bounds[0], ylim=bounds[1],
               title=f"{title}\n{subtitle}")
        if ndim == 3:
            ax.set(zlabel="t-SNE dimension 3", zlim=bounds[2])
            ax.view_init(elev=22, azim=-60)
            ax.set_box_aspect((1, 1, .8))
        if regions is not None:
            note = ("Shading: supported prediction regions; unshaded gaps are not extrapolated." if ndim == 2 else
                    ("Translucent faces: approximate class transitions inside the supported grid." if regions["faces"] else
                     "No class transition detected inside the supported grid; gaps are not extrapolated."))
            legend_artists.append(fig.text(.12, .01, note, fontsize=8.5, color="#505050"))
        ax.grid(alpha=.12)
        save(fig, name, extra_artists=tuple(legend_artists))

    if dimensions not in ("both", "2", "3"):
        raise ValueError("dimensions must be both, 2, or 3")
    embedding_metadata = {}
    for ndim in ((2, 3) if dimensions == "both" else (int(dimensions),)):
        local_settings = {**settings, "dimensions": ndim}
        embedding_key = hashlib.sha256(stacked.tobytes() + json.dumps(local_settings, sort_keys=True).encode()).hexdigest()
        coordinates_path = output / f"tsne_{ndim}d_coordinates.npz"
        coordinates = None
        if coordinates_path.exists():
            with np.load(coordinates_path, allow_pickle=False) as cache:
                if str(cache["key"].item()) == embedding_key:
                    coordinates = cache["coordinates"]
        if coordinates is None:
            print(f"Joint {ndim}D t-SNE: {len(ids)} matched images x {len(methods)} conditions; CPU only", flush=True)
            coordinates = _fit_tsne(stacked, local_settings)
        if not np.isfinite(coordinates).all():
            raise ValueError("t-SNE returned non-finite coordinates")
        _npz_write(coordinates_path, coordinates=coordinates, ids=ids, labels=labels,
                   methods=np.asarray(methods), key=np.asarray(embedding_key))
        lo, hi = coordinates.min(0), coordinates.max(0)
        pad = np.maximum((hi - lo) * .06, 1e-3)
        bounds = [(float(lo[d] - pad[d]), float(hi[d] + pad[d])) for d in range(ndim)]
        offsets = {method: coordinates[i * len(ids):(i + 1) * len(ids)] for i, method in enumerate(methods)}
        regions = (projected_regions(coordinates, projected_predictions, np.tile(ids, len(methods)), bounds, seed)
                   if boundary_overlays else None)
        with (output / f"tsne_{ndim}d_coordinates.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("condition", "test_id", "true_class", *[f"tsne_{d + 1}" for d in range(ndim)]))
            for method in methods:
                writer.writerows((method, int(v), int(label), *map(float, xy))
                                 for v, label, xy in zip(ids, labels, offsets[method]))
        scatter_panel(methods, f"tsne_{ndim}d_all", "Final test: clean and all evaluated attacks", offsets, bounds, ndim)
        scatter_panel(["clean"], f"tsne_{ndim}d_clean", "Final test: clean reference", offsets, bounds, ndim)
        for method in attacks:
            scatter_panel(["clean", method], f"tsne_{ndim}d_{method}",
                          f"Final test: clean vs {DISPLAY.get(method, method)}", offsets, bounds, ndim)
        if regions is not None:
            scatter_panel(methods, f"tsne_{ndim}d_all_projected_boundary",
                          "Final test: clean and all evaluated attacks", offsets, bounds, ndim, regions)
            scatter_panel(["clean"], f"tsne_{ndim}d_clean_projected_boundary",
                          "Final test: clean reference", offsets, bounds, ndim, regions)
            for method in attacks:
                scatter_panel(["clean", method], f"tsne_{ndim}d_{method}_projected_boundary",
                              f"Final test: clean vs {DISPLAY.get(method, method)}", offsets, bounds, ndim, regions)
        embedding_metadata[f"{ndim}d"] = {**local_settings, "embedding_sha256": embedding_key, "axis_limits": bounds}
        if regions is not None:
            embedding_metadata[f"{ndim}d"]["projected_surrogate"] = regions["report"]
            _npz_write(output / f"projected_regions_{ndim}d.npz", predicted_class=regions["codes"],
                       supported=regions["supported"], **{f"axis_{d + 1}": a for d, a in enumerate(regions["axes"])})
    _json_write(output / "plot_manifest.json", {
        "feature_signature": manifest["signature"], "distance_metric": metric,
        "boundary_overlays": boundary_overlays,
        "tsne": embedding_metadata,
        "methods": methods, "markers": {m: MARKERS.get(m, "*") for m in methods},
        "class_colors": colors, "class_counts_per_condition": {int(v): int(np.sum(labels == v)) for v in classes},
        "figures": files,
        "notes": ["Within each dimension, all t-SNE panels reuse one joint embedding and its axes.",
                  "2D and 3D are independent fits, not directly comparable coordinate systems.",
                  "Distances are computed in original high-dimensional features, never in t-SNE space.",
                  "Class colors use ground-truth labels, not predictions.",
                  "No filtering by attack success; unchanged AutoAttack outputs are retained.",
                  "Projected prediction surrogate boundaries are NOT original model decision boundaries.",
                  "Correct/wrong distance components do not define a universal scalar decision threshold.",
                  "t-SNE is descriptive, not a quantitative proof of robustness or causality."]})
    print(f"Feature figures saved to {output}", flush=True)
    return output
