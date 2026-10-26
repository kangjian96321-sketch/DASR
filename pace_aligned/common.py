"""Shared data, model, attack and experiment utilities.

The goal of this module is controlled comparison, not line-by-line reproduction
of either research repository.  It fixes the data/model/attack/optimization
protocol to the settings extracted from ``LACD_all_reply.py`` while leaving
method-specific continual-learning mechanisms in ``sseat.py`` and ``crt.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .selection import SELECTION_POLICY, warmup_epochs, validate_sequence
from my_functions.adversary_tools import NativeAdversary
from my_functions.backbone_options import (
    add_backbone_argument, backbone_spec, backbone_suffix, canonical_backbone,
)
from my_functions.shared_base import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENETTE_MEAN,
    IMAGENETTE_STD,
    EMA as OriginalPACEEMA,
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    UnifiedWideResNet,
)


TASKS: Tuple[str, ...] = ("clean", "patch", "pgd_l2", "pgd")
# Evaluation-only version: invalidate old random-subset results/features, not
# training checkpoints. AA-L2 and AA-Linf must share the same selection policy.
AUTOATTACK_SAMPLING_POLICY = "class_balanced_without_replacement_v1_fullset_preserved"
FINAL_TEST_METHODS: Tuple[str, ...] = (
    "clean",
    "patch",
    "cw_l2",
    "pgd_l2",
    "pgd",
    "autoattack_l2",
    "autoattack",
)

METRIC_COLUMNS: Tuple[str, ...] = (
    "stage",
    "task",
    "epoch",
    "lr",
    "loss",
    "current_ce",
    "replay_ce",
    "crs",
    "replay_batches",
    "replay_examples",
    "replay_passes",
    "memory_size",
    "classification",
    "prototype",
    "feature_kd",
    "reservation",
    "val_clean",
    "val_patch",
    "val_pgd_l2",
    "val_pgd",
    "val_mean",
    "selection_clean",
    "selection_patch",
    "selection_pgd_l2",
    "selection_pgd",
    "selection_mean",
    "best_eligible",
    "warmup_epochs",
)


@dataclass
class ExperimentConfig:
    """Settings shared by PACE, SSEAT and baseline in the aligned comparison."""

    method: str
    selection_policy: str = SELECTION_POLICY
    dataset: str = "cifar100"
    data_root: str = "./data"
    output_root: str = "./runs"
    run_name: str = "three_seed_resources"
    seed: int = 42
    split_seed: int = 42
    evaluation_seed: int = 42
    autoattack_subset_seed: int = 42
    val_size: int = 5000
    batch_size: int = 64
    workers: int = 4
    device: str = "cuda"

    backbone: str = "wrn"
    depth: int = 34
    widen_factor: int = 10
    dropout: float = 0.3
    task_sequence: Tuple[str, ...] = TASKS
    final_test_methods: Tuple[str, ...] = FINAL_TEST_METHODS
    task_epochs: Dict[str, int] = field(
        default_factory=lambda: {"clean": 100, "patch": 30, "pgd_l2": 30, "pgd": 30}
    )
    task_lrs: Dict[str, float] = field(
        default_factory=lambda: {"clean": 0.1, "patch": 0.01, "pgd_l2": 0.01, "pgd": 0.01}
    )
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_ratio: float = 0.05
    warmup_start_factor: float = 0.1
    cosine_eta_min_ratio: float = 0.001
    ema_decay: float = 0.999

    linf_eps: float = 8.0 / 255.0
    linf_alpha: float = 2.0 / 255.0
    l2_eps: float = 0.5
    l2_alpha: float = 0.1
    train_attack_steps: int = 10
    eval_attack_steps: int = 20
    cw_eval_steps: int = 30
    patch_size: int = 8
    patch_alpha: float = 0.1
    autoattack_samples: int = 1000
    autoattack_batch_size: int = 32

    # Final-test diagnostics only; excluded from PACE training identity.
    plot_final_features: bool = False
    tsne_samples: int = 1000
    tsne_perplexity: float = 30.0
    tsne_iterations: int = 1000
    tsne_dimensions: str = "both"
    feature_boundary_overlays: bool = True
    feature_distance: str = "l2"
    feature_bins: int = 60

    validate_every: int = 1
    selection_probe_size: int = 500
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    resume: bool = False
    # Optional explicit run folder, useful when copying legacy checkpoints.
    run_directory: Optional[str] = None

    # SSEAT-only. Defaults restore the method settings used by the release.
    sseat_memory_size: int = 2000
    sseat_replay_batch_size: int = 8
    sseat_uncertainty_views: int = 12
    sseat_base_task_fraction: float = 0.1
    sseat_temperature: float = 1.0
    sseat_crs_weight: float = 1.0
    sseat_replay_weight: float = 1.0

    def __post_init__(self):
        self.backbone = canonical_backbone(self.backbone)

    @property
    def num_classes(self) -> int:
        if self.dataset == "cifar10":
            return 10
        if self.dataset == "cifar100":
            return 100
        if self.dataset == "tiny_imagenet":
            return 200
        if self.dataset == "imagenette":
            return 10
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    @property
    def image_size(self) -> int:
        return {
            "cifar10": 32,
            "cifar100": 32,
            "tiny_imagenet": 64,
            "imagenette": 160,
        }[self.dataset]

    @property
    def run_dir(self) -> Path:
        if self.run_directory:
            return Path(self.run_directory)
        suffix = f"_{self.run_name}" if self.run_name else ""
        return Path(self.output_root) / f"{self.method}{backbone_suffix(self.backbone)}_{self.dataset}_seed{self.seed}{suffix}"


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    add_backbone_argument(parser)
    from .feature_plots import add_feature_arguments
    add_feature_arguments(parser)
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "tiny_imagenet", "imagenette"),
        default="cifar100",
    )
    parser.add_argument("--task-sequence", nargs=4, choices=TASKS, default=TASKS,
                        help="clean patch pgd_l2 pgd OR clean pgd pgd_l2 patch")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./runs")
    parser.add_argument(
        "--run-name",
        default="three_seed_resources",
        help="Suffix used to keep this corrected run separate from earlier logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--autoattack-subset-seed", type=int, default=42)
    parser.add_argument(
        "--autoattack-batch-size", type=int, default=None,
        help="Final AutoAttack batch only; defaults to min(32, training batch size).",
    )
    parser.add_argument(
        "--autoattack-samples", type=int, default=1000,
        help="Class-balanced AutoAttack subset size; 0 (or >= test size) uses the full test set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults: CIFAR 64, Tiny-ImageNet 16, Imagenette 32.",
    )
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after the most recently completed stage in the same run directory.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved protocol and exit without downloading data.",
    )


def config_from_args(method: str, args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        method=method,
        task_sequence=validate_sequence(getattr(args, 'task_sequence', TASKS)),
        backbone=getattr(args, "backbone", "wrn"),
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        split_seed=args.split_seed,
        evaluation_seed=args.evaluation_seed,
        autoattack_subset_seed=args.autoattack_subset_seed,
        batch_size=args.batch_size or {
            "cifar10": 64,
            "cifar100": 64,
            "tiny_imagenet": 16,
            "imagenette": 32,
        }[args.dataset],
        workers=args.workers,
        device=args.device,
        resume=args.resume,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if args.val_size is not None:
        cfg.val_size = args.val_size
    elif cfg.dataset == "imagenette":
        cfg.val_size = 1000
    if cfg.dataset in ("cifar10", "imagenette"):
        cfg.sseat_memory_size = 1000
    elif cfg.dataset == "tiny_imagenet":
        # Twenty exemplars per class, matching CIFAR-100's released density.
        cfg.sseat_memory_size = 20 * cfg.num_classes
    scale = cfg.image_size / 32.0
    cfg.patch_size = cfg.image_size // 4
    cfg.l2_eps = 0.5 * scale
    cfg.l2_alpha = 0.1 * scale
    cfg.autoattack_batch_size = min(32, cfg.batch_size)
    if getattr(args, "autoattack_batch_size", None) is not None:
        if args.autoattack_batch_size <= 0:
            raise ValueError("--autoattack-batch-size must be positive")
        cfg.autoattack_batch_size = args.autoattack_batch_size
    cfg.autoattack_samples = getattr(args, "autoattack_samples", 1000)
    if cfg.autoattack_samples < 0:
        raise ValueError("--autoattack-samples must be nonnegative (0 = full test set)")
    from .feature_plots import apply_feature_arguments
    apply_feature_arguments(cfg, args)
    for name in (
        "sseat_memory_size",
        "sseat_replay_batch_size",
        "sseat_uncertainty_views",
        "sseat_base_task_fraction",
        "sseat_temperature",
        "sseat_crs_weight",
        "sseat_replay_weight",
    ):
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if cfg.run_name and Path(cfg.run_name).name != cfg.run_name:
        raise ValueError("run_name must be a single directory-name component")
    if cfg.sseat_memory_size <= 0:
        raise ValueError("sseat_memory_size must be positive")
    if cfg.sseat_replay_batch_size <= 0:
        raise ValueError("sseat_replay_batch_size must be positive")
    if cfg.sseat_uncertainty_views <= 1:
        raise ValueError("sseat_uncertainty_views must be greater than one")
    if not 0.0 <= cfg.sseat_base_task_fraction <= 1.0:
        raise ValueError("sseat_base_task_fraction must be in [0, 1]")
    if cfg.sseat_temperature <= 0.0:
        raise ValueError("sseat_temperature must be positive")
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, object]:
    result = asdict(cfg)
    result["task_sequence"] = list(cfg.task_sequence)
    result["final_test_methods"] = list(cfg.final_test_methods)
    result["run_dir"] = str(cfg.run_dir)
    result["num_classes"] = cfg.num_classes
    result["image_size"] = cfg.image_size
    result["resolved_backbone"] = backbone_spec(cfg)
    return result


def print_config(cfg: ExperimentConfig) -> None:
    print(json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and device.index is not None:
        visible_count = torch.cuda.device_count()
        if device.index >= visible_count:
            raise ValueError(
                f"Requested {requested}, but only {visible_count} CUDA device(s) are visible. "
                "Use logical cuda:0 or set CUDA_VISIBLE_DEVICES before Python starts."
            )
    return device


@contextmanager
def isolated_random_seed(seed: int, device: Optional[torch.device] = None):
    """Use a deterministic RNG stream without changing the training stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: List[int] = []
    if device is not None and device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        random.seed(seed)
        np.random.seed(seed)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def training_attack_seed(
    cfg: ExperimentConfig,
    task_index: int,
    epoch: int,
    batch_index: int,
    stream: int = 0,
) -> int:
    """A stable, method-independent seed for a generated training batch."""

    return int(cfg.seed * 10_000_000 + task_index * 1_000_000 + epoch * 10_000 + batch_index * 10 + stream)


def tensor_storage_bytes(*objects: object) -> int:
    """Count unique tensor storage bytes recursively, without double counting views."""

    seen = set()
    total = 0

    def visit(value: object) -> None:
        nonlocal total
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), str(value.device))
            if key not in seen:
                seen.add(key)
                total += int(storage.nbytes())
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for obj in objects:
        visit(obj)
    return total


class StageResourceTracker:
    """Collect comparable per-stage cost statistics for all three methods."""

    ATTACK_CATEGORIES = (
        "current_training",
        "replay_training",
        "memory_update",
        "prototype_initialization",
        "prototype_update",
    )

    def __init__(
        self,
        method: str,
        cfg: ExperimentConfig,
        model: nn.Module,
        stage: int,
        task: str,
        device: torch.device,
    ) -> None:
        self.method = method
        self.cfg = cfg
        self.model = model
        self.stage = stage
        self.task = task
        self.device = device
        self.training_wall_time_seconds = 0.0
        self.gpu_training_time_seconds = 0.0
        self.generated = {name: 0 for name in self.ATTACK_CATEGORIES}
        self.prior_peak_allocated_bytes = 0
        self.prior_peak_reserved_bytes = 0
        self._section_start: Optional[float] = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def begin_timed_section(self) -> None:
        if self._section_start is not None:
            raise RuntimeError("A resource-timing section is already active")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._section_start = time.perf_counter()

    def end_timed_section(self) -> None:
        if self._section_start is None:
            raise RuntimeError("No resource-timing section is active")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.prior_peak_allocated_bytes = max(
                self.prior_peak_allocated_bytes,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.prior_peak_reserved_bytes = max(
                self.prior_peak_reserved_bytes,
                int(torch.cuda.max_memory_reserved(self.device)),
            )
        elapsed = time.perf_counter() - self._section_start
        self.training_wall_time_seconds += elapsed
        if self.device.type == "cuda":
            self.gpu_training_time_seconds += elapsed
        self._section_start = None

    @contextmanager
    def timed(self):
        self.begin_timed_section()
        try:
            yield
        finally:
            self.end_timed_section()

    def add_adversarial_samples(self, category: str, count: int) -> None:
        if category not in self.generated:
            raise ValueError(f"Unknown adversarial-sample category: {category}")
        self.generated[category] += int(count)

    def state_dict(self) -> Dict[str, object]:
        return {
            "training_wall_time_seconds": self.training_wall_time_seconds,
            "gpu_training_time_seconds": self.gpu_training_time_seconds,
            "generated": dict(self.generated),
            "peak_gpu_memory_allocated_bytes": self.prior_peak_allocated_bytes,
            "peak_gpu_memory_reserved_bytes": self.prior_peak_reserved_bytes,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.training_wall_time_seconds = float(state.get("training_wall_time_seconds", 0.0))
        self.gpu_training_time_seconds = float(state.get("gpu_training_time_seconds", 0.0))
        stored = state.get("generated", {})
        if isinstance(stored, Mapping):
            for name in self.generated:
                self.generated[name] = int(stored.get(name, 0))
        self.prior_peak_allocated_bytes = int(
            state.get("peak_gpu_memory_allocated_bytes", 0)
        )
        self.prior_peak_reserved_bytes = int(
            state.get("peak_gpu_memory_reserved_bytes", 0)
        )

    def finish(self, replay_storage_bytes: int) -> Dict[str, object]:
        if self._section_start is not None:
            self.end_timed_section()
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        generated_total = sum(self.generated.values())
        if self.device.type == "cuda":
            peak_allocated = self.prior_peak_allocated_bytes
            peak_reserved = self.prior_peak_reserved_bytes
            gpu_time: Optional[float] = self.gpu_training_time_seconds
        else:
            peak_allocated = 0
            peak_reserved = 0
            gpu_time = None
        return {
            "method": self.method,
            "dataset": self.cfg.dataset,
            "train_seed": self.cfg.seed,
            "split_seed": self.cfg.split_seed,
            "evaluation_seed": self.cfg.evaluation_seed,
            "autoattack_subset_seed": self.cfg.autoattack_subset_seed,
            "stage": self.stage,
            "task": self.task,
            "image_size": self.cfg.image_size,
            "batch_size": self.cfg.batch_size,
            "training_wall_time_seconds": self.training_wall_time_seconds,
            "gpu_training_time_seconds": gpu_time,
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "generated_adversarial_samples": generated_total,
            **{
                f"generated_{name}_adversarial_samples": count
                for name, count in self.generated.items()
            },
            "replay_storage_bytes": int(replay_storage_bytes),
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
        }


def upsert_resource_row(run_dir: Path, row: Mapping[str, object]) -> None:
    """Save JSON and CSV forms, replacing the same stage on resume."""

    json_path = run_dir / "resource_metrics.json"
    rows: List[Dict[str, object]] = []
    if json_path.exists():
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = [dict(item) for item in loaded]
    rows = [
        item
        for item in rows
        if not (
            int(item.get("stage", -1)) == int(row["stage"])
            and int(item.get("train_seed", -1)) == int(row["train_seed"])
        )
    ]
    rows.append(dict(row))
    rows.sort(key=lambda item: (int(item["train_seed"]), int(item["stage"])))
    save_json(json_path, rows)
    csv_path = run_dir / "resource_metrics.csv"
    fieldnames = list(dict.fromkeys(key for item in rows for key in item))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IndexedDataset(Dataset):
    """Return a subset-local index together with image and label."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


class TinyImageNetValidation(Dataset):
    """Read the original Tiny-ImageNet validation layout without reorganizing it."""

    def __init__(self, root: Path, class_to_idx: Mapping[str, int], transform) -> None:
        self.transform = transform
        annotation_path = root / "val" / "val_annotations.txt"
        image_root = root / "val" / "images"
        if not annotation_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet validation data must contain val/val_annotations.txt "
                "and val/images/."
            )
        samples = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1] in class_to_idx:
                samples.append((image_root / fields[0], class_to_idx[fields[1]]))
        self.samples = sorted(samples, key=lambda item: item[0].name)
        if not self.samples:
            raise RuntimeError("No labelled Tiny-ImageNet validation images were found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


@dataclass
class DataBundle:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    selection_val: DataLoader
    test: DataLoader
    train_indices: Sequence[int]
    val_indices: Sequence[int]


def dataset_target(dataset: Dataset, index: int) -> int:
    """Read a target without decoding the corresponding image when possible."""
    if isinstance(dataset, Subset):
        source_index = int(dataset.indices[int(index)])
        return dataset_target(dataset.dataset, source_index)

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return int(targets[int(index)])

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return int(samples[int(index)][1])

    return int(dataset[int(index)][1])


def class_balanced_indices(
    dataset: Dataset,
    size: int,
    num_classes: int,
    seed: int,
) -> List[int]:
    """Select a deterministic, approximately class-balanced subset.

    Counts differ by at most one whenever every requested class has enough
    examples.  The selected indices are interleaved by class, which also makes
    every complete ``num_classes``-sized prefix balanced.  Small synthetic
    smoke-test datasets may contain fewer observed classes than ``num_classes``;
    those runs balance over the observed classes only.
    """
    size = int(size)
    num_classes = int(num_classes)
    if size <= 0 or size > len(dataset):
        raise ValueError(
            f"Balanced subset size must be in [1, {len(dataset)}], got {size}"
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    buckets: List[List[int]] = [[] for _ in range(num_classes)]
    for index in range(len(dataset)):
        label = dataset_target(dataset, index)
        if not 0 <= label < num_classes:
            raise ValueError(f"Invalid class label {label}")
        buckets[label].append(index)

    observed_classes = [
        class_id for class_id, bucket in enumerate(buckets) if bucket
    ]
    if not observed_classes:
        raise ValueError("Cannot balance an empty-label dataset")
    if size >= num_classes and len(observed_classes) != num_classes:
        missing = [
            class_id for class_id, bucket in enumerate(buckets) if not bucket
        ]
        raise ValueError(
            "Balanced subset cannot cover every configured class; "
            f"missing classes: {missing}"
        )

    active_classes = (
        list(range(num_classes)) if size >= num_classes else observed_classes
    )
    rng = random.Random(int(seed))
    rng.shuffle(active_classes)
    for bucket in buckets:
        rng.shuffle(bucket)

    base, remainder = divmod(size, len(active_classes))
    quotas = {
        class_id: base + int(rank < remainder)
        for rank, class_id in enumerate(active_classes)
    }
    for class_id, quota in quotas.items():
        if len(buckets[class_id]) < quota:
            raise ValueError(
                f"Class {class_id} has {len(buckets[class_id])} samples, "
                f"but {quota} are required"
            )

    selected: List[int] = []
    for rank in range(max(quotas.values())):
        for class_id in active_classes:
            if rank < quotas[class_id]:
                selected.append(buckets[class_id][rank])

    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError("Invalid class-balanced subset")
    return selected


def autoattack_subset_indices(dataset: Dataset, cfg: ExperimentConfig) -> List[int]:
    """Fixed, label-balanced AA subset; full-set mode retains every example.

    Use ground-truth labels, never model predictions or attack success. The
    seed depends on neither the method/model nor the AA norm. Strict balancing
    fails explicitly if a class lacks its quota; it never duplicates examples.
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("AutoAttack evaluation dataset is empty")
    if cfg.autoattack_samples < 0:
        raise ValueError("autoattack_samples must be nonnegative (0 = full test set)")
    count = total if cfg.autoattack_samples == 0 else min(cfg.autoattack_samples, total)
    if count == total:
        # Preserve full-test-set behavior and its original seeded ordering.
        generator = torch.Generator().manual_seed(cfg.autoattack_subset_seed)
        return torch.randperm(total, generator=generator).tolist()
    try:
        return class_balanced_indices(dataset, count, cfg.num_classes, cfg.autoattack_subset_seed)
    except ValueError as exc:
        raise ValueError(
            f"Cannot construct a class-balanced AutoAttack subset: {exc}. "
            "Use a smaller --autoattack-samples value, verify class coverage, "
            "or use --autoattack-samples 0 for the unmodified full test set."
        ) from exc


def build_dataloaders(cfg: ExperimentConfig) -> DataBundle:
    """Build one shared split and preprocessing pipeline for all three methods."""

    from torchvision import datasets, transforms

    if cfg.dataset == "imagenette":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(round(cfg.image_size / 0.875)),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(cfg.image_size, padding=cfg.image_size // 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
            ]
        )

    if cfg.dataset in ("cifar10", "cifar100"):
        dataset_cls = datasets.CIFAR10 if cfg.dataset == "cifar10" else datasets.CIFAR100
        train_aug_full = dataset_cls(
            cfg.data_root, train=True, download=True, transform=train_transform
        )
        train_eval_full = dataset_cls(
            cfg.data_root, train=True, download=False, transform=eval_transform
        )
        test_set = dataset_cls(
            cfg.data_root, train=False, download=True, transform=eval_transform
        )
    else:
        data_root = Path(cfg.data_root)
        expected_names = (
            ("tiny-imagenet-200", "tiny_imagenet")
            if cfg.dataset == "tiny_imagenet"
            else ("imagenette2-160", "imagenette2", "imagenette")
        )
        candidates = [data_root] + [data_root / name for name in expected_names]
        dataset_root = next(
            (candidate for candidate in candidates if (candidate / "train").is_dir()),
            None,
        )
        if dataset_root is None:
            raise FileNotFoundError(
                f"Cannot find {cfg.dataset} below {data_root}. Expected a train/ "
                f"directory or one of: {', '.join(expected_names)}."
            )
        train_aug_full = datasets.ImageFolder(dataset_root / "train", transform=train_transform)
        train_eval_full = datasets.ImageFolder(dataset_root / "train", transform=eval_transform)
        if cfg.dataset == "tiny_imagenet" and (
            dataset_root / "val" / "val_annotations.txt"
        ).is_file():
            test_set = TinyImageNetValidation(
                dataset_root,
                train_aug_full.class_to_idx,
                eval_transform,
            )
        else:
            test_set = datasets.ImageFolder(dataset_root / "val", transform=eval_transform)

    if len(getattr(train_aug_full, "classes", range(cfg.num_classes))) != cfg.num_classes:
        raise RuntimeError(
            f"{cfg.dataset} must contain {cfg.num_classes} classes, found "
            f"{len(getattr(train_aug_full, 'classes', []))}."
        )
    if cfg.val_size <= 0 or cfg.val_size >= len(train_aug_full):
        raise ValueError(
            f"val_size={cfg.val_size} must be between 1 and {len(train_aug_full) - 1}"
        )

    # Split membership is fixed by split_seed; train_seed only changes model
    # initialization, shuffling, augmentation and attack randomness.
    permutation = list(range(len(train_aug_full)))
    split_random = random.Random(cfg.split_seed)
    split_random.shuffle(permutation)
    split_point = len(permutation) - cfg.val_size
    train_indices = permutation[:split_point]
    val_indices = permutation[split_point:]

    train_set = IndexedDataset(Subset(train_aug_full, train_indices))
    train_eval_set = Subset(train_eval_full, train_indices)

    # Put a deterministic class-balanced monitor panel at the start of the
    # validation order.  Full validation can then expose both its complete
    # accuracy and the monitor-prefix accuracy from one attack-generation pass.
    raw_val_set = Subset(train_eval_full, val_indices)
    selection_count = min(cfg.selection_probe_size, len(raw_val_set))
    balanced_positions = class_balanced_indices(
        raw_val_set,
        selection_count,
        cfg.num_classes,
        cfg.split_seed + 6151,
    )
    balanced_position_set = set(balanced_positions)
    ordered_positions = balanced_positions + [
        index
        for index in range(len(raw_val_set))
        if index not in balanced_position_set
    ]
    val_indices = [val_indices[index] for index in ordered_positions]
    val_set = Subset(train_eval_full, val_indices)

    use_pin = cfg.device.startswith("cuda")
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=use_pin,
        persistent_workers=cfg.workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(train_eval_set, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    selection_val_loader = DataLoader(
        Subset(val_set, range(selection_count)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return DataBundle(
        train=train_loader,
        train_eval=train_eval_loader,
        val=val_loader,
        selection_val=selection_val_loader,
        test=test_loader,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def make_backbone(cfg: ExperimentConfig) -> nn.Module:
    """Shared factory; keep the WRN class/initialization exactly as before."""

    if cfg.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif cfg.dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif cfg.dataset == "tiny_imagenet":
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
    elif cfg.dataset == "imagenette":
        mean, std = IMAGENETTE_MEAN, IMAGENETTE_STD
    else:
        raise ValueError(cfg.dataset)
    if canonical_backbone(cfg.backbone) == "resnet18":
        from my_functions.resnet_base import UnifiedResNet18
        return UnifiedResNet18(
            num_classes=cfg.num_classes, dataset_mean=mean, dataset_std=std,
            small_input=cfg.dataset in ("cifar10", "cifar100"),
        )
    return UnifiedWideResNet(
        depth=cfg.depth,
        widen_factor=cfg.widen_factor,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout,
        dataset_mean=mean,
        dataset_std=std,
        use_mlp_projector=False,
    )


class EMA:
    """Compatibility wrapper around PACE's parameter-only EMA implementation."""

    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.impl = OriginalPACEEMA(model, decay=decay)

    def update(self, model: Optional[nn.Module] = None) -> None:
        if model is not None and model is not self.model:
            raise ValueError("PACE EMA must update the model it was created for")
        self.impl.update()

    def apply_shadow(self) -> None:
        self.impl.apply_shadow()

    def restore(self) -> None:
        self.impl.restore()

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.impl.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.impl.load_state_dict(state_dict)

    @contextmanager
    def average_parameters(self):
        """Temporarily expose PACE EMA parameters with current BN buffers."""

        self.apply_shadow()
        try:
            yield self.model
        finally:
            self.restore()


def clone_frozen(model: nn.Module) -> nn.Module:
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def attack_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "attack_logits"):
        return model.attack_logits(x)
    output = model(x)
    return output[0] if isinstance(output, tuple) else output


class _SemanticAttackView(nn.Module):
    """Expose classifier logits to attacks, hiding optional feature outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attack_logits(self.model, x)


def generate_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    task: str,
    cfg: ExperimentConfig,
    training: bool,
    rng_seed: Optional[int] = None,
) -> torch.Tensor:
    if rng_seed is not None:
        with isolated_random_seed(rng_seed, x.device):
            return generate_attack(model, x, y, task, cfg, training, rng_seed=None)
    if task == "clean":
        return x.detach()
    steps = cfg.train_attack_steps if training else cfg.eval_attack_steps
    attack_model = _SemanticAttackView(model)
    attack_model.train(model.training)
    if task == "fgsm":
        return NativeAdversary.fgsm(attack_model, x, y, eps=cfg.linf_eps)
    if task == "bim":
        return NativeAdversary.bim(
            attack_model, x, y, eps=cfg.linf_eps, alpha=cfg.linf_alpha, iters=steps
        )
    if task == "mifgsm":
        return NativeAdversary.mifgsm(
            attack_model, x, y, eps=cfg.linf_eps, alpha=cfg.linf_alpha, iters=steps
        )
    if task == "pgd":
        return NativeAdversary.pgd(
            attack_model, x, y, eps=cfg.linf_eps, alpha=cfg.linf_alpha, iters=steps
        )
    if task == "pgd_l2":
        return NativeAdversary.pgd_l2(
            attack_model, x, y, eps=cfg.l2_eps, alpha=cfg.l2_alpha, iters=steps
        )
    if task == "patch":
        return NativeAdversary.patch(
            attack_model, x, y, patch_size=cfg.patch_size, iters=steps
        )
    if task == "pgd_cw":
        return NativeAdversary.pgd_cw(
            attack_model, x, y, eps=cfg.linf_eps, alpha=cfg.linf_alpha, iters=steps
        )
    if task == "cw_l2":
        if training:
            # This reproduces get_adv_images('cw_l2') in the supplied module.
            return NativeAdversary.cw_l2(
                attack_model, x, y, c=1.0, kappa=0, iters=cfg.train_attack_steps, lr=0.01
            )
        # This reproduces LACD_all_reply.py's final bounded CW-L2 evaluation.
        return NativeAdversary.cw_l2(
            attack_model,
            x,
            y,
            c=10.0,
            kappa=0,
            iters=cfg.cw_eval_steps,
            lr=0.05,
            bound_eps=cfg.l2_eps,
        )
    raise ValueError(f"Unknown task: {task}")


def tensor_augment(x: torch.Tensor, cutout_size: Optional[int] = None) -> torch.Tensor:
    """Independent CIFAR augmentation for SSEAT's ADR/CRS computations."""

    image_size = x.size(-1)
    padding = image_size // 8
    cutout_size = image_size // 4 if cutout_size is None else cutout_size
    padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    cropped = torch.empty_like(x)
    for index in range(x.size(0)):
        top = int(torch.randint(0, 2 * padding + 1, (1,), device=x.device))
        left = int(torch.randint(0, 2 * padding + 1, (1,), device=x.device))
        cropped[index] = padded[
            index, :, top : top + image_size, left : left + image_size
        ]
    flip = torch.rand(x.size(0), 1, 1, 1, device=x.device) < 0.5
    cropped = torch.where(flip, torch.flip(cropped, dims=(3,)), cropped)

    brightness = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
    contrast = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
    channel_mean = cropped.mean(dim=(2, 3), keepdim=True)
    cropped = ((cropped - channel_mean) * contrast + channel_mean) * brightness
    cropped = cropped.clamp(0, 1)

    if cutout_size > 0:
        for index in range(x.size(0)):
            top = int(
                torch.randint(0, image_size + 1 - cutout_size, (1,), device=x.device)
            )
            left = int(
                torch.randint(0, image_size + 1 - cutout_size, (1,), device=x.device)
            )
            cropped[index, :, top : top + cutout_size, left : left + cutout_size] = 0
    return cropped


import math

def js_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    log_teacher = F.log_softmax(
        teacher_logits.detach() / temperature, dim=1
    )
    log_student = F.log_softmax(
        student_logits / temperature, dim=1
    )

    log_mixture = (
        torch.logaddexp(log_teacher, log_student)
        - math.log(2.0)
    )

    teacher_prob = log_teacher.exp()
    student_prob = log_student.exp()

    teacher_kl = (
        teacher_prob * (log_teacher - log_mixture)
    ).sum(dim=1).mean()

    student_kl = (
        student_prob * (log_student - log_mixture)
    ).sum(dim=1).mean()

    return 0.5 * (teacher_kl + student_kl)


def make_optimizer(model: nn.Module, task: str, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.SGD(
        parameters,
        lr=cfg.task_lrs[task],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    task: str,
    cfg: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    epochs = cfg.task_epochs[task]
    count = warmup_epochs(cfg, task)
    if count == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.task_lrs[task] * cfg.cosine_eta_min_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.warmup_start_factor,
        end_factor=1.0,
        total_iters=count,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - count),
        eta_min=cfg.task_lrs[task] * cfg.cosine_eta_min_ratio,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[count],
    )


def base_prediction(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return semantic predictions; expanded pseudo-class heads are unsupported."""
    if logits.size(1) != num_classes:
        raise ValueError("Classifier output does not match the dataset class count")
    return logits.argmax(dim=1)


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "evaluation_logits"):
        return model.evaluation_logits(x)
    output = model(x)
    return output[0] if isinstance(output, tuple) else output


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    attack: str,
    cfg: ExperimentConfig,
    device: torch.device,
    feature_collector=None,
) -> float:
    model.eval()
    if feature_collector is not None:
        from torch.utils.data import SequentialSampler
        from .feature_plots import forward_with_features
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("Final feature collection requires a sequential test loader for stable sample IDs")
    correct = 0
    total = 0
    attack_seed = cfg.evaluation_seed + {
        "clean": 0,
        "patch": 101,
        "cw_l2": 211,
        "pgd_l2": 307,
        "pgd": 401,
    }.get(attack, 997)
    # Validation must not consume the training RNG stream, and every method
    # must receive the same random-start attacks.
    with isolated_random_seed(attack_seed, device):
        for batch_index, batch in enumerate(loader):
            if cfg.max_eval_batches is not None and batch_index >= cfg.max_eval_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.enable_grad():
                attacked = generate_attack(model, x, y, attack, cfg, training=False)
            with torch.no_grad():
                if feature_collector is None:
                    logits = _forward_logits(model, attacked)
                else:
                    logits, features = forward_with_features(model, attacked)
                prediction = base_prediction(logits, cfg.num_classes)
                if feature_collector is not None:
                    feature_collector.add(range(total, total + y.numel()), y, features, prediction)
            correct += int((prediction == y).sum())
            total += y.numel()
    accuracy = 100.0 * correct / max(1, total)
    if feature_collector is not None:
        feature_collector.finish(accuracy)
    return accuracy


def evaluate_seen(
    model: nn.Module,
    loader: DataLoader,
    seen_tasks: Sequence[str],
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    return {
        task: evaluate_accuracy(model, loader, task, cfg, device)
        for task in seen_tasks
    }


def evaluate_for_selection(
    model: nn.Module,
    loaders: DataBundle,
    current_task: str,
    seen_tasks: Sequence[str],
    cfg: ExperimentConfig,
    device: torch.device,
    current_accuracy: Optional[float] = None,
) -> Dict[str, float]:
    """Retain the SSEAT/CRT current-full, historical-monitor selection split.

    The current task is evaluated on the full validation set. Every previous
    task, including clean once it becomes historical, is evaluated on the
    fixed first ``selection_probe_size`` validation examples.
    """

    current = (
        evaluate_accuracy(model, loaders.val, current_task, cfg, device)
        if current_accuracy is None
        else current_accuracy
    )
    result = {current_task: current}
    historical = [task for task in seen_tasks if task != current_task]
    if historical:
        result.update(evaluate_seen(model, loaders.selection_val, historical, cfg, device))
    return result


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_metrics(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    unexpected = set(row) - set(METRIC_COLUMNS)
    if unexpected:
        raise ValueError(f"Unknown metric columns: {sorted(unexpected)}")
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if tuple(existing_header) != METRIC_COLUMNS:
            raise RuntimeError(
                f"{path} has an incompatible header. Use a new --run-name rather "
                "than appending corrected rows to an older CSV."
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    task_index: int,
    epoch: int,
    best_score: float,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    # PACE saves a CPU clone while EMA parameters are temporarily applied.
    # Cloning is essential: ``state_dict`` otherwise shares storage with the
    # live model and ``ema.restore()`` would change the in-memory payload.
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().clone() for name, value in ema.state_dict().items()}
    return {
        "model": model_state,
        # Parameter-only EMA state, retained for diagnostics/resume parity.
        # It is intentionally not loadable as a complete model state dict.
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "task_index": task_index,
        "epoch": epoch,
        "best_score": best_score,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": dict(extra or {}),
        "selection_policy": SELECTION_POLICY,
    }


def load_model_state(path: Path, model: nn.Module, device: torch.device, key: str = "model") -> Dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload[key])
    return payload


def prepare_run_directory(cfg: ExperimentConfig) -> None:
    """Prevent accidental mixing of old and corrected experiment logs."""

    if cfg.resume:
        if not cfg.run_dir.exists():
            raise FileNotFoundError(f"Cannot resume: {cfg.run_dir} does not exist")
        config_path = cfg.run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError("Cannot verify resume without config.json")
        if config_path.exists():
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            if stored.get("selection_policy") != SELECTION_POLICY:
                raise ValueError("Checkpoint selection policy mismatch; start a NEW run")
            from .feature_plots import PLOT_FIELDS
            ignored = set(PLOT_FIELDS) | {"resume", "device", "workers", "output_root",
                "run_name", "run_directory", "run_dir", "data_root", "autoattack_batch_size",
                "autoattack_samples", "autoattack_subset_seed", "final_test_methods"}
            current = config_to_dict(cfg)
            changed = [k for k, v in current.items() if k not in ignored and stored.get(k) != v]
            if changed:
                raise ValueError("Training configuration mismatch: " + ", ".join(changed))
            # Missing backbone means legacy WRN. Check before overwriting any
            # metadata, even when the caller supplies an explicit run folder.
            if (stored.get("dataset", cfg.dataset) != cfg.dataset
                    or backbone_spec(stored) != backbone_spec(cfg)):
                raise ValueError(
                    "Checkpoint architecture/dataset mismatch: use the original "
                    "--backbone and --dataset, or start a separate experiment."
                )
        return
    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {cfg.run_dir}. Choose a new --run-name "
            "or pass --resume for a completed-stage restart."
        )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def _frozen_model_parameters(model: nn.Module):
    """Keep input gradients available without differentiating model weights.

    AutoAttack's check_dynamic calls model(x) with grad mode enabled even when
    x does not require gradients. Freezing weights avoids an unnecessary graph
    for that check; APGD/FAB still differentiate their requires_grad inputs.
    Restore each original flag, including baseline's already-frozen parameters.
    """
    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, original_flags):
            parameter.requires_grad_(flag)


def prepare_final_evaluation(model: nn.Module, device: torch.device) -> None:
    """Drop unused gradients and allocator cache, not live training objects.

    Callers must release their optimizer/checkpoint/teacher references first.
    This does not move or change the selected model parameters or buffers.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    gc.collect()
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def final_evaluation(
    model: nn.Module,
    loaders: DataBundle,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    prepare_final_evaluation(model, device)
    if cfg.plot_final_features:
        from .feature_plots import make_feature_store
        store = make_feature_store(model, loaders.test, cfg)
        results = {}
        for method in store.methods:
            if store.has(method):
                results[method] = store.accuracy(method)
            elif method in ("autoattack_l2", "autoattack"):
                norm, eps = ("L2", cfg.l2_eps) if method == "autoattack_l2" else ("Linf", cfg.linf_eps)
                results[method] = evaluate_autoattack(model, loaders.test, norm, eps, cfg, device,
                                                     feature_collector=store.collector(method))
            else:
                results[method] = evaluate_accuracy(model, loaders.test, method, cfg, device,
                                                   feature_collector=store.collector(method))
            save_json(cfg.run_dir / "final_test.json", results)
        store.render(cfg)
        return results
    basic_methods = [
        method
        for method in cfg.final_test_methods
        if method not in ("autoattack", "autoattack_l2")
    ]
    results = evaluate_seen(model, loaders.test, basic_methods, cfg, device)
    if "autoattack_l2" in cfg.final_test_methods:
        results["autoattack_l2"] = evaluate_autoattack(model, loaders.test, "L2", cfg.l2_eps, cfg, device)
    if "autoattack" in cfg.final_test_methods:
        results["autoattack"] = evaluate_autoattack(model, loaders.test, "Linf", cfg.linf_eps, cfg, device)
    save_json(cfg.run_dir / "final_test.json", results)
    return results


def evaluate_autoattack(
    model: nn.Module,
    loader: DataLoader,
    norm: str,
    eps: float,
    cfg: ExperimentConfig,
    device: torch.device,
    feature_collector=None,
) -> float:
    """Run standard AutoAttack with full-split tensors on CPU, GPU batches only.

    Keep one standard-evaluation call over the fixed subset. Splitting that
    call into independent attacks would change attack batching/RNG behavior.
    Both AutoAttack and the final accuracy pass obey autoattack_batch_size.
    """

    if cfg.autoattack_batch_size <= 0:
        raise ValueError("autoattack_batch_size must be positive")
    if cfg.autoattack_samples < 0:
        raise ValueError("autoattack_samples must be nonnegative (0 = full test set)")

    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError(
            "AutoAttack is required only for final evaluation. Install it separately "
            "or copy a complete autoattack package into this environment; see README.md."
        ) from exc

    total_examples = len(loader.dataset)
    indices = autoattack_subset_indices(loader.dataset, cfg)
    count = len(indices)
    mode = "full_test_set" if count == total_examples else "class_balanced_subset"
    class_counts = {str(label): 0 for label in range(cfg.num_classes)}
    for index in indices:
        label = str(dataset_target(loader.dataset, index))
        class_counts[label] = class_counts.get(label, 0) + 1
    # Test-dataset-relative indices, in the exact input order used by both AA
    # norms and by paired clean/adversarial feature collection.
    save_json(cfg.run_dir / "autoattack_subset.json", {
        "sampling_policy": AUTOATTACK_SAMPLING_POLICY, "mode": mode,
        "dataset": cfg.dataset, "test_dataset_size": total_examples,
        "requested_samples": cfg.autoattack_samples, "effective_samples": count,
        "subset_seed": cfg.autoattack_subset_seed, "num_classes": cfg.num_classes,
        "class_counts": class_counts, "indices": indices,
        "index_space": "loader.test.dataset (zero-based)",
    })
    print(f"AutoAttack {norm}: {count}/{total_examples} test examples, "
          f"sampling={mode}, batch_size={cfg.autoattack_batch_size}, device={device}; "
          f"class counts and indices: {cfg.run_dir / 'autoattack_subset.json'}", flush=True)
    subset_loader = DataLoader(
        Subset(loader.dataset, indices),
        batch_size=cfg.autoattack_batch_size,
        shuffle=False,
        num_workers=0,
    )
    images, labels = [], []
    for batch in subset_loader:
        images.append(batch[0].detach().cpu())
        labels.append(batch[1].detach().cpu())
    selected_images = torch.cat(images)
    selected_labels = torch.cat(labels)
    del images, labels, batch

    prepare_final_evaluation(model, device)
    attack_model = _SemanticAttackView(model).to(device).eval()
    adversary = AutoAttack(
        attack_model,
        norm=norm,
        eps=eps,
        version="standard",
        verbose=True,
        log_path=str(cfg.run_dir / f"autoattack_{norm}.log"),
        seed=cfg.evaluation_seed,
        device=str(device),
    )
    # Keep input gradients enabled and all standard checks/attacks intact.
    # Model-weight gradients are unnecessary for robustness evaluation.
    with isolated_random_seed(cfg.evaluation_seed, device), _frozen_model_parameters(attack_model), torch.enable_grad():
        adversarial = adversary.run_standard_evaluation(
            selected_images,
            selected_labels,
            bs=cfg.autoattack_batch_size,
        )
    # Official AutoAttack returns on the input device (CPU here). The explicit
    # transfer also supports compatible versions that return GPU tensors.
    adversarial = adversarial.detach().cpu()
    del adversary, selected_images
    correct = 0
    with torch.no_grad():
        for start in range(0, count, cfg.autoattack_batch_size):
            end = min(start + cfg.autoattack_batch_size, count)
            x_batch = adversarial[start:end].to(device)
            y_batch = selected_labels[start:end].to(device)
            if feature_collector is None:
                prediction = attack_model(x_batch).argmax(dim=1)
            else:
                from .feature_plots import forward_with_features
                logits, features = forward_with_features(model, x_batch)
                prediction = base_prediction(logits, cfg.num_classes)
                feature_collector.add(indices[start:end], y_batch, features, prediction)
            correct += int(prediction.eq(y_batch).sum().item())
            del x_batch, y_batch, prediction
    accuracy = 100.0 * correct / count
    if feature_collector is not None:
        feature_collector.finish(accuracy)
    return accuracy


def write_environment_manifest(cfg: ExperimentConfig) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.run_dir / "config.json", config_to_dict(cfg))
    manifest = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(cfg.run_dir / "environment.json", manifest)
