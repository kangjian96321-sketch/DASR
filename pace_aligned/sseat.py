"""PACE-aligned implementation of SSEAT (ADR + CRS).

This is the project's ADR + CRS adaptation under the shared WRN/ResNet-18
protocol, not a bit-for-bit reproduction of an upstream experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# torch.cuda.set_per_process_memory_fraction(0.8, device=0)

from .selection import best_eligible, selection_mean as mean_selection, warmup_epochs, SELECTION_POLICY
from .common import (
    EMA,
    ExperimentConfig,
    StageResourceTracker,
    append_metrics,
    build_dataloaders,
    checkpoint_payload,
    clone_frozen,
    config_from_args,
    evaluate_for_selection,
    evaluate_seen,
    final_evaluation,
    generate_attack,
    js_divergence,
    make_backbone,
    make_optimizer,
    make_scheduler,
    print_config,
    prepare_run_directory,
    resolve_device,
    save_json,
    set_seed,
    tensor_storage_bytes,
    training_attack_seed,
    upsert_resource_row,
    write_environment_manifest,
)


def sseat_training_view(
    images: torch.Tensor,
    image_size: int,
    dataset: str,
) -> torch.Tensor:
    """Apply crop/flip, dataset-appropriate AutoAugment and Cutout replay views."""

    from torchvision import transforms

    policy = (
        transforms.AutoAugmentPolicy.CIFAR10
        if dataset in ("cifar10", "cifar100")
        else transforms.AutoAugmentPolicy.IMAGENET
    )
    transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy),
        ]
    )
    source_device = images.device
    images_u8 = images.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
    augmented = torch.stack([transform(image) for image in images_u8])
    augmented = augmented.to(source_device, non_blocking=True).float().div(255.0)
    cutout_size = image_size // 2
    if cutout_size > 0:
        for index in range(augmented.size(0)):
            top = int(torch.randint(0, image_size + 1 - cutout_size, (1,), device=source_device))
            left = int(torch.randint(0, image_size + 1 - cutout_size, (1,), device=source_device))
            augmented[index, :, top : top + cutout_size, left : left + cutout_size] = 0
    return augmented


class ReplayTensorDataset(Dataset):
    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor):
        self.images_u8 = images_u8
        self.labels = labels
        self.task_ids = task_ids

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int):
        return self.images_u8[index].float().div(255.0), self.labels[index], self.task_ids[index]


class ADRMemory:
    """SSEAT interval-sampled replay memory ordered by augmentation uncertainty."""

    def __init__(
        self,
        capacity: int,
        uncertainty_views: int,
        num_classes: int,
        base_task_fraction: float,
        image_size: int,
    ):
        self.capacity = capacity
        self.uncertainty_views = uncertainty_views
        self.num_classes = num_classes
        self.base_task_fraction = base_task_fraction
        self.images_u8 = torch.empty((0, 3, image_size, image_size), dtype=torch.uint8)
        self.labels = torch.empty((0,), dtype=torch.long)
        self.task_ids = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.numel()

    def dataset(self) -> ReplayTensorDataset:
        return ReplayTensorDataset(self.images_u8, self.labels, self.task_ids)

    def loader(self, cfg: ExperimentConfig) -> Optional[DataLoader]:
        if len(self) == 0:
            return None
        return DataLoader(
            self.dataset(),
            # The released SSEAT code uses a replay batch size of eight.
            batch_size=cfg.sseat_replay_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=cfg.device.startswith("cuda"),
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "images_u8": self.images_u8,
            "labels": self.labels,
            "task_ids": self.task_ids,
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.images_u8 = state["images_u8"].cpu()
        self.labels = state["labels"].cpu()
        self.task_ids = state["task_ids"].cpu()

    def statistics(self) -> Dict[str, object]:
        return {
            "size": len(self),
            "class_counts": dict(sorted(Counter(self.labels.tolist()).items())),
            "task_counts": dict(sorted(Counter(self.task_ids.tolist()).items())),
        }

    @staticmethod
    def _evenly_spaced(indices: torch.Tensor, count: int) -> torch.Tensor:
        """Choose deterministic interval samples without duplicated positions."""

        if count <= 0 or indices.numel() == 0:
            return indices[:0]
        if count >= indices.numel():
            return indices
        positions = torch.arange(count, dtype=torch.long) * indices.numel() // count
        return indices[positions]

    @staticmethod
    def _randaugment(images: torch.Tensor) -> torch.Tensor:
        """Apply independent torchvision RandAugment transforms to uint8 images."""

        from torchvision.transforms import RandAugment

        transform = RandAugment()
        source_device = images.device
        images_u8 = images.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
        augmented = torch.stack([transform(image) for image in images_u8])
        return augmented.to(source_device, non_blocking=True).float().div(255.0)

    @torch.no_grad()
    def _uncertainty(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        predictions = []
        for _ in range(self.uncertainty_views):
            # SSEAT's ADR score is a 12-view RandAugment variance ratio.
            logits = model(self._randaugment(images))
            predictions.append(logits.argmax(dim=1))
        votes = F.one_hot(torch.stack(predictions), num_classes=self.num_classes).sum(dim=0)
        maximum_vote = votes.max(dim=1).values.float()
        return 1.0 - maximum_vote / float(self.uncertainty_views)

    def _score_stored(self, model: nn.Module, cfg: ExperimentConfig, device: torch.device):
        if len(self) == 0:
            empty_float = torch.empty((0,), dtype=torch.float32)
            return self.images_u8, self.labels, self.task_ids, empty_float
        loader = DataLoader(self.dataset(), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        score_parts = []
        for images, _, _ in loader:
            score_parts.append(self._uncertainty(model, images.to(device)).cpu())
        return self.images_u8, self.labels, self.task_ids, torch.cat(score_parts)

    def update(
        self,
        model: nn.Module,
        train_eval_loader: DataLoader,
        task: str,
        task_index: int,
        cfg: ExperimentConfig,
        device: torch.device,
        resource_tracker: Optional[StageResourceTracker] = None,
    ) -> None:
        """Select Dr_(t+1) from Dr_t union the current stage, per SSEAT Alg. 1."""

        model.eval()
        old_images, old_labels, old_tasks, old_scores = self._score_stored(model, cfg, device)
        image_parts: List[torch.Tensor] = [old_images]
        label_parts: List[torch.Tensor] = [old_labels]
        task_parts: List[torch.Tensor] = [old_tasks]
        score_parts: List[torch.Tensor] = [old_scores]

        for batch_index, batch in enumerate(train_eval_loader):
            if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
                break
            images, labels = batch[0].to(device), batch[1].to(device)
            with torch.enable_grad():
                attacked = generate_attack(
                    model,
                    images,
                    labels,
                    task,
                    cfg,
                    training=True,
                    rng_seed=training_attack_seed(
                        cfg, task_index, cfg.task_epochs[task], batch_index, stream=2
                    ),
                )
            if resource_tracker is not None and task != "clean":
                resource_tracker.add_adversarial_samples("memory_update", labels.numel())
            with torch.no_grad():
                scores = self._uncertainty(model, attacked)
            image_parts.append(attacked.mul(255).round().clamp(0, 255).to(torch.uint8).cpu())
            label_parts.append(labels.cpu())
            task_parts.append(torch.full_like(labels.cpu(), task_index))
            score_parts.append(scores.cpu())

        all_images = torch.cat(image_parts)
        all_labels = torch.cat(label_parts)
        all_tasks = torch.cat(task_parts)
        all_scores = torch.cat(score_parts)
        keep = min(self.capacity, all_labels.numel())
        selected_parts: List[torch.Tensor] = []
        base_quota = keep // self.num_classes
        quota_remainder = keep % self.num_classes

        # The release performs interval sampling inside each class and reserves
        # ten percent of each class quota for task 0 before filling the rest.
        for class_id in range(self.num_classes):
            quota = base_quota + int(class_id < quota_remainder)
            class_indices = torch.nonzero(all_labels == class_id, as_tuple=False).flatten()
            if quota == 0 or class_indices.numel() == 0:
                continue
            ordered = class_indices[torch.argsort(all_scores[class_indices], stable=True)]
            clean_ordered = ordered[all_tasks[ordered] == 0]
            reserved_count = min(
                int(quota * self.base_task_fraction),
                clean_ordered.numel(),
            )
            reserved = self._evenly_spaced(clean_ordered, reserved_count)
            remaining_mask = ~torch.isin(ordered, reserved)
            remaining = self._evenly_spaced(ordered[remaining_mask], quota - reserved.numel())
            selected_parts.extend([reserved, remaining])

        selected = torch.cat(selected_parts) if selected_parts else torch.empty(0, dtype=torch.long)
        if selected.numel() < keep:
            # This is only needed for unusually imbalanced/custom datasets.
            unselected = torch.ones(all_labels.numel(), dtype=torch.bool)
            unselected[selected] = False
            remaining = torch.nonzero(unselected, as_tuple=False).flatten()
            remaining = remaining[torch.argsort(all_scores[remaining], stable=True)]
            selected = torch.cat(
                [selected, self._evenly_spaced(remaining, keep - selected.numel())]
            )
        self.images_u8 = all_images[selected].contiguous()
        self.labels = all_labels[selected].contiguous()
        self.task_ids = all_tasks[selected].contiguous()


def train_epoch(
    model: nn.Module,
    teacher: Optional[nn.Module],
    ema: EMA,
    current_loader: DataLoader,
    memory_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    task: str,
    task_index: int,
    epoch: int,
    cfg: ExperimentConfig,
    device: torch.device,
    resource_tracker: Optional[StageResourceTracker] = None,
) -> Dict[str, float]:
    model.train()
    memory_iterator = iter(memory_loader) if memory_loader is not None else None
    current_steps = len(current_loader)
    if cfg.max_train_batches is not None:
        current_steps = min(current_steps, cfg.max_train_batches)
    replay_steps = min(len(memory_loader), current_steps) if memory_loader is not None else 0
    if replay_steps:
        # Spread one pass over memory across the current-data epoch. This avoids
        # the previous implementation's roughly nine complete memory passes.
        offsets = torch.arange(replay_steps, dtype=torch.long)
        scheduled = ((2 * offsets + 1) * current_steps // (2 * replay_steps)).tolist()
        replay_schedule = set(scheduled)
    else:
        replay_schedule = set()
    totals = {
        "loss": 0.0,
        "current_ce": 0.0,
        "replay_ce": 0.0,
        "crs": 0.0,
        "count": 0.0,
        "replay_count": 0.0,
        "replay_batches": 0.0,
    }

    for batch_index, batch in enumerate(current_loader):
        if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
            break
        images, labels = batch[0].to(device), batch[1].to(device)
        with torch.enable_grad():
            attacked = generate_attack(
                model,
                images,
                labels,
                task,
                cfg,
                training=True,
                rng_seed=training_attack_seed(cfg, task_index, epoch, batch_index),
            )
        if resource_tracker is not None and task != "clean":
            resource_tracker.add_adversarial_samples("current_training", labels.numel())
        current_ce = F.cross_entropy(model(attacked), labels)
        loss = current_ce
        replay_ce = torch.zeros((), device=device)
        crs = torch.zeros((), device=device)

        if memory_iterator is not None and teacher is not None and batch_index in replay_schedule:
            replay_images, replay_labels, _ = next(memory_iterator)
            replay_images = replay_images.to(device, non_blocking=True)
            replay_labels = replay_labels.to(device, non_blocking=True)
            previous_view = sseat_training_view(replay_images, cfg.image_size, cfg.dataset)
            current_view = sseat_training_view(replay_images, cfg.image_size, cfg.dataset)
            with torch.no_grad():
                teacher_logits = teacher(previous_view)
            student_logits = model(current_view)
            replay_ce = F.cross_entropy(student_logits, replay_labels)
            crs = js_divergence(teacher_logits, student_logits, cfg.sseat_temperature)
            replay_objective = 0.5 * replay_ce + cfg.sseat_crs_weight * crs
            loss = loss + cfg.sseat_replay_weight * replay_objective

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "Non-finite SSEAT loss at "
                f"task={task}, epoch={epoch + 1}, batch={batch_index}: "
                f"current_ce={float(current_ce.detach())}, "
                f"replay_ce={float(replay_ce.detach())}, crs={float(crs.detach())}. "
                "Check that the adversarial-stage LR is 0.01 and do not resume "
                "from the earlier NaN checkpoint."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update(model)

        batch_size = labels.numel()
        totals["loss"] += float(loss.detach()) * batch_size
        totals["current_ce"] += float(current_ce.detach()) * batch_size
        totals["count"] += batch_size
        if batch_index in replay_schedule and memory_iterator is not None and teacher is not None:
            replay_size = replay_labels.numel()
            totals["replay_ce"] += float(replay_ce.detach()) * replay_size
            totals["crs"] += float(crs.detach()) * replay_size
            totals["replay_count"] += replay_size
            totals["replay_batches"] += 1

    count = max(1.0, totals.pop("count"))
    replay_count = totals.pop("replay_count")
    replay_batches_used = totals.pop("replay_batches")
    memory_size = len(memory_loader.dataset) if memory_loader is not None else 0
    return {
        "loss": totals["loss"] / count,
        "current_ce": totals["current_ce"] / count,
        "replay_ce": totals["replay_ce"] / max(1.0, replay_count),
        "crs": totals["crs"] / max(1.0, replay_count),
        "replay_batches": int(replay_batches_used),
        "replay_examples": int(replay_count),
        "replay_passes": replay_count / max(1, memory_size),
    }


def run(cfg: ExperimentConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    prepare_run_directory(cfg)
    write_environment_manifest(cfg)
    loaders = build_dataloaders(cfg)
    model = make_backbone(cfg).to(device)
    memory = ADRMemory(
        cfg.sseat_memory_size,
        cfg.sseat_uncertainty_views,
        cfg.num_classes,
        cfg.sseat_base_task_fraction,
        cfg.image_size,
    )
    memory_path = cfg.run_dir / "adr_memory.pt"
    completed_path = cfg.run_dir / "completed_stage.pt"
    start_stage = 0
    if cfg.resume and not completed_path.exists():
        raise FileNotFoundError(
            f"Cannot resume {cfg.run_dir}: completed_stage.pt is missing. "
            "Use a new --run-name to restart cleanly."
        )
    if cfg.resume and completed_path.exists():
        completed = torch.load(completed_path, map_location="cpu", weights_only=False)
        if completed.get('selection_policy') != SELECTION_POLICY:
            raise ValueError('Checkpoint selection policy mismatch')
        model.load_state_dict(completed["model"])
        start_stage = int(completed["stage"]) + 1
        del completed
        if not memory_path.exists():
            raise FileNotFoundError('SSEAT completed-stage resume requires adr_memory.pt')
        if memory_path.exists():
            memory.load_state_dict(torch.load(memory_path, map_location="cpu", weights_only=False))

    summaries_path = cfg.run_dir / "stage_summaries.json"
    stage_summaries = (
        json.loads(summaries_path.read_text(encoding="utf-8"))
        if cfg.resume and summaries_path.exists()
        else []
    )
    for task_index in range(start_stage, len(cfg.task_sequence)):
        task = cfg.task_sequence[task_index]
        resource_tracker = StageResourceTracker(
            "sseat", cfg, model, task_index, task, device
        )
        teacher = clone_frozen(model) if task_index > 0 else None
        ema = EMA(model, cfg.ema_decay)
        optimizer = make_optimizer(model, task, cfg)
        scheduler = make_scheduler(optimizer, task, cfg)
        replay_loader = memory.loader(cfg)
        best_score = float("-inf")
        best_path = cfg.run_dir / f"stage_{task_index:02d}_{task}_best.pt"

        for epoch in range(cfg.task_epochs[task]):
            with resource_tracker.timed():
                train_metrics = train_epoch(
                    model,
                    teacher,
                    ema,
                    loaders.train,
                    replay_loader,
                    optimizer,
                    task,
                    task_index,
                    epoch,
                    cfg,
                    device,
                    resource_tracker,
                )
            if (epoch + 1) % cfg.validate_every == 0 or epoch + 1 == cfg.task_epochs[task]:
                seen = cfg.task_sequence[: task_index + 1]
                # Match PACE exactly: average trainable parameters only, while
                # keeping the current model's BN running statistics/buffers.
                with ema.average_parameters():
                    selection = evaluate_for_selection(
                        model,
                        loaders,
                        task,
                        seen,
                        cfg,
                        device,
                    )
                    historical = [name for name in seen if name != task]
                    validation = {task: selection[task]}
                    if historical:
                        validation.update(evaluate_seen(model, loaders.val, historical, cfg, device))
                    mean_accuracy = sum(validation.values()) / len(validation)
                    selection_mean = mean_selection(selection, seen)
                    row = {
                        "stage": task_index,
                        "task": task,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        **train_metrics,
                        "memory_size": len(memory),
                        **{f"val_{name}": value for name, value in validation.items()},
                        "val_mean": mean_accuracy,
                        **{f"selection_{name}": value for name, value in selection.items()},
                        "selection_mean": selection_mean,
                        "best_eligible": best_eligible(cfg, task, epoch),
                        "warmup_epochs": warmup_epochs(cfg, task),
                    }
                    append_metrics(cfg.run_dir / "metrics.csv", row)
                    print(row)
                    payload = checkpoint_payload(
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        task_index,
                        epoch,
                        best_score=(max(best_score, selection_mean)
                                    if best_eligible(cfg, task, epoch) else best_score),
                        extra={"method": "sseat", "memory_size": len(memory)},
                    )
                    torch.save(payload, cfg.run_dir / "progress.pt")
                    if best_eligible(cfg, task, epoch) and selection_mean > best_score:
                        best_score = selection_mean
                        torch.save(payload, best_path)
            # PACE advances the scheduler after validation/checkpointing.
            scheduler.step()

        if not best_path.exists():
            raise RuntimeError(f"No checkpoint was produced for stage {task_index}: {task}")
        # Only model weights need to reach CUDA, not the checkpoint's saved
        # optimizer, EMA and other training tensors.
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        # ``model`` is the complete PACE-style hybrid checkpoint: EMA
        # trainable parameters plus the current BN buffers.
        model.load_state_dict(best["model"])
        del best
        model.eval()
        with resource_tracker.timed():
            memory.update(
                model,
                loaders.train_eval,
                task,
                task_index,
                cfg,
                device,
                resource_tracker,
            )
        torch.save(memory.state_dict(), memory_path)
        save_json(cfg.run_dir / "adr_memory_stats.json", memory.statistics())
        torch.save({"stage": task_index, "model": model.state_dict(), "selection_policy": SELECTION_POLICY}, completed_path)
        summary = {
            "stage": task_index,
            "task": task,
            "best_selection_mean": best_score,
            "memory_size": len(memory),
            "memory_task_counts": memory.statistics()["task_counts"],
        }
        stage_summaries.append(summary)
        save_json(summaries_path, stage_summaries)
        replay_bytes = tensor_storage_bytes(
            memory.images_u8, memory.labels, memory.task_ids
        )
        upsert_resource_row(
            cfg.run_dir,
            resource_tracker.finish(replay_storage_bytes=replay_bytes),
        )
        # The payload holds optimizer.state_dict() references. Release them
        # together with the teacher/EMA before constructing the next stage or
        # entering final evaluation. No saved checkpoint is removed.
        payload = None
        del teacher, ema, optimizer, scheduler, replay_loader, resource_tracker

    results = final_evaluation(model, loaders, cfg, device)
    print("Final test:", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    from .common import add_shared_arguments

    parser = argparse.ArgumentParser(description="PACE-aligned SSEAT")
    add_shared_arguments(parser)
    parser.add_argument(
        "--sseat-memory-size",
        type=int,
        default=None,
        help=(
            "Defaults: CIFAR-10/Imagenette 1000, CIFAR-100 2000, "
            "Tiny-ImageNet 4000 (20 exemplars per class)."
        ),
    )
    parser.add_argument("--sseat-replay-batch-size", type=int, default=8)
    parser.add_argument("--sseat-uncertainty-views", type=int, default=12)
    parser.add_argument("--sseat-base-task-fraction", type=float, default=0.1)
    parser.add_argument("--sseat-temperature", type=float, default=1.0)
    parser.add_argument("--sseat-crs-weight", type=float, default=1.0)
    parser.add_argument("--sseat-replay-weight", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args("sseat", args)
    if args.dry_run:
        print_config(cfg)
        return
    run(cfg)


if __name__ == "__main__":
    main()
