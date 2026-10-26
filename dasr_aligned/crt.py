"""DASR-aligned reproduction of Regularized Continual Robust Training (RCRT).

The method-specific components follow Dai et al., ICML 2025:

* continual fine-tuning over the attacks known so far;
* FT-Croce online attack replay, which samples one known attack per minibatch
  with probability proportional to its observed error rate; and
* adversarial logit-distance regularization (ALR).

The surrounding training and evaluation protocol is local to this project.
L2/Linf ALR follows the pinned official L2Reg implementation; Patch remains a
project-specific extension. This does not reproduce the complete upstream run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .crt_lp_alr import LP_ALR_IMPLEMENTATION, LP_TASKS, generate_lp_alr_view, official_lp_alr
from .selection import best_eligible, selection_mean as mean_selection, warmup_epochs, SELECTION_POLICY
from .common import (
    EMA,
    ExperimentConfig,
    StageResourceTracker,
    append_metrics,
    attack_logits,
    build_dataloaders,
    checkpoint_payload,
    config_from_args,
    evaluate_for_selection,
    evaluate_seen,
    final_evaluation,
    generate_attack,
    isolated_random_seed,
    make_backbone,
    make_optimizer,
    make_scheduler,
    prepare_run_directory,
    print_config,
    resolve_device,
    save_json,
    set_seed,
    training_attack_seed,
    upsert_resource_row,
    write_environment_manifest,
)


@dataclass(frozen=True)
class CRTOptions:
    """Method-specific settings kept separate from the shared protocol."""

    strategy: str = "croce"
    use_alr: bool = True
    alr_steps: int = 1
    alr_weight_initial: float = 1.0
    alr_weight_finetune: float = 0.5

    def __post_init__(self):
        if self.strategy not in ("croce", "single"):
            raise ValueError("Unknown CRT strategy")
        if not isinstance(self.alr_steps, int) or self.alr_steps <= 0:
            raise ValueError("CRT ALR steps must be a positive integer")
        if any(not math.isfinite(w) or w < 0 for w in
               (self.alr_weight_initial, self.alr_weight_finetune)):
            raise ValueError("CRT ALR weights must be finite and nonnegative")


def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.flatten(1)
    norms = flat.norm(p=2, dim=1).clamp_min(1e-12)
    factors = torch.minimum(torch.ones_like(norms), eps / norms)
    return delta * factors.view(-1, 1, 1, 1)


def _alr_step_size(task: str, cfg: ExperimentConfig, steps: int) -> float:
    """Use the schedule implemented by the authors for ALR maximization."""

    if task == "pgd_l2":
        eps = cfg.l2_eps
    elif task in ("pgd", "fgsm", "bim", "mifgsm"):
        eps = cfg.linf_eps
    elif task == "patch":
        # The aligned Patch attack updates pixels with a 0.1 sign step.
        return 0.1
    else:
        raise ValueError(f"ALR is not implemented for task: {task}")
    if steps == 1:
        return eps
    if steps > 3:
        return eps / float(steps - 2)
    return eps / float(steps)


def generate_alr_view(
    model: nn.Module,
    images: torch.Tensor,
    task: str,
    cfg: ExperimentConfig,
    steps: int,
    rng_seed: int,
) -> torch.Tensor:
    """Maximize clean-to-perturbed logit distance inside ``task``'s constraint.

    The output is detached.  Gradients for model optimization are computed only
    later, when the final ALR value is added to the classification objective.
    """

    if task == "clean":
        return images.detach()
    if steps <= 0:
        raise ValueError("ALR steps must be positive")
    if task in LP_TASKS:
        return generate_lp_alr_view(model, images, task, cfg, steps, rng_seed)

    with isolated_random_seed(rng_seed, images.device):
        was_training = model.training
        model.eval()
        clean = images.detach()
        with torch.no_grad():
            clean_reference = attack_logits(model, clean).detach()

        if task == "patch":
            batch, _, height, width = clean.shape
            patch_size = cfg.patch_size
            if patch_size > min(height, width):
                raise ValueError(
                    f"patch_size={patch_size} exceeds input {height}x{width}"
                )
            mask = torch.zeros_like(clean)
            for index in range(batch):
                top = int(
                    torch.randint(
                        0, height - patch_size + 1, (1,), device=clean.device
                    )
                )
                left = int(
                    torch.randint(
                        0, width - patch_size + 1, (1,), device=clean.device
                    )
                )
                mask[
                    index,
                    :,
                    top : top + patch_size,
                    left : left + patch_size,
                ] = 1.0
            random_patch = torch.rand_like(clean)
            adversarial = torch.where(mask.bool(), random_patch, clean).detach()
            step_size = _alr_step_size(task, cfg, steps)
            for _ in range(steps):
                adversarial.requires_grad_(True)
                distance = torch.linalg.vector_norm(
                    attack_logits(model, adversarial) - clean_reference,
                    ord=2,
                    dim=1,
                ).mean()
                gradient = torch.autograd.grad(distance, adversarial)[0]
                candidate = adversarial.detach() + step_size * gradient.sign()
                adversarial = torch.where(mask.bool(), candidate, clean)
                adversarial = adversarial.clamp(0.0, 1.0).detach()
        else:
            raise ValueError(f"ALR is not implemented for task: {task}")

        model.train(was_training)
        return adversarial.detach()


def adversarial_logit_regularization(
    model: nn.Module,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
) -> torch.Tensor:
    """Mean L2 distance between benign and adversarial logits (ALR)."""

    clean_logits = attack_logits(model, clean)
    adversarial_logits = attack_logits(model, adversarial)
    return torch.linalg.vector_norm(
        adversarial_logits - clean_logits, ord=2, dim=1
    ).mean()


def compute_alr(model, images, task, cfg, steps, rng_seed):
    if task in LP_TASKS:
        return official_lp_alr(model, images, task, cfg, steps, rng_seed)
    view = generate_alr_view(model, images, task, cfg, steps, rng_seed)
    return adversarial_logit_regularization(model, images, view)


def _choose_attack(
    attacks: Sequence[str],
    current_task: str,
    strategy: str,
    correct: Mapping[str, int],
    totals: Mapping[str, int],
) -> str:
    if not attacks:
        return "clean"
    if strategy == "single" or len(attacks) == 1:
        return current_task
    if strategy != "croce":
        raise ValueError(f"Unknown CRT strategy: {strategy}")

    # This follows the released FT-Croce loop: sampling is uniform until each
    # attack has an observation, then proportional to its empirical error.
    if all(totals[name] > 0 for name in attacks):
        weights = [
            max(1e-6, 1.0 - correct[name] / float(totals[name]))
            for name in attacks
        ]
    else:
        weights = [0.5 for _ in attacks]
    return random.choices(list(attacks), weights=weights, k=1)[0]


def train_epoch(
    model: nn.Module,
    ema: EMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    current_task: str,
    known_attacks: Sequence[str],
    stage: int,
    epoch: int,
    cfg: ExperimentConfig,
    options: CRTOptions,
    device: torch.device,
    resource_tracker: Optional[StageResourceTracker] = None,
) -> Dict[str, object]:
    model.train()
    attack_correct = {name: 0 for name in known_attacks}
    attack_totals = {name: 0 for name in known_attacks}
    attack_batches = {name: 0 for name in known_attacks}
    totals = {
        "loss": 0.0,
        "current_ce": 0.0,
        "current_count": 0,
        "replay_ce": 0.0,
        "replay_count": 0,
        "alr": 0.0,
        "count": 0,
        "replay_batches": 0,
    }

    adversarial_stage = current_task != "clean"
    if adversarial_stage:
        alr_weight = (
            options.alr_weight_initial
            if len(known_attacks) == 1
            else options.alr_weight_finetune
        )
    else:
        alr_weight = 0.0

    for batch_index, batch in enumerate(loader):
        if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
            break
        images, labels = batch[0].to(device), batch[1].to(device)
        selected_attack = _choose_attack(
            known_attacks,
            current_task,
            options.strategy,
            attack_correct,
            attack_totals,
        )

        optimizer.zero_grad(set_to_none=True)
        primary_seed = training_attack_seed(
            cfg, stage, epoch, batch_index, stream=0
        )
        with torch.enable_grad():
            attacked = generate_attack(
                model,
                images,
                labels,
                selected_attack,
                cfg,
                training=True,
                rng_seed=primary_seed,
            )
        if adversarial_stage and resource_tracker is not None:
            category = (
                "current_training"
                if selected_attack == current_task
                else "replay_training"
            )
            resource_tracker.add_adversarial_samples(category, labels.numel())

        logits = attack_logits(model, attacked)
        classification = F.cross_entropy(logits, labels)

        alr = torch.zeros((), device=device)
        if adversarial_stage and options.use_alr and alr_weight > 0.0:
            alr = compute_alr(
                model,
                images,
                selected_attack,
                cfg,
                steps=options.alr_steps,
                rng_seed=training_attack_seed(
                    cfg, stage, epoch, batch_index, stream=1
                ),
            )
            if resource_tracker is not None:
                resource_tracker.add_adversarial_samples(
                    "regularization_training", labels.numel()
                )

        loss = classification + alr_weight * alr
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite CRT loss at stage={stage}, epoch={epoch}, "
                f"batch={batch_index}, attack={selected_attack}"
            )
        loss.backward()
        optimizer.step()
        ema.update(model)

        batch_size = labels.numel()
        with torch.no_grad():
            correct_count = int(logits.argmax(dim=1).eq(labels).sum())
        if selected_attack in attack_totals:
            attack_correct[selected_attack] += correct_count
            attack_totals[selected_attack] += batch_size
            attack_batches[selected_attack] += 1

        totals["loss"] += float(loss.detach()) * batch_size
        totals["alr"] += float(alr.detach()) * batch_size
        totals["count"] += batch_size
        if selected_attack == current_task:
            totals["current_ce"] += float(classification.detach()) * batch_size
            totals["current_count"] += batch_size
        else:
            totals["replay_ce"] += float(classification.detach()) * batch_size
            totals["replay_count"] += batch_size
            totals["replay_batches"] += 1

    count = max(1, int(totals["count"]))
    current_count = max(1, int(totals["current_count"]))
    replay_count = max(1, int(totals["replay_count"]))
    return {
        "loss": float(totals["loss"]) / count,
        "current_ce": float(totals["current_ce"]) / current_count,
        "replay_ce": float(totals["replay_ce"]) / replay_count,
        # Reuse the shared auxiliary-loss column for CRT's ALR value.
        "crs": float(totals["alr"]) / count,
        "replay_batches": int(totals["replay_batches"]),
        "replay_examples": int(totals["replay_count"]),
        # CRT has no finite replay dataset, so a pass count is not meaningful.
        "replay_passes": 0.0,
        "attack_batches": attack_batches,
    }


def run(cfg: ExperimentConfig, options: CRTOptions) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    # Validate ALR options before any saved metadata is overwritten.
    from dataclasses import asdict
    option_record = {"lp_alr_implementation": LP_ALR_IMPLEMENTATION, **asdict(options)}
    if cfg.resume:
        option_path = cfg.run_dir / "crt_config.json"
        if not option_path.exists():
            raise FileNotFoundError("Cannot verify CRT resume without crt_config.json")
        stored = json.loads(option_path.read_text(encoding="utf-8"))
        if any(stored.get(k) != v for k, v in option_record.items()):
            raise ValueError("CRT ALR/configuration mismatch; start a NEW run")
    prepare_run_directory(cfg)
    write_environment_manifest(cfg)
    save_json(
        cfg.run_dir / "crt_config.json",
        {
            **option_record,
            "selection_policy": SELECTION_POLICY,
            "paper": "Dai et al., ICML 2025",
            "strategy": options.strategy,
            "use_alr": options.use_alr,
            "alr_steps": options.alr_steps,
            "alr_weight_initial": options.alr_weight_initial,
            "alr_weight_finetune": options.alr_weight_finetune,
            "replay": "online attack regeneration; no stored adversarial images",
            "shared_protocol": "DASR-aligned common.py",
        },
    )

    loaders = build_dataloaders(cfg)
    model = make_backbone(cfg).to(device)
    completed_path = cfg.run_dir / "completed_stage.pt"
    start_stage = 0
    if cfg.resume and not completed_path.exists():
        raise FileNotFoundError(
            f"Cannot resume {cfg.run_dir}: completed_stage.pt is missing. "
            "Use a new --run-name to restart cleanly."
        )
    if cfg.resume:
        completed = torch.load(
            completed_path, map_location="cpu", weights_only=False
        )
        if completed.get("selection_policy") != SELECTION_POLICY:
            raise ValueError("Checkpoint selection policy mismatch")
        model.load_state_dict(completed["model"])
        start_stage = int(completed["stage"]) + 1
        del completed

    summaries_path = cfg.run_dir / "stage_summaries.json"
    stage_summaries = (
        json.loads(summaries_path.read_text(encoding="utf-8"))
        if cfg.resume and summaries_path.exists()
        else []
    )

    for stage in range(start_stage, len(cfg.task_sequence)):
        current_task = cfg.task_sequence[stage]
        known_attacks = tuple(
            task
            for task in cfg.task_sequence[1 : stage + 1]
            if task != "clean"
        )
        resource_tracker = StageResourceTracker(
            "crt", cfg, model, stage, current_task, device
        )
        # Keep the common tracker schema intact while exposing CRT's additional
        # online adversarial view separately in resource_metrics.*.
        resource_tracker.generated["regularization_training"] = 0
        ema = EMA(model, cfg.ema_decay)
        optimizer = make_optimizer(model, current_task, cfg)
        scheduler = make_scheduler(optimizer, current_task, cfg)
        best_score = float("-inf")
        best_path = cfg.run_dir / f"stage_{stage:02d}_{current_task}_best.pt"
        stage_attack_batches = {name: 0 for name in known_attacks}

        for epoch in range(cfg.task_epochs[current_task]):
            with resource_tracker.timed():
                train_metrics = train_epoch(
                    model=model,
                    ema=ema,
                    loader=loaders.train,
                    optimizer=optimizer,
                    current_task=current_task,
                    known_attacks=known_attacks,
                    stage=stage,
                    epoch=epoch,
                    cfg=cfg,
                    options=options,
                    device=device,
                    resource_tracker=resource_tracker,
                )
            epoch_attack_batches = train_metrics.pop("attack_batches")
            for name, value in epoch_attack_batches.items():
                stage_attack_batches[name] = stage_attack_batches.get(name, 0) + int(value)

            if (epoch + 1) % cfg.validate_every == 0 or epoch + 1 == cfg.task_epochs[current_task]:
                seen = cfg.task_sequence[: stage + 1]
                with ema.average_parameters():
                    selection = evaluate_for_selection(
                        model, loaders, current_task, seen, cfg, device
                    )
                    historical = [name for name in seen if name != current_task]
                    validation = {current_task: selection[current_task]}
                    if historical:
                        validation.update(
                            evaluate_seen(model, loaders.val, historical, cfg, device)
                        )
                    mean_accuracy = sum(validation.values()) / len(validation)
                    selection_mean = mean_selection(selection, seen)
                    row = {
                        "stage": stage,
                        "task": current_task,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        **train_metrics,
                        "memory_size": 0,
                        **{
                            f"val_{name}": value
                            for name, value in validation.items()
                        },
                        "val_mean": mean_accuracy,
                        **{
                            f"selection_{name}": value
                            for name, value in selection.items()
                        },
                        "selection_mean": selection_mean,
                        "best_eligible": best_eligible(cfg, current_task, epoch),
                        "warmup_epochs": warmup_epochs(cfg, current_task),
                    }
                    append_metrics(cfg.run_dir / "metrics.csv", row)
                    print(row)
                    payload = checkpoint_payload(
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        stage,
                        epoch,
                        best_score=(max(best_score, selection_mean)
                                    if best_eligible(cfg, current_task, epoch) else best_score),
                        extra={
                            "method": "regularized_crt",
                            "strategy": options.strategy,
                            "known_attacks": list(known_attacks),
                            "attack_batches": dict(stage_attack_batches),
                            "alr_steps": options.alr_steps,
                        },
                    )
                    torch.save(payload, cfg.run_dir / "progress.pt")
                    if best_eligible(cfg, current_task, epoch) and selection_mean > best_score:
                        best_score = selection_mean
                        torch.save(payload, best_path)
            scheduler.step()

        if not best_path.exists():
            raise RuntimeError(
                f"No checkpoint was produced for stage {stage}: {current_task}"
            )
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        del best
        model.eval()
        torch.save(
            {"stage": stage, "model": model.state_dict(), "selection_policy": SELECTION_POLICY}, completed_path
        )
        summary = {
            "stage": stage,
            "task": current_task,
            "best_selection_mean": best_score,
            "lp_alr_implementation": LP_ALR_IMPLEMENTATION,
            "selection_policy": SELECTION_POLICY,
            "known_attacks": list(known_attacks),
            "attack_batches": stage_attack_batches,
            "strategy": options.strategy,
            "alr_weight": (
                0.0
                if current_task == "clean" or not options.use_alr
                else (
                    options.alr_weight_initial
                    if len(known_attacks) == 1
                    else options.alr_weight_finetune
                )
            ),
        }
        stage_summaries.append(summary)
        save_json(summaries_path, stage_summaries)
        upsert_resource_row(
            cfg.run_dir,
            resource_tracker.finish(replay_storage_bytes=0),
        )
        payload = None
        del ema, optimizer, scheduler, resource_tracker

    results = final_evaluation(model, loaders, cfg, device)
    print("Final test:", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    from .common import add_shared_arguments

    parser = argparse.ArgumentParser(
        description="DASR-aligned Regularized Continual Robust Training (ICML 2025)"
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--crt-strategy",
        choices=("croce", "single"),
        default="croce",
        help="croce is the recommended online replay baseline; single is an ablation.",
    )
    parser.add_argument(
        "--crt-use-alr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--crt-alr-steps", type=int, default=1)
    parser.add_argument("--crt-alr-weight-initial", type=float, default=1.0)
    parser.add_argument("--crt-alr-weight-finetune", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args("crt", args)
    try:
        options = CRTOptions(
            strategy=args.crt_strategy,
            use_alr=args.crt_use_alr,
            alr_steps=args.crt_alr_steps,
            alr_weight_initial=args.crt_alr_weight_initial,
            alr_weight_finetune=args.crt_alr_weight_finetune,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print_config(cfg)
        print(json.dumps(options.__dict__, indent=2, ensure_ascii=False))
        return
    run(cfg, options)


if __name__ == "__main__":
    main()
