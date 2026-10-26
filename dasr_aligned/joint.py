"""Joint-Mean: supervised training on all chosen threats in every minibatch.

The default joint objective is mean(CE(clean), CE(Patch), CE(PGD-L2),
CE(PGD-Linf)). All attacks use the current model, before its optimizer update.
There is no teacher, replay selector/cache, LACD, or CRT ALR. Optional clean
pretraining precedes the joint phase; attacks are never introduced sequentially.
Original DASR/CRT/SSEAT files and defaults are intentionally left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import (
    AUTOATTACK_SAMPLING_POLICY, TASKS, EMA, ExperimentConfig,
    StageResourceTracker, add_shared_arguments, attack_logits,
    build_dataloaders, config_from_args, config_to_dict, evaluate_accuracy,
    evaluate_autoattack, evaluate_seen, generate_attack, make_backbone,
    make_optimizer, make_scheduler, prepare_final_evaluation, resolve_device,
    set_seed, training_attack_seed, upsert_resource_row, write_environment_manifest,
)
from .selection import best_eligible, selection_mean, warmup_epochs
from .feature_plots import PLOT_FIELDS, _model_hash, make_feature_store


VERSION = "joint_mean_v1_20261005"


@dataclass(frozen=True)
class JointOptions:
    attacks: tuple[str, ...] = TASKS
    grad_clip: float = 0.0

    def validate(self):
        if (not self.attacks or len(set(self.attacks)) != len(self.attacks)
                or any(name not in TASKS for name in self.attacks)):
            raise ValueError("Choose distinct --joint-attacks from clean patch pgd_l2 pgd")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("--grad-clip must be finite and nonnegative (0 disables clipping)")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def training_spec(cfg, options):
    values = config_to_dict(cfg)
    # Evaluation-only options may change on a completed run without retraining.
    for key in (*PLOT_FIELDS, "resume", "device", "workers", "output_root",
                "run_name", "run_directory", "run_dir", "data_root",
                "autoattack_batch_size", "autoattack_samples", "autoattack_subset_seed",
                "final_test_methods"):
        values.pop(key, None)
    if cfg.backbone == "resnet18":
        for key in ("depth", "widen_factor", "dropout"):
            values.pop(key, None)
    return {"version": VERSION, "config": values, "options": asdict(options),
            "objective": "unweighted_mean_per_condition_cross_entropy",
            "batchnorm": "shared_buffers_separate_condition_forwards",
            "selection": "EMA, full validation, mean over phase conditions; zeros included"}


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def epoch_loader(dataset, cfg, stage, epoch):
    # Same epoch seeding convention as DASR. Fresh workers also reset augmentation
    # RNG streams at an epoch boundary, including after a resumed process.
    seed = cfg.seed * 1_000_000 + stage * 10_000 + epoch
    set_seed(seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.workers, pin_memory=cfg.device.startswith("cuda"),
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def train_epoch(model, ema, loader, optimizer, conditions, stage, epoch,
                cfg, options, device, tracker=None):
    """Average all condition losses, then make exactly one update per clean batch.

    Detached inputs for all conditions are generated before any training-mode
    forward. This fixes parameters AND BN buffers throughout attack generation.
    Forward/backward are then performed one condition at a time to avoid keeping
    four activation graphs. Shared BN updates once per condition; this is NOT
    numerically equivalent to concatenating four conditions into one BN batch.
    """
    model.train()
    examples = updates = 0
    loss_sum = 0.0
    ce_sum = {name: 0.0 for name in conditions}
    correct = {name: 0 for name in conditions}
    attack_examples = {name: 0 for name in conditions if name != "clean"}
    for batch_index, batch in enumerate(loader):
        if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
            break
        images, labels = batch[0].to(device), batch[1].to(device)
        count = labels.numel()
        optimizer.zero_grad(set_to_none=True)
        views = []
        for name in conditions:
            with torch.enable_grad():
                view = generate_attack(
                    model, images, labels, name, cfg, training=True,
                    rng_seed=training_attack_seed(
                        cfg, stage, epoch, batch_index, stream=TASKS.index(name)),
                ).detach()
            views.append(view)
            if name != "clean":
                attack_examples[name] += count
                if tracker is not None:
                    tracker.add_adversarial_samples("current_training", count)
        for name, view in zip(conditions, views):
            logits = attack_logits(model, view)
            ce = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError(
                    f"Non-finite joint CE: stage={stage}, epoch={epoch}, "
                    f"batch={batch_index}, condition={name}")
            (ce / len(conditions)).backward()
            ce_sum[name] += float(ce.detach()) * count
            loss_sum += float(ce.detach()) * count / len(conditions)
            correct[name] += int(logits.detach().argmax(1).eq(labels).sum())
        if options.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        ema.update(model)
        examples += count
        updates += 1
        del views, view, logits, ce
    if examples == 0:
        raise ValueError("No training examples; check the dataset and --max-train-batches")
    return {
        "loss": loss_sum / examples, "classification": loss_sum / examples,
        "condition_ce": {name: value / examples for name, value in ce_sum.items()},
        "train_accuracy": {name: 100.0 * value / examples for name, value in correct.items()},
        "clean_source_examples": examples, "condition_examples": examples * len(conditions),
        "optimizer_updates": updates, "generated_by_attack": attack_examples,
        "generated_adversarial_samples": sum(attack_examples.values()),
    }


def evaluation_spec(model, loaders, cfg, signature):
    dataset = loaders.test.dataset
    ordering = getattr(dataset, "samples", None)
    if ordering is None:
        targets = getattr(dataset, "targets", None)
        ordering = None if targets is None else list(map(int, targets))
    return {
        "training_signature": signature, "model_sha256": _model_hash(model),
        "data_root": str(Path(cfg.data_root).resolve()), "test_size": len(dataset),
        "test_order_sha256": digest(ordering),
        "preprocessing": repr(getattr(dataset, "transform", None)),
        "evaluation_seed": cfg.evaluation_seed, "batch_size": cfg.batch_size,
        "autoattack_sampling_policy": AUTOATTACK_SAMPLING_POLICY,
        "autoattack_samples": cfg.autoattack_samples,
        "autoattack_samples_effective": (
            len(dataset) if cfg.autoattack_samples == 0
            else min(cfg.autoattack_samples, len(dataset))),
        "autoattack_batch_size": cfg.autoattack_batch_size,
        "autoattack_subset_seed": cfg.autoattack_subset_seed,
        "eval_attack_steps": cfg.eval_attack_steps, "cw_eval_steps": cfg.cw_eval_steps,
        "max_eval_batches": cfg.max_eval_batches,
    }


def final_test(model, loaders, cfg, device, signature):
    """Checkpoint individual evaluation conditions, including a partial AA run."""
    prepare_final_evaluation(model, device)
    spec = evaluation_spec(model, loaders, cfg, signature)
    fingerprint = digest(spec)
    meta_path = cfg.run_dir / "final_test_meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    results = dict(old.get("results", {})) if old.get("fingerprint") == fingerprint else {}
    store = make_feature_store(model, loaders.test, cfg) if cfg.plot_final_features else None
    methods = list(store.methods if store else cfg.final_test_methods)
    results = {key: value for key, value in results.items() if key in methods}

    def save(complete):
        atomic_json(meta_path, {"fingerprint": fingerprint, "evaluation_settings": spec,
                               "requested_methods": methods, "results": results,
                               "complete": complete})
        atomic_json(cfg.run_dir / "final_test.json", results)

    save(False)
    for method in methods:
        if store is not None and store.has(method):
            results[method] = store.accuracy(method)
        elif method not in results or store is not None:
            extra = {"feature_collector": store.collector(method)} if store else {}
            if method in ("autoattack_l2", "autoattack"):
                norm, eps = ("L2", cfg.l2_eps) if method == "autoattack_l2" else ("Linf", cfg.linf_eps)
                results[method] = evaluate_autoattack(
                    model, loaders.test, norm, eps, cfg, device, **extra)
            else:
                results[method] = evaluate_accuracy(
                    model, loaders.test, method, cfg, device, **extra)
        save(False)
        print(f"[final] {method}: {results[method]:.4f}%", flush=True)
    if store is not None:
        store.render(cfg)
    save(True)
    atomic_json(cfg.run_dir / "run_complete.json", {
        "training_signature": signature, "training_complete": True,
        "evaluation_complete": True, "final_test": results,
        "limited_batch_run": cfg.max_train_batches is not None or cfg.max_eval_batches is not None,
    })
    return results


def run(cfg: ExperimentConfig, options: JointOptions, eval_only=False):
    options.validate()
    if cfg.batch_size <= 0 or cfg.workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    phases = tuple(cfg.task_sequence)
    if (not phases or phases[-1] != "joint" or phases not in (("joint",), ("clean", "joint"))
            or any(cfg.task_epochs[name] < 1 for name in phases)):
        raise ValueError("Expected optional positive clean pretraining followed by joint training")
    spec = training_spec(cfg, options)
    signature = digest(spec)
    resuming = cfg.resume or eval_only
    last_path = cfg.run_dir / "last.pt"
    saved = None
    if resuming:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume: {last_path} is missing")
        # Only load checkpoints produced by a trusted local run.
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError("Joint training configuration mismatch. Keep the original batch size, "
                             "attacks, epochs, backbone and training settings, or use a new --run-name.")
        if not 0 <= int(saved["phase_index"]) < len(phases):
            raise ValueError("Invalid joint checkpoint phase")
        if eval_only and not (saved["phase_index"] == len(phases) - 1
                              and saved["next_epoch"] == cfg.task_epochs["joint"]):
            raise ValueError("--eval-only requires a completed joint training phase")
    elif cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError("Run folder is not empty; choose a new --run-name or use --resume")

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    loaders = build_dataloaders(cfg)
    split = {"train_indices": loaders.train_indices, "val_indices": loaders.val_indices,
             "test_size": len(loaders.test.dataset)}
    split_signature = digest(split)
    if saved is not None and saved["split_signature"] != split_signature:
        raise ValueError("Data split differs from the checkpoint")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(cfg)
    atomic_json(cfg.run_dir / "joint_config.json", {
        "signature": signature, "training_spec": spec, "config": config_to_dict(cfg),
        "objective": "arithmetic mean of CE over joint_conditions",
        "joint_conditions": list(options.attacks),
        "condition_weights": {name: 1.0 / len(options.attacks) for name in options.attacks},
        "has_teacher_or_replay": False,
    })
    atomic_json(cfg.run_dir / "data_split.json", {"signature": split_signature, **split})
    atomic_json(cfg.run_dir / "run_complete.json", {
        "training_signature": signature, "training_complete": False, "evaluation_complete": False})
    print(f"Resolved training: {cfg.dataset}, {cfg.backbone}, batch_size={cfg.batch_size}, "
          f"joint_conditions={options.attacks}, AA_batch={cfg.autoattack_batch_size}", flush=True)
    print(f"Output: {cfg.run_dir}", flush=True)
    model = make_backbone(cfg).to(device)
    rows = list(saved["epoch_rows"]) if saved else []
    summaries = list(saved["stage_summaries"]) if saved else []
    begin = int(saved["phase_index"]) if saved else 0

    for stage in range(begin, len(phases)):
        phase = phases[stage]
        conditions = ("clean",) if phase == "clean" else options.attacks
        optimizer = make_optimizer(model, phase, cfg)
        scheduler = make_scheduler(optimizer, phase, cfg)
        ema = EMA(model, cfg.ema_decay)
        tracker = StageResourceTracker("joint_mean", cfg, model, stage, phase, device)
        start_epoch, best_score, best_snapshot = 0, -math.inf, None
        if saved is not None and stage == saved["phase_index"]:
            model.load_state_dict(saved["model"])
            ema.load_state_dict({name: value.to(device) for name, value in saved["ema"].items()})
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            tracker.load_state_dict(saved["resource_state"])
            start_epoch = int(saved["next_epoch"])
            best_score, best_snapshot = saved["best_score"], saved["best_snapshot"]
            if not 0 <= start_epoch <= cfg.task_epochs[phase]:
                raise ValueError("Invalid checkpoint epoch")
        saved = None
        for epoch in range(start_epoch, cfg.task_epochs[phase]):
            loader = epoch_loader(loaders.train.dataset, cfg, stage, epoch)
            lr_used = optimizer.param_groups[0]["lr"]
            with tracker.timed():
                stats = train_epoch(model, ema, loader, optimizer, conditions, stage,
                                    epoch, cfg, options, device, tracker)
            del loader
            with ema.average_parameters():
                validation = evaluate_seen(model, loaders.val, conditions, cfg, device)
                # Include 0% results. Excluding them silently inflates the selector.
                score = selection_mean(validation, conditions)
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite validation score")
                if best_eligible(cfg, phase, epoch) and score > best_score:
                    best_score = score
                    best_snapshot = {
                        "model": cpu_tree(model.state_dict()), "phase": phase,
                        "epoch": epoch + 1, "combined_metric": score,
                        "validation": validation, "signature": signature,
                        "selection": "EMA full-validation arithmetic mean",
                    }
            row = {"stage": stage, "task": phase, "epoch": epoch + 1, "lr": lr_used,
                   **stats, "validation": validation, "combined_metric": score,
                   "best_eligible": best_eligible(cfg, phase, epoch),
                   "warmup_epochs": warmup_epochs(cfg, phase)}
            rows.append(row)
            scheduler.step()
            if best_snapshot is not None:
                atomic_checkpoint(cfg.run_dir / f"{phase}_best.pt", best_snapshot)
            payload = {
                "version": VERSION, "signature": signature, "split_signature": split_signature,
                "phase_index": stage, "phase": phase, "next_epoch": epoch + 1,
                "model": cpu_tree(model.state_dict()), "ema": cpu_tree(ema.state_dict()),
                "optimizer": cpu_tree(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(), "best_score": best_score,
                "best_snapshot": best_snapshot, "epoch_rows": rows,
                "stage_summaries": summaries, "resource_state": tracker.state_dict(),
            }
            # This is the authoritative commit point; JSON logs are rebuilt from
            # it on resume, even if a previous process died during log export.
            atomic_checkpoint(last_path, payload)
            del payload
            atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
            print(f"[{phase} {epoch + 1}/{cfg.task_epochs[phase]}] "
                  f"loss={stats['loss']:.5f} val={validation} mean={score:.4f} "
                  f"best={best_score:.4f} generated={stats['generated_adversarial_samples']}",
                  flush=True)
        if best_snapshot is None:
            raise RuntimeError(f"No selected checkpoint for {phase}")
        # The next phase begins from the selected EMA model, not the last epoch.
        model.load_state_dict(best_snapshot["model"])
        atomic_checkpoint(cfg.run_dir / f"{phase}_best.pt", best_snapshot)
        summary = {"stage": stage, "task": phase, "best_epoch": best_snapshot["epoch"],
                   "combined_metric": best_score, "validation": best_snapshot["validation"],
                   "conditions": list(conditions), "planned_epochs": cfg.task_epochs[phase]}
        summaries = [item for item in summaries if item["stage"] != stage] + [summary]
        atomic_json(cfg.run_dir / "stage_summaries.json", summaries)
        atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
        upsert_resource_row(cfg.run_dir, tracker.finish(replay_storage_bytes=0))
        del optimizer, scheduler, ema, tracker, best_snapshot

    # No optimizer/EMA/teacher or GPU checkpoint payload remains alive during AA.
    saved = None
    atomic_json(cfg.run_dir / "run_complete.json", {
        "training_signature": signature, "training_complete": True, "evaluation_complete": False})
    return final_test(model, loaders, cfg, device, signature)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Joint-Mean: optional clean pretraining, then all attacks jointly on each minibatch")
    add_shared_arguments(parser)
    # Explicit entry defaults: common.py's old Imagenette batch=4 cannot override them.
    parser.set_defaults(dataset="imagenette", backbone="resnet18", batch_size=32,
                        autoattack_batch_size=4, autoattack_samples=0,
                        output_root="./runs_joint", run_name="joint_mean")
    help_overrides = {
        "batch_size": "Source clean images per training batch (default: 32).",
        "backbone": "Network: resnet18 (default) or wrn; no pretrained weights.",
        "autoattack_batch_size": "Independent final AutoAttack batch (default: 4).",
        "autoattack_samples": "Balanced AA subset; default 0 evaluates the full test set.",
        "resume": "Resume from last.pt at the last complete epoch, or finish final evaluation.",
    }
    for action in parser._actions:
        if action.dest in help_overrides:
            action.help = help_overrides[action.dest]
    parser.add_argument("--clean-epochs", type=int, default=100,
                        help="Clean pretraining epochs; 0 starts joint training from scratch")
    parser.add_argument("--joint-epochs", type=int, default=90)
    parser.add_argument("--clean-lr", type=float, default=0.1)
    parser.add_argument("--joint-lr", type=float, default=None,
                        help="Default 0.01 after clean pretraining, or 0.1 from scratch")
    parser.add_argument("--joint-attacks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient norm limit; 0 disables it")
    parser.add_argument("--run-dir", help="Explicit output directory, overrides --run-name")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load the completed joint best model and evaluate without training")
    parser.add_argument("--skip-autoattack", action="store_true")
    parser.add_argument("--train-attack-steps", type=int, default=10)
    parser.add_argument("--eval-attack-steps", type=int, default=20)
    parser.add_argument("--cw-eval-steps", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--l2-eps", type=float, default=None)
    parser.add_argument("--l2-alpha", type=float, default=None)
    parser.add_argument("--linf-eps", type=float, default=None)
    parser.add_argument("--linf-alpha", type=float, default=None)
    return parser


def resolve_config(args):
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("--batch-size must be positive; --workers must be nonnegative")
    if args.clean_epochs < 0 or args.joint_epochs <= 0:
        raise ValueError("--clean-epochs must be >= 0 and --joint-epochs must be > 0")
    for name in ("max_train_batches", "max_eval_batches"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    cfg = config_from_args("joint_mean", args)
    cfg.run_directory = args.run_dir
    cfg.resume = args.resume or args.eval_only
    cfg.task_sequence = ("clean", "joint") if args.clean_epochs else ("joint",)
    cfg.task_epochs = {"clean": args.clean_epochs, "joint": args.joint_epochs}
    joint_lr = args.joint_lr if args.joint_lr is not None else (0.01 if args.clean_epochs else 0.1)
    cfg.task_lrs = {"clean": args.clean_lr, "joint": joint_lr}
    for name in ("train_attack_steps", "eval_attack_steps", "cw_eval_steps"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
        setattr(cfg, name, value)
    for name in ("patch_size", "l2_eps", "l2_alpha", "linf_eps", "linf_alpha"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if not 0 < cfg.patch_size <= cfg.image_size:
        raise ValueError("Patch size must be in 1..image_size")
    for name in ("l2_eps", "l2_alpha", "linf_eps", "linf_alpha"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if any(not math.isfinite(lr) or lr <= 0 for lr in cfg.task_lrs.values()):
        raise ValueError("Learning rates must be finite and positive")
    if cfg.val_size <= 0:
        raise ValueError("--val-size must be positive")
    if args.skip_autoattack:
        cfg.final_test_methods = tuple(name for name in cfg.final_test_methods
                                       if not name.startswith("autoattack"))
    options = JointOptions(tuple(args.joint_attacks), args.grad_clip)
    options.validate()
    return cfg, options


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg, options = resolve_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"config": config_to_dict(cfg), "joint": asdict(options),
                          "training_spec": training_spec(cfg, options)},
                         indent=2, ensure_ascii=False))
        return
    run(cfg, options, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
