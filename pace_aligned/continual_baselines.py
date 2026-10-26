"""PACE-protocol runners for the supplied DER++, GSA, TRADES and MART scripts.

Only the surrounding protocol is shared. These methods do not acquire PACE's
DTR/FAR/LACD mechanisms. See baseline_losses.py and docs/protocol.md
for exact variant definitions and changes from the supplied July scripts.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from .selection import best_eligible, selection_mean, warmup_epochs, validate_sequence
from .baseline_losses import (
    METHODS, BaselineOptions, backward_gsa, generate_kl_attack,
    mart_objective, trades_objective,
)
from .baseline_memory import ReplayMemory
from .common import (
    TASKS, EMA, StageResourceTracker, add_shared_arguments, attack_logits,
    build_dataloaders, config_from_args, config_to_dict, evaluate_seen, generate_attack,
    make_backbone, make_optimizer, make_scheduler, resolve_device, set_seed,
    training_attack_seed, upsert_resource_row, write_environment_manifest,
)
from .feature_plots import PLOT_FIELDS
from .joint import atomic_checkpoint, atomic_json, cpu_tree, digest, epoch_loader, final_test


VERSION = "four_baselines_pace_aligned_v1_20261008"
BATCH_DEFAULTS = {"cifar10": 64, "cifar100": 64, "tiny_imagenet": 16, "imagenette": 32}


def protocol_settings(cfg, options):
    values = config_to_dict(cfg)
    for name in (*PLOT_FIELDS, "resume", "device", "workers", "output_root", "run_name",
                 "run_directory", "run_dir", "data_root", "autoattack_batch_size",
                 "autoattack_samples", "autoattack_subset_seed", "final_test_methods"):
        values.pop(name, None)
    # Irrelevant baseline-only options from the common dataclass do not identify
    # these methods; architecture-inactive WRN options are also excluded.
    for name in list(values):
        if name.startswith(("sseat_",)):
            values.pop(name)
    if cfg.backbone == "resnet18":
        for name in ("depth", "widen_factor", "dropout"):
            values.pop(name, None)
    return {
        "version": VERSION, "config": values, "method_options": options.spec(),
        "selection": "EMA mean accuracy of ALL seen tasks on full validation; zeros included",
        "stage_start": "previous stage selected EMA model",
        "memory_boundary": "retain final exposure reservoir, not rewind to best epoch",
    }


def make_memory(cfg, options):
    if options.method not in ("derpp", "gsa"):
        return None
    return ReplayMemory(options.buffer_size, (3, cfg.image_size, cfg.image_size),
                        cfg.num_classes, options.method == "derpp", cfg.seed + 8257)


def train_epoch(model, ema, loader, optimizer, task, stage, epoch, cfg, options,
                memory, device, tracker=None):
    model.train()
    sums = dict(loss=0., classification=0., replay_ce=0., logit_mse=0.,
                robust_kl=0., gradient_conflicts=0, gradient_projections=0,
                examples=0, replay_examples=0, replay_batches=0, optimizer_updates=0,
                generated_current=0, generated_replay=0)
    for batch_index, batch in enumerate(loader):
        if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
            break
        images, labels = batch[0].to(device), batch[1].to(device)
        count = labels.numel()
        optimizer.zero_grad(set_to_none=True)
        seed = training_attack_seed(cfg, stage, epoch, batch_index, stream=0)
        if options.method == "trades" and options.trades_inner_loss == "kl":
            attacked = generate_kl_attack(model, images, labels, task, cfg, seed)
        else:
            attacked = generate_attack(model, images, labels, task, cfg, training=True, rng_seed=seed)
        if task != "clean":
            sums["generated_current"] += count
            if tracker is not None:
                tracker.add_adversarial_samples("current_training", count)

        replay = memory.sample(options.replay_batch_size or cfg.batch_size, device) if memory else None
        if replay is not None and options.method == "gsa":
            # Preserve the supplied GSA semantics: CURRENT attack on stored clean
            # views, not each view's historical attack and not fixed adversarial replay.
            replay["images"] = generate_attack(
                model, replay["images"], replay["labels"], task, cfg, training=True,
                rng_seed=training_attack_seed(cfg, stage, epoch, batch_index, stream=1))
            if task != "clean":
                sums["generated_replay"] += len(replay["labels"])
                if tracker is not None:
                    tracker.add_adversarial_samples("replay_training", len(replay["labels"]))

        model.train()
        zero = images.new_zeros(())
        replay_ce, mse, robust = zero, zero, zero
        if options.method in ("trades", "mart") and task != "clean":
            clean_logits = attack_logits(model, images)
            logits = attack_logits(model, attacked)
            objective = trades_objective if options.method == "trades" else mart_objective
            loss, classification, robust = objective(clean_logits, logits, labels, options.beta(task))
        else:
            logits = attack_logits(model, attacked)
            classification = F.cross_entropy(logits, labels)
            loss = classification

        if replay is not None:
            replay_logits = attack_logits(model, replay["images"])
            replay_ce = F.cross_entropy(replay_logits, replay["labels"])
            if options.method == "derpp":
                mse = F.mse_loss(replay_logits, replay["logits"])
                loss = loss + options.der_alpha * mse + options.der_beta * replay_ce
            else:
                # Report the sum for scale diagnostics; projected GSA gradients
                # are not, in general, the gradient of this scalar sum.
                loss = loss + replay_ce
            sums["replay_examples"] += len(replay["labels"])
            sums["replay_batches"] += 1

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite {options.method} loss: {task} epoch {epoch}")
        if options.method == "gsa" and replay is not None:
            diagnostics = backward_gsa(classification, replay_ce, model)
            sums["gradient_conflicts"] += int(diagnostics["gradient_conflict"])
            sums["gradient_projections"] += 1
        else:
            loss.backward()
        if options.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        ema.update(model)
        if memory is not None:
            # The stored DER logits were computed BEFORE optimizer.step; they
            # describe the same fixed adversarial pixels that enter the buffer.
            memory.add(attacked if options.method == "derpp" else images, labels, stage,
                       logits=logits if options.method == "derpp" else None)
        for name, value in (("loss", loss), ("classification", classification),
                            ("replay_ce", replay_ce), ("logit_mse", mse), ("robust_kl", robust)):
            sums[name] += float(value.detach()) * count
        sums["examples"] += count
        sums["optimizer_updates"] += 1
    if not sums["examples"]:
        raise ValueError("No training examples; check --max-train-batches and dataset size")
    for name in ("loss", "classification", "replay_ce", "logit_mse", "robust_kl"):
        sums[name] /= sums["examples"]
    sums["gradient_conflict_rate"] = sums["gradient_conflicts"] / max(1, sums["gradient_projections"])
    sums["generated_adversarial_samples"] = sums["generated_current"] + sums["generated_replay"]
    return sums


def run(cfg, options, eval_only=False):
    options.validate()
    tasks = validate_sequence(cfg.task_sequence)
    if any(cfg.task_epochs[t] <= 0 for t in tasks):
        raise ValueError("Expected clean -> patch -> pgd_l2 -> pgd with positive epoch counts")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    settings = protocol_settings(cfg, options)
    signature = digest(settings)
    last_path = cfg.run_dir / "last.pt"
    saved = None
    if cfg.resume or eval_only:
        if not last_path.is_file():
            raise FileNotFoundError(f"No aligned checkpoint at {last_path}; old July checkpoints are incompatible")
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError("Training configuration/method mismatch. Keep the original settings "
                             "(including batch and buffer size), or use a new --run-name.")
        if not 0 <= int(saved["stage"]) < len(tasks):
            raise ValueError("Invalid saved stage")
        if eval_only and not (saved["stage"] == len(tasks) - 1
                              and saved["next_epoch"] == cfg.task_epochs[tasks[-1]]):
            raise ValueError("--eval-only requires completed training of the final stage")
    elif cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; use --resume or a new --run-name")

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    loaders = build_dataloaders(cfg)
    split = {"train_indices": loaders.train_indices, "val_indices": loaders.val_indices,
             "test_size": len(loaders.test.dataset)}
    split_signature = digest(split)
    if saved is not None and saved["split_signature"] != split_signature:
        raise ValueError("Data split differs from saved run")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(cfg)
    atomic_json(cfg.run_dir / "baseline_config.json",
                {"signature": signature, "settings": settings, "config": config_to_dict(cfg)})
    atomic_json(cfg.run_dir / "data_split.json", {"signature": split_signature, **split})
    atomic_json(cfg.run_dir / "run_complete.json",
                {"training_signature": signature, "training_complete": False, "evaluation_complete": False})
    print(f"Resolved {options.method}: dataset={cfg.dataset}, backbone={cfg.backbone}, "
          f"batch_size={cfg.batch_size}, AA_batch={cfg.autoattack_batch_size}, AA_samples={cfg.autoattack_samples}",
          flush=True)
    print(f"Method options: {options.spec()}\nOutput: {cfg.run_dir}", flush=True)
    model = make_backbone(cfg).to(device)
    memory = make_memory(cfg, options)
    if memory is not None:
        print(f"CPU replay capacity: {memory.summary()['cpu_allocated_bytes'] / 2**30:.3f} GiB", flush=True)
    rows = list(saved["epoch_rows"]) if saved else []
    summaries = list(saved["stage_summaries"]) if saved else []
    begin = int(saved["stage"]) if saved else 0

    for stage in range(begin, len(tasks)):
        task = tasks[stage]
        optimizer = make_optimizer(model, task, cfg)
        scheduler = make_scheduler(optimizer, task, cfg)
        ema = EMA(model, cfg.ema_decay)
        tracker = StageResourceTracker(options.method, cfg, model, stage, task, device)
        best_score, best_snapshot, start_epoch = -math.inf, None, 0
        if saved is not None:
            model.load_state_dict(saved["model"])
            ema.load_state_dict({name: value.to(device) for name, value in saved["ema"].items()})
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            tracker.load_state_dict(saved["resource_state"])
            if memory is not None:
                if saved["memory"] is None:
                    raise ValueError("Replay baseline checkpoint is missing its memory")
                memory.load_state_dict(saved["memory"])
            start_epoch = int(saved["next_epoch"])
            best_score, best_snapshot = saved["best_score"], saved["best_snapshot"]
            if not 0 <= start_epoch <= cfg.task_epochs[task]:
                raise ValueError("Invalid saved epoch")
        saved = None
        for epoch in range(start_epoch, cfg.task_epochs[task]):
            loader = epoch_loader(loaders.train.dataset, cfg, stage, epoch)
            lr_used = optimizer.param_groups[0]["lr"]
            with tracker.timed():
                stats = train_epoch(model, ema, loader, optimizer, task, stage, epoch,
                                    cfg, options, memory, device, tracker)
            del loader
            improved = False
            with ema.average_parameters():
                validation = evaluate_seen(model, loaders.val, tasks[:stage + 1], cfg, device)
                score = selection_mean(validation, tasks[:stage + 1])
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite validation score")
                if best_eligible(cfg, task, epoch) and score > best_score:
                    improved = True
                    best_score = score
                    best_snapshot = {
                        "model": cpu_tree(model.state_dict()), "epoch": epoch + 1,
                        "stage": stage, "task": task, "signature": signature,
                        "combined_metric": score, "validation": validation,
                        "model_kind": "selected_EMA_parameters_and_BN_buffers",
                    }
            row = {"stage": stage, "task": task, "epoch": epoch + 1, "lr": lr_used,
                   **stats, "validation": validation, "combined_metric": score,
                   "best_eligible": best_eligible(cfg, task, epoch),
                   "warmup_epochs": warmup_epochs(cfg, task),
                   "selection_score": score, "memory_size": len(memory) if memory is not None else 0}
            rows.append(row)
            scheduler.step()
            if improved:
                atomic_checkpoint(cfg.run_dir / f"stage{stage}_{task}_best.pt", best_snapshot)
            payload = {
                "version": VERSION, "signature": signature, "settings": settings,
                "split_signature": split_signature, "stage": stage, "task": task,
                "next_epoch": epoch + 1, "model": cpu_tree(model.state_dict()),
                "ema": cpu_tree(ema.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(), "best_score": best_score,
                "best_snapshot": best_snapshot, "memory": memory.state_dict() if memory is not None else None,
                "epoch_rows": rows, "stage_summaries": summaries, "resource_state": tracker.state_dict(),
            }
            # Single authoritative epoch commit. JSON logs are reconstructed
            # from this payload if interruption occurs during a log export.
            atomic_checkpoint(last_path, payload)
            del payload
            atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
            print(f"[{options.method}/{task} {epoch + 1}/{cfg.task_epochs[task]}] "
                  f"loss={stats['loss']:.5f} validation={validation} mean={score:.4f} "
                  f"replay={stats['replay_examples']}", flush=True)
        if best_snapshot is None:
            raise RuntimeError(f"No best snapshot for {task}")
        model.load_state_dict(best_snapshot["model"])
        atomic_checkpoint(cfg.run_dir / f"stage{stage}_{task}_best.pt", best_snapshot)
        summary = {"stage": stage, "task": task, "best_epoch": best_snapshot["epoch"],
                   "combined_metric": best_score, "validation": best_snapshot["validation"],
                   "memory": memory.summary() if memory is not None else None}
        summaries = [entry for entry in summaries if entry["stage"] != stage] + [summary]
        atomic_json(cfg.run_dir / "stage_summaries.json", summaries)
        atomic_json(cfg.run_dir / "accuracy_matrix.json",
                    [entry["validation"] for entry in sorted(summaries, key=lambda x: x["stage"])])
        atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
        storage = memory.summary()["valid_storage_bytes"] if memory is not None else 0
        upsert_resource_row(cfg.run_dir, tracker.finish(replay_storage_bytes=storage))
        del optimizer, scheduler, ema, tracker, best_snapshot

    if memory is not None:
        atomic_json(cfg.run_dir / "memory_summary.json", memory.summary())
    memory = saved = None
    atomic_json(cfg.run_dir / "run_complete.json",
                {"training_signature": signature, "training_complete": True, "evaluation_complete": False})
    # Reuse the tested streaming AA / per-condition resume / paired-feature path.
    return final_test(model, loaders, cfg, device, signature)


def build_parser(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    parser = argparse.ArgumentParser(description=f"{method.upper()} in the current PACE protocol")
    add_shared_arguments(parser)
    parser.set_defaults(run_name="pace_aligned_v1", output_root="./runs_baselines",
                        autoattack_samples=0)
    help_overrides = {
        "batch_size": "Defaults: CIFAR 64, Tiny-ImageNet 16, Imagenette 32; saved explicitly.",
        "autoattack_samples": "Balanced AA subset size; default 0 uses the entire test set.",
        "autoattack_batch_size": "Independent AA batch; default 4 for Imagenette, min(32,batch) otherwise.",
        "resume": "Resume last.pt from its last complete epoch; old July checkpoints are incompatible.",
    }
    for action in parser._actions:
        if action.dest in help_overrides:
            action.help = help_overrides[action.dest]
    parser.add_argument("--clean-epochs", type=int, default=100)
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--clean-lr", type=float, default=0.1)
    parser.add_argument("--attack-lr", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="PACE norm limit; 0 disables clipping")
    parser.add_argument("--run-dir")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-autoattack", action="store_true")
    parser.add_argument("--selection-probe-size", type=int, default=500,
                        help="Align validation ordering to PACE monitor size; selection still uses full validation.")
    parser.add_argument("--depth", type=int, default=34)
    parser.add_argument("--widen-factor", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    for flag, default in (("train-attack-steps", 10), ("eval-attack-steps", 20), ("cw-eval-steps", 30)):
        parser.add_argument("--" + flag, type=int, default=default)
    parser.add_argument("--patch-size", type=int)
    for flag in ("l2-eps", "l2-alpha", "linf-eps", "linf-alpha"):
        parser.add_argument("--" + flag, type=float)
    if method in ("derpp", "gsa"):
        parser.add_argument("--buffer-size", type=int, default=5120,
                            help="Float32 CPU reservoir capacity; 5120 Imagenette images cost about 1.46 GiB.")
        parser.add_argument("--replay-batch-size", type=int, default=None,
                            help="Defaults to training batch size; sample without replacement.")
    if method == "derpp":
        parser.add_argument("--der-alpha", type=float, default=0.5)
        parser.add_argument("--der-beta", type=float, default=0.5)
    if method == "trades":
        parser.add_argument("--trades-inner-loss", choices=("kl", "ce"), default="kl",
                            help="kl is standard TRADES; ce retains the supplied legacy inner objective.")
    if method in ("trades", "mart"):
        parser.add_argument("--beta", type=float, help="Override all three adversarial-stage beta values.")
        parser.add_argument("--beta-patch", type=float, default=1.)
        parser.add_argument("--beta-l2", type=float, default=1.)
        parser.add_argument("--beta-linf", type=float, default=6.)
    return parser


def resolve_config(method, args):
    args.batch_size = BATCH_DEFAULTS[args.dataset] if args.batch_size is None else args.batch_size
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    if min(args.clean_epochs, args.attack_epochs) <= 0:
        raise ValueError("Epoch counts must be positive")
    if args.autoattack_batch_size is None:
        args.autoattack_batch_size = 4 if args.dataset == "imagenette" else min(32, args.batch_size)
    for name in ("max_train_batches", "max_eval_batches"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    cfg = config_from_args(method, args)
    cfg.run_directory = args.run_dir
    cfg.resume = args.resume or args.eval_only
    cfg.depth, cfg.widen_factor, cfg.dropout = args.depth, args.widen_factor, args.dropout
    if cfg.backbone == "wrn" and (cfg.depth < 10 or (cfg.depth - 4) % 6
                                  or cfg.widen_factor < 1 or not 0 <= cfg.dropout < 1):
        raise ValueError("Invalid WRN depth, width or dropout")
    cfg.task_epochs = {task: args.clean_epochs if task == "clean" else args.attack_epochs for task in TASKS}
    cfg.task_lrs = {task: args.clean_lr if task == "clean" else args.attack_lr for task in TASKS}
    cfg.ema_decay = args.ema_decay
    if not 0 <= cfg.ema_decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    if any(not math.isfinite(lr) or lr <= 0 for lr in cfg.task_lrs.values()):
        raise ValueError("Learning rates must be finite and positive")
    cfg.selection_probe_size = args.selection_probe_size
    if not 0 < cfg.selection_probe_size <= cfg.val_size:
        raise ValueError("selection-probe-size must be positive and <= val-size")
    for name in ("train_attack_steps", "eval_attack_steps", "cw_eval_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
        setattr(cfg, name, getattr(args, name))
    for name in ("patch_size", "l2_eps", "l2_alpha", "linf_eps", "linf_alpha"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if not 0 < cfg.patch_size <= cfg.image_size:
        raise ValueError("Patch size must be in 1..image_size")
    for name in ("l2_eps", "l2_alpha", "linf_eps", "linf_alpha"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.skip_autoattack:
        cfg.final_test_methods = tuple(name for name in cfg.final_test_methods if not name.startswith("autoattack"))
    fields = ("buffer_size", "replay_batch_size", "der_alpha", "der_beta", "trades_inner_loss",
              "beta_patch", "beta_l2", "beta_linf", "grad_clip")
    values = {name: getattr(args, name) for name in fields if hasattr(args, name)}
    if getattr(args, "beta", None) is not None:
        values.update(beta_patch=args.beta, beta_l2=args.beta, beta_linf=args.beta)
    options = BaselineOptions(method=method, **values)
    options.validate()
    return cfg, options


def main(method, argv=None):
    parser = build_parser(method)
    args = parser.parse_args(argv)
    try:
        cfg, options = resolve_config(method, args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"config": config_to_dict(cfg), "options": options.spec(),
                          "settings": protocol_settings(cfg, options)}, indent=2, ensure_ascii=False))
        return
    run(cfg, options, eval_only=args.eval_only)
