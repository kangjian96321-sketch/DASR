"""GSA historical-threat adaptation behind the original host command interface.

Retains the official accumulated PN-based class weighting. Fixed labels make
the new-class-only auxiliary CE inapplicable: use ONE weighted sample mean over
current plus source-attack replay instead of counting current CE twice.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dasr_aligned import continual_baselines as legacy
from dasr_aligned.common import (
    EMA, StageResourceTracker, attack_logits, build_dataloaders, config_to_dict,
    evaluate_seen, generate_attack, make_backbone, make_optimizer, make_scheduler,
    resolve_device, set_seed, training_attack_seed, upsert_resource_row,
    write_environment_manifest,
)
from dasr_aligned.joint import atomic_checkpoint, atomic_json, cpu_tree, digest, epoch_loader, final_test
from dasr_aligned.selection import best_eligible, selection_mean, validate_sequence, warmup_epochs
from . import VERSION, UPSTREAM_COMMIT
from .objective import GradientStatistics
from .memory import HistoricalMemory
from .routing import regenerate_history


@dataclass(frozen=True)
class GSAOptions:
    method: str = "gsa"
    buffer_size: int = 5120
    replay_batch_size: int | None = None
    warmup_steps: int = 5
    epsilon: float = 1e-12
    grad_clip: float = 1.0

    def validate(self):
        if self.method != "gsa" or self.buffer_size <= 0:
            raise ValueError("Invalid method or nonpositive replay capacity")
        if self.replay_batch_size is not None and self.replay_batch_size <= 0:
            raise ValueError("replay_batch_size must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.warmup_steps < 0 or not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("warmup_steps and grad_clip must be nonnegative")

    def spec(self):
        return {**asdict(self), "upstream_commit": UPSTREAM_COMMIT,
                "variant": "GSA_adapted_fixed_labels_historical_threats_single_objective",
                "stored": "clean_input_label_source_stage",
                "replay_attack": "SOURCE_stage_attack_against_CURRENT_student",
                "replay_sampling": "uniform_without_replacement_from_global_historical_reservoir",
                "reservoir_stream": "one_training_sample_pass_per_completed_stage",
                "class_space": "fixed_semantic_classes_no_task_expansion",
                "objective": "single_mean_GSA_weighted_CE_over_current_and_history",
                "current_auxiliary_CE": False,
                "mixed_batch": "concatenate_current_and_replay_logits; sample_mean",
                "rate": "accumulated_positive_gradient_magnitude/accumulated_negative_gradient_magnitude",
                "weights": "2*sigmoid(accumulated_PN_ratio-1), detached",
                "statistics": "once_per_mixed_batch_before_loss; accumulated_across_selected_stages",
                "stage_boundary": "restore_selected_EMA_model_and_same_epoch_statistics; then_commit_memory_once",
                "base_stage": "shared_base_without_reconstructed_statistics; otherwise_single_GSA_current_loss"}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = [p.relative_to(root).as_posix() for p in Path(__file__).parent.glob("*.py")]
    names += ["dasr_aligned/common.py", "dasr_aligned/joint.py", "dasr_aligned/baseline_memory.py",
              "dasr_aligned/continual_baselines.py", "dasr_aligned/selection.py"]
    names += [n for n in ("dasr_aligned/shared_base.py", "dasr_aligned/attack_protocols.py") if (root / n).is_file()]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}


def protocol_settings(cfg, options):
    settings = legacy.protocol_settings(cfg, options)
    settings.update(version=VERSION, source_sha256=source_hashes(),
                    memory_boundary="frozen_in_stage; one_pass_commit_after_stage; separate_sampling_rng",
                    gsa_statistics_boundary="restore_statistics_at_selected_best_epoch",
                    validation_coverage="all_seen_tasks_on_host_validation_loader",
                    diagnostic_eval_batch_limit=cfg.max_eval_batches)
    return settings


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_epoch(model, ema, loader, optimizer, task, stage, epoch, cfg, options,
                memory, statistics, device, tracker=None):
    model.train()
    statistics.start_stage(stage)
    memory.assert_history(stage)
    sums = {key: 0 for key in ("examples", "replay_examples", "clean_replay_examples", "replay_batches",
                               "optimizer_updates", "generated_current", "generated_replay", "gsa_updates")}
    sums.update(loss=0., current_ce=0., replay_ce=0., objective_examples=0, replay_by_source={})
    for batch_index, batch in enumerate(loader):
        if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
            break
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        attacked = generate_attack(model, images, labels, task, cfg, training=True,
                                  rng_seed=training_attack_seed(cfg, stage, epoch, batch_index, stream=0))
        if task != "clean":
            sums["generated_current"] += len(labels)
            if tracker is not None:
                tracker.add_adversarial_samples("current_training", len(labels))
        replay = memory.sample(options.replay_batch_size or cfg.batch_size, device) if stage else None
        if stage and replay is None:
            raise RuntimeError("Historical replay was requested but memory is empty")
        replay_attacked = None
        if replay is not None:
            replay_attacked, routed, generated = regenerate_history(
                model, replay, cfg.task_sequence, stage, epoch, batch_index, cfg)
            sums["generated_replay"] += generated
            sums["replay_examples"] += len(replay["labels"])
            sums["clean_replay_examples"] += len(replay["labels"]) - generated
            sums["replay_batches"] += 1
            if tracker is not None:
                tracker.add_adversarial_samples("replay_training", generated)
            for source, record in routed.items():
                aggregate = sums["replay_by_source"].setdefault(source, {"attack": record["attack"],
                                                                       "examples": 0, "generated": 0})
                aggregate["examples"] += record["examples"]
                aggregate["generated"] += record["generated"]
        model.train()
        current_logits = attack_logits(model, attacked)
        # Separate current/replay forwards retain host BN behavior; a single
        # concatenated objective controls gradients. Diagnostic CEs are detached.
        if replay is not None:
            replay_logits = attack_logits(model, replay_attacked)
            mixed_logits = torch.cat((current_logits, replay_logits), dim=0)
            mixed_labels = torch.cat((labels, replay["labels"]), dim=0)
            sums["replay_ce"] += float(F.cross_entropy(replay_logits.detach(), replay["labels"])) * len(replay["labels"])
        else:
            mixed_logits, mixed_labels = current_logits, labels
        loss = statistics.loss(mixed_logits, mixed_labels, stage)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite GSA loss: {task}, epoch {epoch}, batch {batch_index}")
        loss.backward()
        if options.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        ema.update(model)
        sums["loss"] += float(loss.detach()) * len(mixed_labels)
        sums["objective_examples"] += len(mixed_labels)
        sums["current_ce"] += float(F.cross_entropy(current_logits.detach(), labels)) * len(labels)
        sums["examples"] += len(labels)
        sums["optimizer_updates"] += 1
        sums["gsa_updates"] += 1
    if not sums["examples"]:
        raise ValueError("No training examples")
    sums["loss"] /= sums["objective_examples"]
    sums["current_ce"] /= sums["examples"]
    sums["replay_ce"] /= max(1, sums["replay_examples"])
    sums["generated_adversarial_samples"] = sums["generated_current"] + sums["generated_replay"]
    sums["gsa"] = statistics.summary()
    return sums


def run(cfg, options, eval_only=False):
    options.validate()
    device = resolve_device(cfg.device)
    if hasattr(cfg, "attack_backend"):
        from dasr_aligned.attack_protocols import validate_runtime
        validate_runtime(cfg, device)
    tasks = validate_sequence(cfg.task_sequence)
    if any(cfg.task_epochs[t] <= 0 for t in tasks) or cfg.batch_size <= 0:
        raise ValueError("Epoch counts and batch size must be positive")
    settings = protocol_settings(cfg, options)
    signature = digest(settings)
    last_path = cfg.run_dir / "last.pt"
    saved = None
    if cfg.resume or eval_only:
        if not last_path.is_file():
            raise FileNotFoundError(f"No GSA-historical checkpoint at {last_path}")
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("version") != VERSION or saved.get("signature") != signature:
            raise ValueError("GSA-historical code/configuration mismatch; other GSA checkpoints are incompatible")
        if not 0 <= int(saved["stage"]) < len(tasks) or saved.get("phase") not in ("training", "stage_complete"):
            raise ValueError("Invalid saved stage/phase")
        if eval_only and not (saved["stage"] == len(tasks) - 1 and saved["phase"] == "stage_complete"):
            raise ValueError("--eval-only requires a completed final stage")
    elif cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; use --resume or a new --run-name")
    set_seed(cfg.seed)
    cfg.device = str(device)
    loaders = build_dataloaders(cfg)
    split = {"train_indices": loaders.train_indices, "val_indices": loaders.val_indices,
             "test_size": len(loaders.test.dataset)}
    split_signature = digest(split)
    if saved is not None and saved["split_signature"] != split_signature:
        raise ValueError("Data split differs from saved run")
    model = make_backbone(cfg).to(device)
    if eval_only:
        model.load_state_dict(saved["best_snapshot"]["model"])
        return final_test(model, loaders, cfg, device, signature)
    shared = None
    if getattr(cfg, "shared_clean", False) and saved is None:
        from dasr_aligned.shared_base import acquire
        shared = acquire(cfg, loaders, model, device)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if shared is not None:
        from dasr_aligned.shared_base import record_import
        record_import(cfg, shared)
    write_environment_manifest(cfg)
    atomic_json(cfg.run_dir / "baseline_config.json",
                {"signature": signature, "settings": settings, "config": config_to_dict(cfg)})
    atomic_json(cfg.run_dir / "data_split.json", {"signature": split_signature, **split})
    atomic_json(cfg.run_dir / "run_complete.json",
                {"training_signature": signature, "training_complete": False, "evaluation_complete": False})
    print(f"{VERSION}: source-attack replay; single objective; host validation policy; output={cfg.run_dir}", flush=True)
    print(json.dumps(options.spec(), ensure_ascii=False), flush=True)
    memory = HistoricalMemory(options.buffer_size, (3, cfg.image_size, cfg.image_size), cfg.num_classes, cfg.seed + 8257)
    statistics = GradientStatistics(cfg.num_classes, options.warmup_steps, options.epsilon).to(device)
    rows = list(saved["epoch_rows"]) if saved else []
    summaries = list(saved["stage_summaries"]) if saved else []
    begin = int(saved["stage"]) if saved else 0
    if saved is not None and saved["phase"] == "stage_complete":
        if saved["next_epoch"] != cfg.task_epochs[tasks[begin]]:
            raise ValueError("Completed stage has inconsistent epoch count")
        model.load_state_dict(saved["model"])
        memory.load_state_dict(saved["memory"])
        statistics.load_state_dict(saved["gsa_statistics"])
        if memory.last_committed_stage != begin:
            raise ValueError("Completed stage has not committed its memory")
        restore_rng(saved["rng"])
        begin += 1
        saved = None
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
            memory.load_state_dict(saved["memory"])
            statistics.load_state_dict(saved["gsa_statistics"])
            start_epoch = int(saved["next_epoch"])
            best_score, best_snapshot = saved["best_score"], saved["best_snapshot"]
            if not 0 <= start_epoch <= cfg.task_epochs[task]:
                raise ValueError("Invalid saved epoch")
            restore_rng(saved["rng"])
        saved = None
        memory.assert_history(stage)
        statistics.start_stage(stage)

        def checkpoint(next_epoch, phase="training"):
            return {"version": VERSION, "signature": signature, "settings": settings,
                    "split_signature": split_signature, "stage": stage, "task": task, "phase": phase,
                    "next_epoch": next_epoch, "model": cpu_tree(model.state_dict()),
                    "ema": cpu_tree(ema.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(), "best_score": best_score,
                    "best_snapshot": best_snapshot, "memory": memory.state_dict(),
                    "gsa_statistics": cpu_tree(statistics.state_dict()), "rng": rng_state(),
                    "epoch_rows": rows, "stage_summaries": summaries, "resource_state": tracker.state_dict()}

        if stage == 0 and shared is not None:
            model.load_state_dict(shared["model"])
            best_score = shared["accuracy"]
            best_snapshot = {"model": cpu_tree(model.state_dict()), "epoch": shared["epoch"] + 1,
                             "stage": 0, "task": task, "signature": signature,
                             "gsa_statistics": cpu_tree(statistics.state_dict()),
                             "combined_metric": best_score, "validation": {task: best_score},
                             "shared_base_signature": shared["signature"],
                             "shared_base_model_sha256": shared["model_sha256"]}
            ema = EMA(model, cfg.ema_decay)
            start_epoch = cfg.task_epochs[task]
            atomic_checkpoint(last_path, checkpoint(start_epoch))
            shared = None
        for epoch in range(start_epoch, cfg.task_epochs[task]):
            loader = epoch_loader(loaders.train.dataset, cfg, stage, epoch)
            lr_used = optimizer.param_groups[0]["lr"]
            with tracker.timed():
                stats = train_epoch(model, ema, loader, optimizer, task, stage, epoch,
                                    cfg, options, memory, statistics, device, tracker)
            del loader
            improved = False
            with ema.average_parameters():
                validation = evaluate_seen(model, loaders.val, tasks[:stage + 1], cfg, device)
                score = selection_mean(validation, tasks[:stage + 1])
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite validation score")
                if best_eligible(cfg, task, epoch) and score > best_score:
                    improved, best_score = True, score
                    best_snapshot = {"model": cpu_tree(model.state_dict()), "epoch": epoch + 1,
                                     "stage": stage, "task": task, "signature": signature,
                                     "gsa_statistics": cpu_tree(statistics.state_dict()),
                                     "combined_metric": score, "validation": validation,
                                     "model_kind": "selected_EMA_parameters_and_BN_buffers"}
            rows.append({"stage": stage, "task": task, "epoch": epoch + 1, "lr": lr_used,
                         **stats, "validation": validation, "combined_metric": score,
                         "best_eligible": best_eligible(cfg, task, epoch),
                         "warmup_epochs": warmup_epochs(cfg, task), "memory_size": len(memory),
                         "memory": memory.summary(), "validation_dataset_size": len(loaders.val.dataset), "diagnostic_eval_batch_limit": cfg.max_eval_batches})
            scheduler.step()
            if improved:
                atomic_checkpoint(cfg.run_dir / f"stage{stage}_{task}_best.pt", best_snapshot)
            atomic_checkpoint(last_path, checkpoint(epoch + 1))
            atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
            print(f"[GSA-historical/{task} {epoch + 1}/{cfg.task_epochs[task]}] "
                  f"loss={stats['loss']:.5f} mean={score:.4f} replay={stats['replay_examples']}", flush=True)
        if best_snapshot is None:
            raise RuntimeError(f"No eligible best snapshot for {task}")
        # Both selected model and accumulated weighting state come from the same
        # selected epoch. Epoch logs retain the full actually executed history.
        model.load_state_dict(best_snapshot["model"])
        statistics.load_state_dict(best_snapshot["gsa_statistics"])
        with tracker.timed():
            added = memory.commit_stage(loaders.train_eval, stage, cfg, device)
        summary = {"stage": stage, "task": task, "best_epoch": best_snapshot["epoch"],
                   "combined_metric": best_score, "validation": best_snapshot["validation"],
                   "committed_stage_sample_pairs": added,
                   "memory": memory.summary(), "gsa": statistics.summary()}
        summaries = [entry for entry in summaries if entry["stage"] != stage] + [summary]
        atomic_checkpoint(cfg.run_dir / f"stage{stage}_{task}_best.pt", best_snapshot)
        atomic_json(cfg.run_dir / "stage_summaries.json", summaries)
        atomic_json(cfg.run_dir / "accuracy_matrix.json",
                    [entry["validation"] for entry in sorted(summaries, key=lambda x: x["stage"])])
        atomic_json(cfg.run_dir / "epoch_metrics.json", rows)
        upsert_resource_row(cfg.run_dir, tracker.finish(replay_storage_bytes=memory.summary()["valid_storage_bytes"]))
        # Durable stage-complete commit prevents duplicate memory insertion on resume.
        atomic_checkpoint(last_path, checkpoint(cfg.task_epochs[task], phase="stage_complete"))
        del optimizer, scheduler, ema, tracker, best_snapshot
    atomic_json(cfg.run_dir / "memory_summary.json", memory.summary())
    atomic_json(cfg.run_dir / "gsa_statistics.json", statistics.summary())
    atomic_json(cfg.run_dir / "run_complete.json",
                {"training_signature": signature, "training_complete": True, "evaluation_complete": False})
    del memory, statistics
    return final_test(model, loaders, cfg, device, signature)


def from_host_options(options):
    return GSAOptions(buffer_size=options.buffer_size, replay_batch_size=options.replay_batch_size,
                      grad_clip=options.grad_clip)
