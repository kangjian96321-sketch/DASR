"""F/G decomposition and historical-adversarial replay sources for DASR.

The monitor cache is always established from each task's selected checkpoint.
Training caches are either eager, absent, or built on first historical use.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import torch

from dasr_aligned.common import (
    attack_logits,
    base_prediction,
    class_balanced_indices,
    generate_attack,
    isolated_random_seed,
    make_backbone,
    tensor_storage_bytes,
)


SOURCE_MODES = (
    "historical",
    "regenerated",
    "fg_adaptive",
    "fg_adaptive_lazy",
)
EAGER_TRAINING_CACHE_MODES = {"historical", "fg_adaptive"}
ADAPTIVE_SOURCE_MODES = {"fg_adaptive", "fg_adaptive_lazy"}
FAR_ATTACK_CATEGORIES = ("history_monitor", "replay_monitor")


def register_tracker(tracker) -> None:
    if tracker is None:
        return
    for name in FAR_ATTACK_CATEGORIES:
        tracker.generated.setdefault(name, 0)


def decompose(a_ss: float, a_ts: float, a_tt: float,
              component_threshold_pp: float = 0.0) -> dict:
    """Signed paired accuracy decomposition, in percentage points."""
    values = (a_ss, a_ts, a_tt)
    if not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 100.0
               for value in values):
        raise ValueError("F/G accuracies must be finite percentages in [0, 100]")
    if not math.isfinite(component_threshold_pp) or component_threshold_pp < 0:
        raise ValueError("component_threshold_pp must be finite and nonnegative")
    forgetting = float(a_ss) - float(a_ts)
    gap = float(a_ts) - float(a_tt)
    degradation = float(a_ss) - float(a_tt)
    f_positive = max(forgetting - component_threshold_pp, 0.0)
    g_positive = max(gap - component_threshold_pp, 0.0)
    historical_probability = (
        f_positive / (f_positive + g_positive)
        if f_positive + g_positive > 0 else 0.0
    )
    return {
        "a_ss": float(a_ss),
        "a_ts": float(a_ts),
        "a_tt": float(a_tt),
        "historical_forgetting_pp": forgetting,
        "attack_reoptimization_gap_pp": gap,
        "degradation_pp": degradation,
        "historical_probability": historical_probability,
        "regenerated_probability": 1.0 - historical_probability,
        "identity_residual_pp": degradation - forgetting - gap,
    }


def source_probabilities(mode: str, metrics: Mapping[str, float] | None) -> tuple[float, float]:
    if mode == "historical":
        return 1.0, 0.0
    if mode == "regenerated":
        return 0.0, 1.0
    if mode not in ADAPTIVE_SOURCE_MODES:
        raise ValueError(f"Unknown replay source mode: {mode}")
    probability = float((metrics or {}).get("historical_probability", 0.0))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Invalid historical replay probability")
    return probability, 1.0 - probability


def choose_source(mode: str, metrics: Mapping[str, float] | None, seed: int) -> str:
    historical_probability, _ = source_probabilities(mode, metrics)
    return (
        "historical"
        if random.Random(int(seed)).random() < historical_probability
        else "regenerated"
    )


def eager_training_cache(mode: str) -> bool:
    if mode not in SOURCE_MODES:
        raise ValueError(f"Unknown replay source mode: {mode}")
    return mode in EAGER_TRAINING_CACHE_MODES


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _attack_seed(cfg, task: str) -> int:
    return cfg.evaluation_seed + {
        "clean": 0,
        "patch": 101,
        "cw_l2": 211,
        "pgd_l2": 307,
        "pgd": 401,
        "fgsm": 503,
        "bim": 607,
        "mifgsm": 701,
    }.get(task, 997)


class FARHistoryBank:
    """Fixed monitor histories and optional bounded training histories."""

    VERSION = "far_history_v2_balanced_reused_validation"

    def __init__(self, path: Path, cfg, loaders, device: torch.device,
                 monitor_size: int = 500, training_size: int = 500,
                 cache_dtype: str = "float32", smoke: bool = False):
        if monitor_size <= 0 or training_size <= 0:
            raise ValueError("FAR monitor/training cache sizes must be positive")
        if cache_dtype not in ("float32", "uint8"):
            raise ValueError("cache_dtype must be float32 or uint8")
        self.path = Path(path)
        self.cfg = cfg
        self.device = device
        self.cache_dtype = cache_dtype
        split_payload = {
            "train": list(map(int, loaders.train_indices)),
            "validation": list(map(int, loaders.val_indices)),
        }
        split_signature = hashlib.sha256(
            json.dumps(split_payload, separators=(",", ":")).encode()
        ).hexdigest()
        if not smoke and set(split_payload["train"]) & set(split_payload["validation"]):
            raise ValueError("FAR monitor and replay splits must be disjoint")
        self.protocol = {
            "version": self.VERSION,
            "dataset": cfg.dataset,
            "split_signature": split_signature,
            "monitor_size": int(monitor_size),
            "training_size": int(training_size),
            "sampling": "deterministic_class_balanced",
            "monitor_source": "full_validation_prefix",
            "fresh_monitor_source": "reused_full_validation",
            "cache_dtype": cache_dtype,
            "batch_size": int(cfg.batch_size),
            "evaluation_seed": int(cfg.evaluation_seed),
            "linf_eps": float(cfg.linf_eps),
            "linf_alpha": float(cfg.linf_alpha),
            "l2_eps": float(cfg.l2_eps),
            "l2_alpha": float(cfg.l2_alpha),
            "patch_size": int(cfg.patch_size),
            "train_attack_steps": int(cfg.train_attack_steps),
            "eval_attack_steps": int(cfg.eval_attack_steps),
            "cw_eval_steps": int(cfg.cw_eval_steps),
        }
        if self.path.exists():
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
            if payload.get("protocol") != self.protocol:
                raise ValueError(
                    "FAR history cache is incompatible with this run; use a new run directory"
                )
            self.panels = payload["panels"]
            self.tasks = payload["tasks"]
        else:
            replay_count = min(training_size, len(loaders.train_eval.dataset))
            replay_indices = class_balanced_indices(
                loaders.train_eval.dataset,
                replay_count,
                cfg.num_classes,
                cfg.split_seed + 7391,
            )
            monitor_count = min(monitor_size, len(loaders.selection_val.dataset))
            if cfg.max_eval_batches is not None:
                monitor_count = min(monitor_count, cfg.max_eval_batches * cfg.batch_size)
            if not smoke and monitor_count != monitor_size:
                raise ValueError(
                    f"Requested FAR monitor size {monitor_size}, but only "
                    f"{monitor_count} balanced validation samples are available"
                )
            if not smoke and replay_count != training_size:
                raise ValueError(
                    f"Requested FAR history size {training_size}, but only "
                    f"{replay_count} training samples are available"
                )
            self.panels = {
                "monitor": self._make_panel(
                    loaders.selection_val.dataset, list(range(monitor_count))
                ),
                "training": self._make_panel(
                    loaders.train_eval.dataset, replay_indices
                ),
            }
            self.tasks = {}
            self.save()

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images.detach().clamp(0, 1).cpu()
        if self.cache_dtype == "uint8":
            return images.mul(255).round().to(torch.uint8)
        return images.float().clone()

    def _encode_adversarial(self, images: torch.Tensor,
                            stored_clean: torch.Tensor) -> torch.Tensor:
        if self.cache_dtype == "float32":
            return images.detach().clamp(0, 1).float().cpu().clone()
        # Round the pixel delta toward the clean point. A normal uint8 round
        # can enlarge individual coordinates and violate an L2 threat budget.
        clean_255 = stored_clean.cpu().float()
        delta_255 = images.detach().clamp(0, 1).cpu().mul(255) - clean_255
        encoded = clean_255 + delta_255.sign() * delta_255.abs().floor()
        return encoded.clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _decode(images: torch.Tensor, device: torch.device) -> torch.Tensor:
        result = images.to(device).float()
        return result.div(255) if images.dtype == torch.uint8 else result

    def _make_panel(self, dataset, indices: Sequence[int]) -> dict:
        if not indices:
            raise ValueError("FAR panel is empty")
        images, labels = [], []
        for index in indices:
            sample = dataset[int(index)]
            images.append(self._encode(sample[0]))
            labels.append(int(sample[1]))
        return {
            "images": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
            "dataset_indices": torch.tensor(indices, dtype=torch.long),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({
            "protocol": self.protocol,
            "panels": self.panels,
            "tasks": self.tasks,
        }, temporary)
        temporary.replace(self.path)

    def state_dict(self) -> dict:
        """Panels and built caches; unbuilt adversarial caches still need best weights."""
        return {
            "protocol": self.protocol,
            "panels": self.panels,
            "tasks": self.tasks,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("protocol") != self.protocol:
            raise ValueError("Resume checkpoint has an incompatible FAR history protocol")
        self.panels = state["panels"]
        self.tasks = state["tasks"]
        self.save()

    def storage_bytes(self) -> int:
        return tensor_storage_bytes(self.panels, self.tasks)

    def summary(self) -> dict:
        return {
            "protocol": self.protocol,
            "tasks": {
                task: {
                    "checkpoint": entry["checkpoint"],
                    "model_sha256": entry["model_sha256"],
                    "baseline_accuracy": entry["baseline_accuracy"],
                    "training_cache_ready": "training_adv" in entry,
                }
                for task, entry in self.tasks.items()
            },
            "storage_bytes": self.storage_bytes(),
        }

    def _generate(self, model, panel_name: str, task: str, training: bool,
                  tracker=None) -> tuple[torch.Tensor, torch.Tensor]:
        panel = self.panels[panel_name]
        adversarial, correct = [], []
        category = "memory_update" if training else "history_monitor"
        was_training = model.training
        model.eval()
        with isolated_random_seed(_attack_seed(self.cfg, task) + (10000 if training else 0),
                                  self.device):
            for start in range(0, len(panel["labels"]), self.cfg.batch_size):
                stored_clean = panel["images"][start:start + self.cfg.batch_size]
                clean = self._decode(stored_clean, self.device)
                labels = panel["labels"][start:start + self.cfg.batch_size].to(self.device)
                with torch.enable_grad():
                    attacked = generate_attack(
                        model, clean, labels, task, self.cfg, training=training
                    )
                stored_adv = self._encode_adversarial(attacked, stored_clean)
                with torch.no_grad():
                    prediction = base_prediction(
                        attack_logits(model, self._decode(stored_adv, self.device)),
                        self.cfg.num_classes,
                    )
                adversarial.append(stored_adv)
                correct.append(prediction.eq(labels).cpu())
                if tracker is not None and task != "clean":
                    tracker.add_adversarial_samples(category, len(labels))
        model.train(was_training)
        return torch.cat(adversarial), torch.cat(correct)

    def establish_task(self, task: str, model, checkpoint_name: str,
                       tracker=None, build_training: bool = False,
                       include_timing: bool = True) -> float:
        register_tracker(tracker)
        model_sha256 = _model_digest(model)
        if task in self.tasks:
            entry = self.tasks[task]
            if entry["model_sha256"] != model_sha256:
                raise ValueError(
                    f"Historical selected model changed for task {task}; use a new run"
                )
            if build_training and "training_adv" not in entry:
                context = tracker.timed() if tracker is not None and include_timing else nullcontext()
                with context:
                    training_adv, _ = self._generate(
                        model, "training", task, training=True, tracker=tracker
                    )
                entry["training_adv"] = training_adv
                self.save()
            return float(entry["baseline_accuracy"])

        context = tracker.timed() if tracker is not None and include_timing else nullcontext()
        with context:
            monitor_adv, monitor_correct = self._generate(
                model, "monitor", task, training=False, tracker=tracker
            )
            entry = {
                "checkpoint": str(checkpoint_name),
                "model_sha256": model_sha256,
                "monitor_adv": monitor_adv,
                "baseline_correct": monitor_correct,
                "baseline_accuracy": 100.0 * int(monitor_correct.sum()) / len(monitor_correct),
            }
            if build_training:
                training_adv, _ = self._generate(
                    model, "training", task, training=True, tracker=tracker
                )
                entry["training_adv"] = training_adv
            self.tasks[task] = entry
            self.save()
        return float(entry["baseline_accuracy"])

    def ensure_training_cache(self, task: str, tracker=None, active_model=None) -> bool:
        """Build once on first historical use; return True iff newly built."""
        if task not in self.tasks:
            raise RuntimeError(f"Missing FAR monitor history for {task}")
        if "training_adv" in self.tasks[task]:
            return False
        if task == "clean":
            # Clean replay is the identity transform, independent of model weights.
            # This also supports a copied resume file without a separate Clean best.
            self.tasks[task]["training_adv"] = self.panels["training"]["images"].clone()
            self.save()
            return True
        checkpoint = self.path.parent / self.tasks[task]["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Lazy cache needs the selected checkpoint: {checkpoint}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" not in payload:
            raise ValueError(f"Lazy cache checkpoint has no native model state: {checkpoint}")
        if active_model is None:
            historical_model = make_backbone(self.cfg).to(self.device)
            historical_model.load_state_dict(payload["model"])
            self.establish_task(
                task, historical_model, checkpoint.name, tracker=tracker,
                build_training=True, include_timing=False,
            )
            del historical_model
        else:
            # Reuse the active model object to avoid holding student, teacher,
            # and a third full historical model in GPU memory. Optimizer/EMA
            # objects retain their own state; all parameters and buffers are
            # restored before the student forward pass.
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in active_model.state_dict().items()
            }
            was_training = active_model.training
            try:
                active_model.load_state_dict(payload["model"])
                self.establish_task(
                    task, active_model, checkpoint.name, tracker=tracker,
                    build_training=True, include_timing=False,
                )
            finally:
                active_model.load_state_dict(current_state)
                active_model.train(was_training)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def sample(self, task: str, count: int, seed: int) -> tuple[torch.Tensor, ...]:
        if task not in self.tasks or "training_adv" not in self.tasks[task]:
            raise RuntimeError(f"Historical training cache is unavailable for {task}")
        panel = self.panels["training"]
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randint(len(panel["labels"]), (int(count),), generator=generator)
        return (
            self._decode(panel["images"][indices], self.device),
            self._decode(self.tasks[task]["training_adv"][indices], self.device),
            panel["labels"][indices].to(self.device),
        )

    def fixed_historical_accuracies(
            self, model, tasks: Sequence[str]) -> dict[str, float]:
        """Evaluate one reference model on every task's fixed monitor history."""
        panel = self.panels["monitor"]
        was_training = model.training
        model.eval()
        result = {}
        try:
            with torch.no_grad():
                for task in tasks:
                    if task not in self.tasks:
                        raise RuntimeError(f"Missing FAR monitor history for {task}")
                    correct = total = 0
                    for start in range(0, len(panel["labels"]), self.cfg.batch_size):
                        labels = panel["labels"][
                            start:start + self.cfg.batch_size
                        ].to(self.device)
                        historical = self._decode(
                            self.tasks[task]["monitor_adv"][
                                start:start + self.cfg.batch_size
                            ],
                            self.device,
                        )
                        prediction = base_prediction(
                            attack_logits(model, historical), self.cfg.num_classes
                        )
                        correct += int(prediction.eq(labels).sum())
                        total += len(labels)
                    if total == 0:
                        raise RuntimeError(f"Empty FAR monitor history for {task}")
                    result[task] = 100.0 * correct / total
        finally:
            model.train(was_training)
        return result

    def evaluate_validation(
            self, model, loader, tasks: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Evaluate full validation and its balanced FAR prefix in one pass.

        Each attack is generated exactly once.  The full-set result is used for
        checkpoint selection, while the prefix result supplies ``a_tt`` for the
        F/G decomposition.  Under uint8 caching, only the prefix prediction is
        recomputed after the same inward quantization used by historical data;
        no second attack is generated.
        """
        panel = self.panels["monitor"]
        monitor_size = len(panel["labels"])
        full_accuracies: dict[str, float] = {}
        monitor_accuracies: dict[str, float] = {}
        was_training = model.training
        model.eval()
        try:
            for task in tasks:
                full_correct = full_total = 0
                monitor_correct = monitor_total = 0
                with isolated_random_seed(_attack_seed(self.cfg, task), self.device):
                    for batch_index, batch in enumerate(loader):
                        if (self.cfg.max_eval_batches is not None
                                and batch_index >= self.cfg.max_eval_batches):
                            break
                        clean = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        with torch.enable_grad():
                            attacked = generate_attack(
                                model, clean, labels, task, self.cfg, training=False
                            )
                        with torch.no_grad():
                            prediction = base_prediction(
                                attack_logits(model, attacked), self.cfg.num_classes
                            )
                        matches = prediction.eq(labels)
                        full_correct += int(matches.sum())
                        full_total += len(labels)

                        remaining = monitor_size - monitor_total
                        take = min(max(remaining, 0), len(labels))
                        if take <= 0:
                            continue

                        panel_slice = slice(monitor_total, monitor_total + take)
                        expected_labels = panel["labels"][panel_slice].to(self.device)
                        expected_clean = self._decode(
                            panel["images"][panel_slice], self.device
                        )
                        if not torch.equal(labels[:take], expected_labels):
                            raise RuntimeError(
                                "Full-validation prefix labels do not match the "
                                "FAR monitor panel"
                            )
                        if not torch.allclose(
                                clean[:take], expected_clean, atol=1.0 / 255.0,
                                rtol=0.0):
                            raise RuntimeError(
                                "Full-validation prefix images do not match the "
                                "FAR monitor panel"
                            )

                        if self.cache_dtype == "float32":
                            monitor_prediction = prediction[:take]
                        else:
                            quantized = self._decode(
                                self._encode_adversarial(
                                    attacked[:take], panel["images"][panel_slice]
                                ),
                                self.device,
                            )
                            with torch.no_grad():
                                monitor_prediction = base_prediction(
                                    attack_logits(model, quantized),
                                    self.cfg.num_classes,
                                )
                        monitor_correct += int(
                            monitor_prediction.eq(expected_labels).sum()
                        )
                        monitor_total += take

                if full_total == 0:
                    raise RuntimeError(f"Empty validation set for {task}")
                if monitor_total != monitor_size:
                    raise RuntimeError(
                        f"Expected {monitor_size} FAR monitor samples for {task}, "
                        f"received {monitor_total}"
                    )
                full_accuracies[task] = 100.0 * full_correct / full_total
                monitor_accuracies[task] = (
                    100.0 * monitor_correct / monitor_total
                )
        finally:
            model.train(was_training)
        return full_accuracies, monitor_accuracies

    def evaluate(self, model, tasks: Sequence[str],
                 fresh_accuracies: Mapping[str, float], tracker=None,
                 component_threshold_pp: float = 0.0,
                 baseline_accuracies: Mapping[str, float] | None = None) -> dict:
        """Evaluate fixed histories and reuse fresh validation accuracies."""
        register_tracker(tracker)
        panel = self.panels["monitor"]
        was_training = model.training
        model.eval()
        result = {}
        try:
            for task in tasks:
                if task not in self.tasks:
                    raise RuntimeError(f"Missing FAR monitor history for {task}")
                if task not in fresh_accuracies:
                    raise RuntimeError(f"Missing reused fresh accuracy for {task}")
                historical_correct = total = 0
                with torch.no_grad():
                    for start in range(0, len(panel["labels"]), self.cfg.batch_size):
                        labels = panel["labels"][
                            start:start + self.cfg.batch_size
                        ].to(self.device)
                        historical = self._decode(
                            self.tasks[task]["monitor_adv"][
                                start:start + self.cfg.batch_size
                            ],
                            self.device,
                        )
                        historical_prediction = base_prediction(
                            attack_logits(model, historical), self.cfg.num_classes
                        )
                        historical_correct += int(
                            historical_prediction.eq(labels).sum()
                        )
                        total += len(labels)
                if baseline_accuracies is None:
                    baseline_accuracy = float(
                        self.tasks[task]["baseline_accuracy"]
                    )
                    reference_mode = "original_task_best"
                else:
                    if task not in baseline_accuracies:
                        raise RuntimeError(
                            f"Missing previous-stage reference for {task}"
                        )
                    baseline_accuracy = float(baseline_accuracies[task])
                    reference_mode = "previous_stage_best"
                result[task] = decompose(
                    baseline_accuracy,
                    100.0 * historical_correct / total,
                    float(fresh_accuracies[task]),
                    component_threshold_pp,
                )
                original_baseline = float(
                    self.tasks[task]["baseline_accuracy"]
                )
                cumulative = decompose(
                    original_baseline,
                    result[task]["a_ts"],
                    result[task]["a_tt"],
                    component_threshold_pp,
                )
                # a_ss is retained for output/checkpoint compatibility. Under
                # the new mode it is the stage reference a_(t-1,s), not the
                # original task-own checkpoint accuracy.
                result[task]["a_reference"] = baseline_accuracy
                result[task]["stage_reference_accuracy"] = baseline_accuracy
                result[task]["original_task_baseline_accuracy"] = original_baseline
                result[task]["reference_mode"] = reference_mode
                result[task]["routing_basis"] = "stage_decomposition"
                result[task]["stage_historical_forgetting_pp"] = result[task][
                    "historical_forgetting_pp"
                ]
                result[task]["stage_degradation_pp"] = result[task][
                    "degradation_pp"
                ]
                result[task]["stage_historical_probability"] = result[task][
                    "historical_probability"
                ]
                result[task]["stage_regenerated_probability"] = result[task][
                    "regenerated_probability"
                ]
                result[task]["cumulative_historical_forgetting_pp"] = cumulative[
                    "historical_forgetting_pp"
                ]
                result[task]["cumulative_degradation_pp"] = cumulative[
                    "degradation_pp"
                ]
                result[task]["cumulative_historical_probability"] = cumulative[
                    "historical_probability"
                ]
                result[task]["cumulative_regenerated_probability"] = cumulative[
                    "regenerated_probability"
                ]
                result[task]["cumulative_identity_residual_pp"] = cumulative[
                    "identity_residual_pp"
                ]
                result[task]["monitor_samples"] = total
                result[task]["source_checkpoint"] = self.tasks[task]["checkpoint"]
        finally:
            model.train(was_training)
        return result
