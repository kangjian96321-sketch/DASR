"""Losses for the user's four baselines, with explicit research variants.

TRADES outer loss and MART retain their defining objectives. KL inner attacks
extend TRADES to the project's Patch/L2/Linf constraints; this is not a
bit-for-bit copy of the official L2 solver. GSA below preserves the supplied
gradient projection plus reference-gradient implementation, not an asserted
reproduction of an unidentified GSA paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .common import attack_logits, isolated_random_seed


METHODS = ("derpp", "gsa", "trades", "mart")


@dataclass(frozen=True)
class BaselineOptions:
    method: str
    buffer_size: int = 5120
    replay_batch_size: int | None = None
    der_alpha: float = 0.5
    der_beta: float = 0.5
    trades_inner_loss: str = "kl"
    beta_patch: float = 1.0
    beta_l2: float = 1.0
    beta_linf: float = 6.0
    grad_clip: float = 1.0

    def validate(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown baseline: {self.method}")
        if self.buffer_size <= 0:
            raise ValueError("--buffer-size must be positive")
        if self.replay_batch_size is not None and self.replay_batch_size <= 0:
            raise ValueError("--replay-batch-size must be positive")
        if self.trades_inner_loss not in ("kl", "ce"):
            raise ValueError("--trades-inner-loss must be kl or ce")
        for name in ("der_alpha", "der_beta", "beta_patch", "beta_l2", "beta_linf", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def beta(self, task):
        return {"clean": 0., "patch": self.beta_patch,
                "pgd_l2": self.beta_l2, "pgd": self.beta_linf}[task]

    def spec(self):
        result = {"method": self.method, "grad_clip": self.grad_clip}
        if self.method in ("derpp", "gsa"):
            result.update(buffer_size=self.buffer_size, replay_batch_size=self.replay_batch_size,
                          sampling="reservoir_over_training_exposures",
                          stage_memory="keep_end_of_stage_memory_when_loading_selected_model")
        if self.method == "derpp":
            result.update(alpha=self.der_alpha, beta=self.der_beta,
                          stored="fixed_adversarial_view_label_preupdate_logit",
                          replay="one_shared_batch_for_MSE_and_CE")
        if self.method == "gsa":
            result.update(variant="provided_gradient_projection_plus_reference",
                          stored="augmented_clean_view_and_label",
                          replay="regenerate_CURRENT_stage_attack_on_clean_memory")
        if self.method in ("trades", "mart"):
            result.update(beta_patch=self.beta_patch, beta_l2=self.beta_l2, beta_linf=self.beta_linf)
        if self.method == "trades":
            result["inner_loss"] = self.trades_inner_loss
            result["kl_solver"] = "projected_ascent_with_DASR_budgets_steps_and_stepsizes"
        return result


def trades_objective(natural_logits, adversarial_logits, labels, beta):
    natural = F.cross_entropy(natural_logits, labels)
    # The clean branch participates in outer optimization, as in TRADES.
    robust = F.kl_div(F.log_softmax(adversarial_logits, 1),
                      F.softmax(natural_logits, 1), reduction="batchmean")
    return natural + beta * robust, natural, robust


def mart_objective(natural_logits, adversarial_logits, labels, beta):
    probabilities = F.softmax(adversarial_logits, 1)
    alternatives = probabilities.topk(2, dim=1).indices
    wrong_class = torch.where(alternatives[:, 0].eq(labels),
                              alternatives[:, 1], alternatives[:, 0])
    boosted = F.cross_entropy(adversarial_logits, labels)
    boosted = boosted + F.nll_loss(torch.log(1.0001 - probabilities + 1e-12), wrong_class)
    natural_probabilities = F.softmax(natural_logits, 1)
    confidence = natural_probabilities.gather(1, labels[:, None]).squeeze(1)
    divergence = F.kl_div(torch.log(probabilities + 1e-12), natural_probabilities,
                          reduction="none").sum(1)
    robust = (divergence * (1.0000001 - confidence)).mean()
    return boosted + beta * robust, boosted, robust


def project_l2(delta, epsilon):
    norms = delta.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
    scale = torch.clamp(epsilon / norms, max=1.)
    return delta * scale[:, None, None, None]


def generate_kl_attack(model, images, labels, task, cfg, rng_seed):
    """Maximize KL(p(clean)||p(adversarial)) inside the current threat set."""
    if task == "clean":
        return images.detach()
    if task not in ("patch", "pgd_l2", "pgd"):
        raise ValueError(f"Unsupported TRADES constraint: {task}")
    clean = images.detach()
    was_training = model.training
    try:
        with isolated_random_seed(rng_seed, images.device):
            model.eval()
            with torch.no_grad():
                reference = F.softmax(attack_logits(model, clean), dim=1)
            if task == "patch":
                _, _, height, width = clean.shape
                side = cfg.patch_size
                if not 0 < side <= min(height, width):
                    raise ValueError("Invalid Patch size")
                mask = torch.zeros_like(clean, dtype=torch.bool)
                for index in range(len(clean)):
                    top = int(torch.randint(height - side + 1, (1,), device=clean.device))
                    left = int(torch.randint(width - side + 1, (1,), device=clean.device))
                    mask[index, :, top:top + side, left:left + side] = True
                adversarial = torch.where(mask, torch.rand_like(clean), clean)
            else:
                # Small random initialization avoids zero KL gradient at x_adv=x.
                delta = 0.001 * torch.randn_like(clean)
                delta = (project_l2(delta, cfg.l2_eps) if task == "pgd_l2"
                         else delta.clamp(-cfg.linf_eps, cfg.linf_eps))
                adversarial = (clean + delta).clamp(0, 1)
            for _ in range(cfg.train_attack_steps):
                with torch.enable_grad():
                    adversarial = adversarial.detach().requires_grad_(True)
                    objective = F.kl_div(
                        F.log_softmax(attack_logits(model, adversarial), dim=1),
                        reference, reduction="batchmean")
                    gradient = torch.autograd.grad(objective, adversarial)[0]
                if task == "pgd_l2":
                    norms = gradient.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                    candidate = adversarial.detach() + cfg.l2_alpha * gradient / norms[:, None, None, None]
                    adversarial = clean + project_l2(candidate - clean, cfg.l2_eps)
                elif task == "pgd":
                    candidate = adversarial.detach() + cfg.linf_alpha * gradient.sign()
                    adversarial = clean + (candidate - clean).clamp(-cfg.linf_eps, cfg.linf_eps)
                else:
                    candidate = adversarial.detach() + cfg.patch_alpha * gradient.sign()
                    adversarial = torch.where(mask, candidate, clean)
                adversarial = adversarial.clamp(0, 1).detach()
            return adversarial
    finally:
        model.train(was_training)


def backward_gsa(current_loss, reference_loss, model):
    """Project a conflicting current gradient, then add the reference gradient.

    Operate per parameter instead of concatenating several WRN-sized vectors.
    Filtering trainable parameters also fixes frozen-parameter offset corruption
    in the original flatten/write-back implementation.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    reference = torch.autograd.grad(reference_loss, parameters, allow_unused=True)
    current = torch.autograd.grad(current_loss, parameters, allow_unused=True)
    zero = current_loss.detach().new_zeros(())
    dot, norm = zero.clone(), zero.clone()
    for new, old in zip(current, reference):
        if old is not None:
            norm = norm + old.float().square().sum()
            if new is not None:
                dot = dot + (new.float() * old.float()).sum()
    conflict = bool(dot < 0)
    coefficient = dot / (norm + 1e-8) if conflict else zero
    for parameter, new, old in zip(parameters, current, reference):
        if new is None and old is None:
            parameter.grad = None
            continue
        gradient = torch.zeros_like(parameter) if new is None else new.detach().clone()
        if old is not None:
            gradient.add_(old.detach() * (1 - coefficient))
        parameter.grad = gradient
    return {"gradient_conflict": conflict, "gradient_dot": float(dot),
            "reference_gradient_norm_sq": float(norm)}
