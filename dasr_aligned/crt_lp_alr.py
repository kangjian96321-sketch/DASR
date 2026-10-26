"""Device-portable alignment of official CRT L2Reg (var_reg=False).

Reference: inspire-group/continual_robust_training, commit
5a6932430219d531aab5736de94292b149ff73ae, train/reg_utils.py:L2Reg.
Keep the caller's model mode, CPU float32 random start, per-step clean forward,
renorm_ projection, zero-gradient fallback, and adversarial-first outer forward.
Input-only autograd avoids retaining inner parameter gradients that the official
training loop discards with optimizer.zero_grad() before the outer backward.
This module does not reproduce the redundant double call in adv_train.py.
"""
from __future__ import annotations

import math
import torch

LP_ALR_IMPLEMENTATION = 'official_l2reg_var_false_port_v1_20261015'
LP_TASKS = ('pgd_l2', 'pgd', 'fgsm', 'bim', 'mifgsm')


def lp_parameters(task, cfg, steps):
    if task not in LP_TASKS:
        raise ValueError(f'Not an Lp ALR task: {task}')
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError('ALR steps must be a positive integer')
    eps = float(cfg.l2_eps if task == 'pgd_l2' else cfg.linf_eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('ALR epsilon must be finite and positive')
    alpha = eps if steps == 1 else eps / (steps - 2 if steps > 3 else steps)
    return eps, alpha


def _maximize(model, clean, task, cfg, steps):
    """Caller controls RNG. No mode changes; BN/Dropout follow official L2Reg."""
    from .common import attack_logits
    eps, alpha = lp_parameters(task, cfg, steps)
    clean = clean.detach()
    is_l2 = task == 'pgd_l2'
    # Official code samples torch.FloatTensor on CPU *before* .cuda(). Keep
    # that random stream, then transfer to the actual device, not hard-coded GPU0.
    radius = 1.0 if is_l2 else eps
    noise = torch.empty(clean.shape, device='cpu', dtype=torch.float32).uniform_(-radius, radius)
    noise = noise.to(clean.device)
    if is_l2:
        noise.renorm_(p=2, dim=0, maxnorm=eps)
    with torch.no_grad():
        # Deliberately no initial clamp: first projection occurs after step 1.
        adversarial = clean + noise
        for _ in range(steps):
            adversarial.requires_grad_(True)
            with torch.enable_grad():
                # Ordering matters for dropout RNG and BatchNorm updates.
                distance = torch.norm(attack_logits(model, adversarial)
                                      - attack_logits(model, clean), p=2, dim=-1).mean()
                # .grad in upstream backward is materialized; autograd.grad may
                # instead return a stride-zero view for a constant gradient.
                gradient = torch.autograd.grad(distance, adversarial)[0].detach().clone()
            if is_l2:
                norms = gradient.reshape(len(clean), -1).norm(p=2, dim=1)
                zero = norms == 0
                # Equivalent to upstream div_ then replacement for zero rows,
                # without creating transient NaNs on those rows.
                denominator = norms.masked_fill(zero, 1.0)
                gradient.div_(denominator.view(-1, 1, 1, 1))
                if bool(zero.any()):
                    gradient[zero] = torch.randn_like(gradient[zero])
                candidate = adversarial + alpha * gradient
                delta = candidate - clean
                delta.renorm_(p=2, dim=0, maxnorm=eps)
            else:
                candidate = adversarial + alpha * gradient.sign()
                delta = (candidate - clean).clamp(-eps, eps)
            adversarial = (clean + delta).clamp(0, 1).detach()
    return adversarial


def generate_lp_alr_view(model, images, task, cfg, steps, rng_seed):
    from .common import isolated_random_seed
    with isolated_random_seed(rng_seed, images.device):
        return _maximize(model, images, task, cfg, steps)


def official_lp_alr(model, images, task, cfg, steps, rng_seed):
    from .common import attack_logits, isolated_random_seed
    with isolated_random_seed(rng_seed, images.device):
        clean = images.detach()
        adversarial = _maximize(model, clean, task, cfg, steps)
        # Both paths contribute model gradients, just as official L2Reg.forward.
        # Outer graph is NOT detached and clean is NOT a cached reference.
        return torch.norm(attack_logits(model, adversarial)
                          - attack_logits(model, clean), p=2, dim=-1).mean()
