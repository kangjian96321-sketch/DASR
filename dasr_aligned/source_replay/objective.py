"""Detached accumulated PN-gradient statistics and GSA-weighted CE.

Reference: Guo et al., CVPR 2023, and gydpku/GSA test_cifar100.py at
3f523ba838af24af3e98b0f9ccda11d89c38b069. Independently written implementation.
For class c, P_c=sum_{y=c}(1-p_c), N_c=sum_{y!=c}p_c,
r_c=P_c/N_c and w_c=2*sigmoid(r_c-1). Use a sample mean, not a
weight-normalized mean. All weights/statistics are detached from autograd.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class GradientStatistics(torch.nn.Module):
    def __init__(self, num_classes: int, warmup_steps: int = 5, epsilon: float = 1e-12):
        super().__init__()
        if num_classes < 2 or warmup_steps < 0:
            raise ValueError("num_classes >= 2 and warmup_steps >= 0 are required")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.num_classes = int(num_classes)
        self.warmup_steps = int(warmup_steps)
        self.epsilon = float(epsilon)
        for name in ("category_count", "positive_probability", "negative_probability"):
            self.register_buffer(name, torch.zeros(num_classes, dtype=torch.float64))
        self.register_buffer("stage_index", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("stage_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_updates", torch.tensor(0, dtype=torch.long))

    def _validate(self, logits, labels):
        if logits.ndim != 2 or logits.shape[1] != self.num_classes or not len(logits):
            raise ValueError("Expected a nonempty [batch, num_classes] logit matrix")
        if labels.shape != (len(logits),) or labels.dtype != torch.long:
            raise ValueError("Expected a one-dimensional int64 label tensor")
        if logits.device != labels.device or logits.device != self.category_count.device:
            raise ValueError("Statistics, logits and labels must share a device")
        if bool((labels < 0).any()) or bool((labels >= self.num_classes).any()):
            raise ValueError("Labels outside the fixed class vocabulary")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite logits in GSA statistics")

    @torch.no_grad()
    def start_stage(self, stage: int):
        if stage < 0:
            raise ValueError("stage must be nonnegative")
        if int(self.stage_index) != stage:
            if stage < int(self.stage_index):
                raise ValueError("GSA stage cannot move backwards")
            self.stage_index.fill_(stage)
            self.stage_updates.zero_()
        # Accumulated statistics persist across tasks, as in upstream GSA.

    @torch.no_grad()
    def update(self, logits, labels, stage: int):
        self._validate(logits, labels)
        self.start_stage(stage)
        probabilities = torch.softmax(logits.detach().to(torch.float64), dim=1)
        true_prob = probabilities.gather(1, labels[:, None]).squeeze(1)
        self.category_count.add_(torch.bincount(labels, minlength=self.num_classes))
        self.positive_probability.scatter_add_(0, labels, true_prob)
        probabilities.scatter_(1, labels[:, None], 0.)
        self.negative_probability.add_(probabilities.sum(0))
        self.stage_updates.add_(1)
        self.total_updates.add_(1)

    @torch.no_grad()
    def ratios(self):
        positive_gradient = (self.category_count - self.positive_probability).clamp_min(0)
        ratio = positive_gradient / self.negative_probability.clamp_min(self.epsilon)
        # A wholly unobserved class has neither positive nor negative evidence.
        absent = (positive_gradient <= self.epsilon) & (self.negative_probability <= self.epsilon)
        return torch.where(absent, torch.ones_like(ratio), ratio)

    @torch.no_grad()
    def weights(self):
        if int(self.stage_updates) <= self.warmup_steps:
            return torch.ones_like(self.category_count)
        return 2. * torch.sigmoid(self.ratios() - 1.)

    def loss(self, logits, labels, stage: int, update: bool = True):
        self._validate(logits, labels)
        if update:
            self.update(logits, labels, stage)
        # Do not normalize by sum(weights); upstream divides by sample count.
        weights = self.weights().to(dtype=logits.dtype)
        return (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).mean()

    @torch.no_grad()
    def summary(self):
        weights = self.weights()
        return {"stage": int(self.stage_index), "stage_updates": int(self.stage_updates),
                "total_updates": int(self.total_updates),
                "category_count": self.category_count.cpu().tolist(),
                "accumulated_pn_ratio": self.ratios().cpu().tolist(),
                "class_weights": weights.cpu().tolist(),
                "weight_min": float(weights.min()), "weight_max": float(weights.max())}
