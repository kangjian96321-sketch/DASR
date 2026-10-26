"""CPU float32 reservoir used by the supplied DER++/GSA adaptations."""
from __future__ import annotations

import random
import torch


class ReplayMemory:
    def __init__(self, capacity, input_shape, num_classes, store_logits, seed):
        if capacity <= 0:
            raise ValueError("Memory capacity must be positive")
        self.capacity = int(capacity)
        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.store_logits = bool(store_logits)
        self.num_seen = 0
        self.rng = random.Random(seed)
        self.images = torch.empty((capacity, *input_shape), dtype=torch.float32)
        self.labels = torch.empty(capacity, dtype=torch.long)
        self.tasks = torch.empty(capacity, dtype=torch.long)
        self.logits = torch.empty((capacity, num_classes), dtype=torch.float32) if store_logits else None

    def __len__(self):
        return min(self.num_seen, self.capacity)

    def add(self, images, labels, stage, logits=None):
        images, labels = images.detach().cpu().float(), labels.detach().cpu().long()
        if images.shape[1:] != self.input_shape or len(images) != len(labels):
            raise ValueError("Memory image/label shape mismatch")
        if self.store_logits:
            if logits is None or logits.shape != (len(labels), self.num_classes):
                raise ValueError("DER++ requires one historical logit vector per sample")
            logits = logits.detach().cpu().float()
        for index in range(len(labels)):
            # For zero-based exposure n, draw uniformly from 0..n inclusive.
            slot = self.num_seen if self.num_seen < self.capacity else self.rng.randrange(self.num_seen + 1)
            if slot < self.capacity:
                self.images[slot].copy_(images[index])
                self.labels[slot] = labels[index]
                self.tasks[slot] = stage
                if self.logits is not None:
                    self.logits[slot].copy_(logits[index])
            self.num_seen += 1

    def sample(self, count, device):
        if count <= 0:
            raise ValueError("Replay sample count must be positive")
        if not len(self):
            return None
        ids = self.rng.sample(range(len(self)), min(count, len(self)))
        return {
            "images": self.images[ids].to(device), "labels": self.labels[ids].to(device),
            "tasks": self.tasks[ids].clone(),
            "logits": self.logits[ids].to(device) if self.logits is not None else None,
        }

    def state_dict(self):
        # Serialize only initialized slots, never uninitialized capacity padding.
        count = len(self)
        return {
            "capacity": self.capacity, "input_shape": self.input_shape,
            "num_classes": self.num_classes, "store_logits": self.store_logits,
            "num_seen": self.num_seen, "rng": self.rng.getstate(),
            "images": self.images[:count].clone(), "labels": self.labels[:count].clone(),
            "tasks": self.tasks[:count].clone(),
            "logits": self.logits[:count].clone() if self.logits is not None else None,
        }

    def load_state_dict(self, state):
        for name in ("capacity", "input_shape", "num_classes", "store_logits"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Replay memory configuration mismatch: {name}")
        count = min(int(state["num_seen"]), self.capacity)
        if (state["num_seen"] < 0 or state["images"].shape != (count, *self.input_shape)
                or state["labels"].shape != (count,) or state["tasks"].shape != (count,)):
            raise ValueError("Invalid replay memory state")
        if self.store_logits and (state["logits"] is None
                                  or state["logits"].shape != (count, self.num_classes)):
            raise ValueError("Invalid historical logits")
        self.num_seen = int(state["num_seen"])
        for name in ("images", "labels", "tasks"):
            getattr(self, name)[:count].copy_(state[name])
        if self.logits is not None:
            self.logits[:count].copy_(state["logits"])
        self.rng.setstate(state["rng"])

    def summary(self):
        count = len(self)
        allocated = sum(t.numel() * t.element_size()
                        for t in (self.images, self.labels, self.tasks, self.logits) if t is not None)
        valid_bytes = count * (self.images[0].numel() * 4 + 16
                               + (self.num_classes * 4 if self.store_logits else 0))
        tasks, counts = self.tasks[:count].unique(return_counts=True)
        return {
            "capacity": self.capacity, "valid_size": count, "seen_training_exposures": self.num_seen,
            "cpu_allocated_bytes": allocated, "valid_storage_bytes": valid_bytes,
            "counts_by_source_stage": {str(int(task)): int(n) for task, n in zip(tasks, counts)},
            "class_balanced": False, "dtype": "float32",
        }
