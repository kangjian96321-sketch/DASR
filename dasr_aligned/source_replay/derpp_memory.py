"""DER++ keeps its historical adversarial-logit anchors and adds clean seeds.

Reservoir slots, capacity, stream update timing and RNG remain the host policy.
All fields of a selected slot are written together; MSE never pairs a freshly
regenerated image with a logit target recorded on different adversarial pixels.
"""
from __future__ import annotations
import torch
from ..baseline_memory import ReplayMemory


class DERSourceMemory(ReplayMemory):
    FORMAT = "derpp_source_attack_dual_view_v1_20261106"

    def __init__(self, capacity, input_shape, num_classes, seed, task_sequence):
        super().__init__(capacity, input_shape, num_classes, True, seed)
        self.task_sequence = tuple(task_sequence)
        self.clean_images = torch.empty_like(self.images)

    def add(self, images, labels, stage, logits=None, clean_images=None):
        if not isinstance(stage, int) or not 0 <= stage < len(self.task_sequence):
            raise ValueError("Invalid source stage")
        if clean_images is None:
            if self.task_sequence[stage] != "clean":
                raise ValueError("DER++ source replay requires the matching clean input")
            clean_images = images
        images = images.detach().cpu().float()
        clean_images = clean_images.detach().cpu().float()
        labels = labels.detach().cpu().long()
        if images.shape != (len(labels), *self.input_shape) or clean_images.shape != images.shape:
            raise ValueError("DER++ clean/anchor/label shapes do not match")
        if logits is None or logits.shape != (len(labels), self.num_classes):
            raise ValueError("DER++ requires anchor logits for every sample")
        logits = logits.detach().cpu().float()
        for i in range(len(labels)):
            slot = self.num_seen if self.num_seen < self.capacity else self.rng.randrange(self.num_seen + 1)
            if slot < self.capacity:
                self.images[slot].copy_(images[i])
                self.clean_images[slot].copy_(clean_images[i])
                self.labels[slot] = labels[i]
                self.tasks[slot] = stage
                self.logits[slot].copy_(logits[i])
            self.num_seen += 1

    def sample(self, count, device):
        if count <= 0:
            raise ValueError("Replay sample count must be positive")
        if not len(self):
            return None
        ids = self.rng.sample(range(len(self)), min(count, len(self)))
        return {"images": self.images[ids].to(device),
                "clean_images": self.clean_images[ids].to(device),
                "labels": self.labels[ids].to(device),
                "tasks": self.tasks[ids].clone(), "logits": self.logits[ids].to(device)}

    def state_dict(self):
        return {**super().state_dict(), "source_replay_format": self.FORMAT,
                "task_sequence": self.task_sequence,
                "clean_images": self.clean_images[:len(self)].clone()}

    def load_state_dict(self, state):
        if state.get("source_replay_format") != self.FORMAT:
            raise ValueError("Old fixed-adversarial DER++ checkpoints lack clean seeds; start a new run")
        if tuple(state.get("task_sequence", ())) != self.task_sequence:
            raise ValueError("DER++ replay task sequence mismatch")
        count = min(int(state["num_seen"]), self.capacity)
        if state["clean_images"].shape != (count, *self.input_shape):
            raise ValueError("Invalid clean replay seeds")
        if len(state["tasks"]) and (bool((state["tasks"] < 0).any()) or
                                  bool((state["tasks"] >= len(self.task_sequence)).any())):
            raise ValueError("Invalid source stage in checkpoint")
        super().load_state_dict(state)
        self.clean_images[:count].copy_(state["clean_images"])

    def summary(self):
        result = super().summary()
        result["cpu_allocated_bytes"] += self.clean_images.numel() * self.clean_images.element_size()
        result["valid_storage_bytes"] += len(self) * self.clean_images[0].numel() * self.clean_images.element_size()
        result.update(stored_views="clean_seed_and_fixed_adversarial_logit_anchor",
                      replay_ce="regenerated_source_attack_against_current_model",
                      replay_mse="fixed_adversarial_anchor_vs_its_stored_preupdate_logits",
                      update_policy="host_reservoir_over_training_exposures")
        return result


def initialize_from_shared_base(memory, model, loaders, cfg, device, tracker=None):
    """Same base seeding as the shared-base host, with clean seeds kept in sync."""
    from ..common import attack_logits, isolated_random_seed, generate_attack, training_attack_seed
    was_training = model.training
    model.eval()
    try:
        with isolated_random_seed(cfg.seed + 8257, device):
            for index, batch in enumerate(loaders.train_eval):
                if cfg.max_train_batches is not None and index >= cfg.max_train_batches:
                    break
                x, y = batch[0].to(device), batch[1].to(device)
                view = generate_attack(model, x, y, cfg.task_sequence[0], cfg, training=True,
                    rng_seed=training_attack_seed(cfg, 0, 0, index, stream=31))
                if cfg.task_sequence[0] != "clean" and tracker is not None:
                    tracker.add_adversarial_samples("memory_update", len(y))
                with torch.no_grad():
                    logits = attack_logits(model, view)
                memory.add(view, y, 0, logits=logits, clean_images=x)
    finally:
        model.train(was_training)
