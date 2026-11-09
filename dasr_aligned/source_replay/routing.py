"""Apply the stored source-stage attack to clean seeds against the current model."""
import hashlib
import torch
from ..common import generate_attack


def historical_attack_seed(cfg, current_stage, epoch, batch_index, source_stage):
    payload = f"gsa-history:{cfg.seed}:{current_stage}:{epoch}:{batch_index}:{source_stage}"
    return int.from_bytes(hashlib.blake2s(payload.encode(), digest_size=4).digest(), "little")


def regenerate_history(model, replay, tasks, current_stage, epoch, batch_index, cfg,
                       attack_function=None, allow_current=False):
    attack_function = generate_attack if attack_function is None else attack_function
    images, labels, sources = replay["images"], replay["labels"], replay["tasks"]
    if sources.dtype != torch.long or sources.shape != (len(labels),) or len(images) != len(labels):
        raise ValueError("Invalid source-stage replay batch")
    if not 0 <= current_stage < len(tasks):
        raise ValueError("Current stage is outside the task sequence")
    max_source = current_stage if allow_current else current_stage - 1
    if bool((sources < 0).any()) or bool((sources > max_source).any()):
        raise ValueError("Replay source is not an available stage")
    result = torch.empty_like(images)
    counts, generated = {}, 0
    was_training = model.training
    try:
        for source in sources.unique(sorted=True).tolist():
            indices = (sources == source).nonzero(as_tuple=True)[0].to(images.device)
            task = tasks[source]
            if task == "clean":
                attacked = images[indices].detach()
            else:
                attacked = attack_function(model, images[indices], labels[indices], task, cfg,
                    training=True, rng_seed=historical_attack_seed(cfg, current_stage, epoch, batch_index, source))
                generated += len(indices)
            if attacked.shape != images[indices].shape or not bool(torch.isfinite(attacked).all()):
                raise FloatingPointError("Invalid regenerated source attack")
            result[indices] = attacked.detach()
            counts[str(source)] = {"attack": task, "examples": len(indices),
                                   "generated": len(indices) if task != "clean" else 0}
    finally:
        model.train(was_training)
    return result.detach(), counts, generated
