"""Shared checkpoint-selection definition, independent of training algorithms."""
import math

SELECTION_POLICY = "post_warmup_mean_all_seen_including_zero_v1_20261025"
OLD_SEQUENCE = ("clean", "patch", "pgd_l2", "pgd")
NEW_SEQUENCE = ("clean", "pgd", "pgd_l2", "patch")


def validate_sequence(sequence):
    sequence = tuple(sequence)
    if sequence not in (OLD_SEQUENCE, NEW_SEQUENCE):
        raise ValueError("Choose clean patch pgd_l2 pgd OR clean pgd pgd_l2 patch")
    return sequence


def warmup_epochs(cfg, task):
    epochs = cfg.task_epochs[task]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Training phases must have positive integer epoch counts")
    if not math.isfinite(cfg.warmup_ratio) or not 0 <= cfg.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    # A one-epoch diagnostic phase has no warmup; a real phase keeps the
    # original scheduler's floor(ratio * epochs) + 1 convention.
    if epochs == 1 or cfg.warmup_ratio == 0:
        return 0
    return max(1, min(epochs - 1, int(cfg.warmup_ratio * epochs) + 1))


def best_eligible(cfg, task, epoch):
    """epoch is zero-based: actual epoch number must be strictly greater than W."""
    return epoch >= warmup_epochs(cfg, task)


def selection_mean(accuracies, expected):
    expected = tuple(expected)
    if not expected or len(expected) != len(set(expected)) or set(accuracies) != set(expected):
        raise ValueError("Selection must contain exactly the evaluated conditions")
    values = [float(accuracies[task]) for task in expected]
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Selection accuracies must be finite percentages in [0,100]")
    return sum(values) / len(values)
