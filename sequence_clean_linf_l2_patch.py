"""Sequence-only setup for the new SSEAT/CRT entries; no training algorithms."""
import json


TASK_SEQUENCE = ('clean', 'pgd', 'pgd_l2', 'patch')


def entry_defaults(parser):
    parser.set_defaults(task_sequence=TASK_SEQUENCE, output_root='./runs_replay_clean_linf_l2_patch',
                        run_name='clean_linf_l2_patch')


def set_sequence(cfg):
    if tuple(cfg.task_sequence) != TASK_SEQUENCE:
        raise ValueError('This entry fixes Clean -> PGD-Linf -> PGD-L2 -> Patch')
    cfg.task_sequence = TASK_SEQUENCE
    return cfg


def validate_resume_sequence(cfg):
    """Reject old-sequence resumes BEFORE native runners can write metadata.

    Existing baseline runners only validate backbone/dataset. Preserve those
    runners unchanged and add this order-specific guard to the new entries.
    This does not add epoch-level resumption or change checkpoint semantics.
    """
    if not cfg.resume:
        return
    path = cfg.run_dir / 'config.json'
    if not path.is_file():
        raise FileNotFoundError(
            f'Cannot verify new-sequence resume: {path} is missing. '
            'Use the original complete run directory or start a new run.')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if tuple(saved.get('task_sequence', ())) != tuple(cfg.task_sequence):
        raise ValueError(
            'Checkpoint task sequence mismatch: expected Clean -> PGD-Linf -> '
            'PGD-L2 -> Patch. Start a new run; do not rename old checkpoints.')
    if saved.get('method') != cfg.method:
        raise ValueError('Checkpoint method mismatch: do not mix SSEAT and CRT run folders.')
