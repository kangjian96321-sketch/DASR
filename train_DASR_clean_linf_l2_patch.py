"""DASR on Clean -> PGD-Linf -> PGD-L2 -> Patch, using the supplied legacy mechanisms.

All remaining arguments are forwarded unchanged to train_DASR.py.
Separate output defaults avoid mixing the old and new sequence experiments.
"""
from __future__ import annotations

import sys

import train_DASR as training


TASK_SEQUENCE = ('clean', 'pgd', 'pgd_l2', 'patch')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The original parser allows long-option abbreviations; reject those too.
    if any(arg.startswith('--') and '--task-sequence'.startswith(arg.split('=', 1)[0])
           for arg in arguments):
        raise SystemExit('This entry fixes Clean -> PGD-Linf -> PGD-L2 -> Patch. '
                         'For another order, use train_DASR.py --task-sequence.')
    training.main([
        '--output-root', './runs_replay_clean_linf_l2_patch',
        '--run-name', 'clean_linf_l2_patch',
        '--task-sequence', *TASK_SEQUENCE,
        *arguments,
    ])


if __name__ == '__main__':
    main()
