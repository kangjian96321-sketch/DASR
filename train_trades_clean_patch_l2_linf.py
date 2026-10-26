"""TRADES: fixed clean_patch_l2_linf order."""
from sequence_entries import baseline_entry
from dasr_aligned.selection import OLD_SEQUENCE


def main(argv=None):
    return baseline_entry("trades", OLD_SEQUENCE, argv)


if __name__ == "__main__":
    main()
