"""GSA: fixed clean_linf_l2_patch order."""
from sequence_entries import baseline_entry
from dasr_aligned.selection import NEW_SEQUENCE


def main(argv=None):
    return baseline_entry("gsa", NEW_SEQUENCE, argv)


if __name__ == "__main__":
    main()
