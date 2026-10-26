"""Joint training; order specifies condition forwards, not sequential tasks."""
from sequence_entries import joint_entry
from pace_aligned.selection import OLD_SEQUENCE


def main(argv=None):
    return joint_entry(OLD_SEQUENCE, argv)


if __name__ == "__main__":
    main()
