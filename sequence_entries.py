"""Small fixed-order launchers; all optimization stays in native runners."""
import sys
from pace_aligned.selection import OLD_SEQUENCE, NEW_SEQUENCE


def baseline_entry(method, sequence=NEW_SEQUENCE, argv=None):
    from pace_aligned.continual_baselines import main
    args = list(sys.argv[1:] if argv is None else argv)
    reject(args, "--task-sequence")
    tag = "clean_linf_l2_patch" if sequence == NEW_SEQUENCE else "clean_patch_l2_linf"
    return main(method, ["--task-sequence", *sequence, "--output-root",
        "./runs_baselines_" + tag, "--run-name", method + "_" + tag, *args])


def joint_entry(sequence=NEW_SEQUENCE, argv=None):
    from pace_aligned.joint import main
    args = list(sys.argv[1:] if argv is None else argv)
    reject(args, "--joint-attacks")
    reject(args, "--task-sequence")
    tag = "clean_linf_l2_patch" if sequence == NEW_SEQUENCE else "clean_patch_l2_linf"
    return main(["--joint-attacks", *sequence, "--output-root", "./runs_joint_" + tag,
                 "--run-name", "joint_" + tag, *args])


def reject(args, flag):
    if any(a.startswith("--") and flag.startswith(a.split("=", 1)[0]) for a in args):
        raise SystemExit(f"This fixed-order entry does not accept {flag}; use the generic entry")
