"""SSEAT on Clean -> PGD-Linf -> PGD-L2 -> Patch; native ADR/CRS unchanged."""
from dasr_aligned import sseat
from sequence_clean_linf_l2_patch import entry_defaults, set_sequence, validate_resume_sequence


def main(argv=None):
    parser = sseat.build_parser()
    entry_defaults(parser)
    args = parser.parse_args(argv)
    cfg = set_sequence(sseat.config_from_args('sseat', args))
    if args.dry_run:
        sseat.print_config(cfg)
        return
    validate_resume_sequence(cfg)
    return sseat.run(cfg)


if __name__ == '__main__':
    main()
