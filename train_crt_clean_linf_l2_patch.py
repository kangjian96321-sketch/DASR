"""CRT on Clean -> PGD-Linf -> PGD-L2 -> Patch; native Croce/ALR unchanged."""
import json

from pace_aligned import crt
from sequence_clean_linf_l2_patch import entry_defaults, set_sequence, validate_resume_sequence


def main(argv=None):
    parser = crt.build_parser()
    entry_defaults(parser)
    args = parser.parse_args(argv)
    cfg = set_sequence(crt.config_from_args('crt', args))
    try:
        options = crt.CRTOptions(
            strategy=args.crt_strategy,
            use_alr=args.crt_use_alr,
            alr_steps=args.crt_alr_steps,
            alr_weight_initial=args.crt_alr_weight_initial,
            alr_weight_finetune=args.crt_alr_weight_finetune,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        crt.print_config(cfg)
        print(json.dumps(options.__dict__, indent=2, ensure_ascii=False))
        return
    validate_resume_sequence(cfg)
    return crt.run(cfg, options)


if __name__ == '__main__':
    main()
