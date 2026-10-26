"""Lightweight backbone naming shared by trainers and experiment launchers."""
BACKBONE_CHOICES = ("wrn", "resnet18")


def canonical_backbone(value):
    aliases = {
        "wrn": "wrn", "wrn34_10": "wrn", "wrn-34-10": "wrn",
        "resnet18": "resnet18", "resnet-18": "resnet18",
    }
    name = str(value).lower()
    if name not in aliases:
        raise ValueError(f"Unknown backbone {value!r}; choose wrn or resnet18")
    return aliases[name]


def backbone_suffix(value):
    # Never rename legacy WRN run directories.
    return "" if canonical_backbone(value) == "wrn" else "_resnet18"


def add_backbone_argument(parser):
    parser.add_argument(
        "--backbone", type=canonical_backbone, choices=BACKBONE_CHOICES, default="wrn",
        help="Network: wrn (default WRN-34-10) or resnet18, both trained from scratch.",
    )


def backbone_spec(config):
    """Effective architecture; WRN-only options do not affect ResNet-18."""
    values = config if isinstance(config, dict) else vars(config)
    name = canonical_backbone(values.get("backbone", "wrn"))
    dataset = values.get("dataset", "cifar100")
    if name == "wrn":
        depth, width = values.get("depth", 34), values.get("widen_factor", 10)
        return {"name": f"wrn-{depth}-{width}", "backbone": name,
                "feature_dim": 64 * width, "stem": "conv3x3_stride1",
                "dropout": values.get("dropout", 0.3), "pretrained": False}
    return {"name": "resnet-18", "backbone": name, "feature_dim": 512,
            "stem": ("conv3x3_stride1_no_maxpool" if dataset in ("cifar10", "cifar100")
                     else "conv7x7_stride2_maxpool3x3_stride2"),
            "dropout": 0.0, "pretrained": False}
