"""Redraw final-test figures from CPU feature caches; no model/GPU/attacks needed."""
import argparse
from pathlib import Path

from dasr_aligned.feature_plots import add_feature_arguments, plot_cached_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plot-seed", type=int, default=None,
                        help="Defaults to the cached evaluation seed")
    add_feature_arguments(parser, include_enable=False)
    args = parser.parse_args(argv)
    plot_cached_features(args.run_dir, samples=args.tsne_samples, perplexity=args.tsne_perplexity,
                         iterations=args.tsne_iterations, metric=args.feature_distance,
                         bins=args.feature_bins, seed=args.plot_seed, dimensions=args.tsne_dimensions,
                         boundary_overlays=args.feature_boundary_overlays)


if __name__ == "__main__":
    main()
