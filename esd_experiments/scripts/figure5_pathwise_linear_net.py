"""CLI for Figure 5: save pathwise linear-network data and optionally plot it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from esd_experiments.core.results import save_result
from esd_experiments.experiments.figure5_pathwise_linear_net import (
    MissingDependencyError,
    fast_config,
    full_config,
    history_to_arrays,
    metadata_for_config,
    run_experiment,
)
from esd_experiments.plots.figure5_pathwise_linear_net import plot_from_data


DEFAULT_OUTPUT = Path("outputs/figures/figure5_pathwise_linear_net.pdf")
DATA_OUTPUT = Path("outputs/figure_data/figure5_pathwise_linear_net.npz")
FAST_DATA_OUTPUT = Path("outputs/smoke_figures/data/figure5_pathwise_linear_net.npz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 5 pathwise ESD for a deep linear network."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output figure path.")
    parser.add_argument("--data-output", type=Path, default=None, help="NPZ path for saved data.")
    parser.add_argument("--plot-from-data", type=Path, default=None, help="Plot from saved NPZ data.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--fast", action="store_true", help="Use a quick smoke configuration.")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_output = args.data_output or (FAST_DATA_OUTPUT if args.fast else DATA_OUTPUT)

    if args.plot_from_data is not None:
        if args.no_plot:
            print("--plot-from-data cannot be combined with --no-plot", file=sys.stderr)
            return 2
        plot_from_data(args.plot_from_data, args.output)
        print(f"Saved Figure 5 pathwise ESD plot to {args.output} from {args.plot_from_data}")
        return 0

    config = fast_config() if args.fast else full_config()
    try:
        history = run_experiment(config=config, seed=args.seed)
        save_result(
            data_output,
            history_to_arrays(history),
            metadata_for_config(config, args.seed, args.fast),
        )
        print(f"Saved Figure 5 data to {data_output}")
        if not args.no_plot:
            plot_from_data(data_output, args.output)
    except MissingDependencyError as exc:
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 2

    if not args.no_plot:
        print(f"Saved Figure 5 pathwise ESD plot to {args.output} from {data_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
