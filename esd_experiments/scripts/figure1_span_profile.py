"""CLI for Figure 1: save span-profile data and optionally plot it."""

from __future__ import annotations

import argparse
from pathlib import Path

from esd_experiments.core.results import save_result
from esd_experiments.experiments.figure1_span_profile import DEFAULT_Q_VALUES, build_config, run_experiment
from esd_experiments.plots.figure1_span_profile import plot_from_data


PAPER_OUTPUT = Path("outputs/figures/figure1_span_profile.pdf")
FAST_OUTPUT = Path("outputs/figures/figure1_span_profile_fast.pdf")
DATA_OUTPUT = Path("outputs/figure_data/figure1_span_profile.npz")
FAST_DATA_OUTPUT = Path("outputs/smoke_figures/data/figure1_span_profile.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the span-profile evolution plot used as Figure 1."
    )
    parser.add_argument("--output", type=Path, default=None, help="PDF path to write.")
    parser.add_argument("--data-output", type=Path, default=None, help="NPZ path for saved data.")
    parser.add_argument("--plot-from-data", type=Path, default=None, help="Plot from saved NPZ data.")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--fast", action="store_true", help="Use a quick smoke configuration.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--d", type=int, default=5_000)
    parser.add_argument("--J", type=int, default=15)
    parser.add_argument("--p", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--D", type=int, default=0)
    parser.add_argument("--q-values", type=float, nargs="+", default=list(DEFAULT_Q_VALUES))
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--snapshot-stride", type=int, default=2_000)
    parser.add_argument("--tau-min", type=float, default=1e-7)
    parser.add_argument("--tau-max", type=float, default=1e-2)
    parser.add_argument("--num-tau", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (FAST_OUTPUT if args.fast else PAPER_OUTPUT)
    data_output = args.data_output or (FAST_DATA_OUTPUT if args.fast else DATA_OUTPUT)

    if args.plot_from_data is not None:
        if args.no_plot:
            raise ValueError("--plot-from-data cannot be combined with --no-plot")
        plot_from_data(args.plot_from_data, output)
        print(f"Wrote {output} from {args.plot_from_data}")
        return

    config = build_config(
        seed=args.seed,
        fast=args.fast,
        n=args.n,
        sigma0=args.sigma0,
        d=args.d,
        J=args.J,
        p=args.p,
        gamma=args.gamma,
        D=args.D,
        q_values=tuple(args.q_values),
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        snapshot_stride=args.snapshot_stride,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        num_tau=args.num_tau,
    )
    arrays, metadata = run_experiment(config)
    save_result(data_output, arrays, metadata)
    print(f"Saved Figure 1 data to {data_output}")

    if not args.no_plot:
        plot_from_data(data_output, output)
        print(f"Wrote {output} from {data_output}")


if __name__ == "__main__":
    main()
