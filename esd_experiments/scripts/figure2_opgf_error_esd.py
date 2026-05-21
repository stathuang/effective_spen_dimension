"""CLI for Figure 2: save OP-GF error and ESD data and optionally plot it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from esd_experiments.core.results import load_result, save_result
from esd_experiments.experiments.figure2_opgf_error_esd import (
    DEFAULT_D_VALUES,
    build_config,
    cache_file,
    log_record_iterations,
    parameter_key,
    result_matches_parameter_key,
    run_experiment,
)
from esd_experiments.plots.figure2_opgf_error_esd import plot_from_data


PAPER_OUTPUT = Path("outputs/figures/figure2_opgf_error_esd.pdf")
FAST_OUTPUT = Path("outputs/figures/figure2_opgf_error_esd_fast.pdf")
DATA_OUTPUT = Path("outputs/figure_data/figure2_opgf_error_esd.npz")
FAST_DATA_OUTPUT = Path("outputs/smoke_figures/data/figure2_opgf_error_esd.npz")
DEFAULT_CACHE_DIR = Path("outputs/cache")
FAST_CACHE_DIR = Path("outputs/smoke_figures/cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the averaged OP-GF error and ESD plot used as Figure 2."
    )
    parser.add_argument("--output", type=Path, default=None, help="PDF path to write.")
    parser.add_argument("--data-output", type=Path, default=None, help="NPZ path for saved data.")
    parser.add_argument("--plot-from-data", type=Path, default=None, help="Plot from saved NPZ data.")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--fast", action="store_true", help="Use a quick smoke configuration.")
    parser.add_argument("--num-experiments", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-cache", action="store_true", help="Use parameter-matched cache data.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for experiment trajectories; 0 chooses a conservative default.",
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--d", type=int, default=5_000)
    parser.add_argument("--J", type=int, default=15)
    parser.add_argument("--p", type=float, default=2.5)
    parser.add_argument("--q", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--D-values", type=int, nargs="+", default=list(DEFAULT_D_VALUES))
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--log-start", type=int, default=200)
    parser.add_argument("--log-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (FAST_OUTPUT if args.fast else PAPER_OUTPUT)
    data_output = args.data_output or (FAST_DATA_OUTPUT if args.fast else DATA_OUTPUT)
    cache_dir = FAST_CACHE_DIR if args.fast and args.cache_dir == DEFAULT_CACHE_DIR else args.cache_dir

    if args.plot_from_data is not None:
        if args.no_plot:
            raise ValueError("--plot-from-data cannot be combined with --no-plot")
        plot_from_data(args.plot_from_data, output)
        print(f"Wrote {output} from {args.plot_from_data}")
        return

    config = build_config(
        seed=args.seed,
        fast=args.fast,
        num_experiments=args.num_experiments,
        n=args.n,
        sigma0=args.sigma0,
        d=args.d,
        J=args.J,
        p=args.p,
        q=args.q,
        gamma=args.gamma,
        D_values=tuple(args.D_values),
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        log_start=args.log_start,
        log_step=args.log_step,
    )
    record_iterations = log_record_iterations(
        config.num_iterations, config.log_start, config.log_step
    )
    total_trajectories = len(config.D_values) * config.num_experiments
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    workers = args.workers
    if workers == 0:
        if config.fast:
            workers = 1
        else:
            workers = min(total_trajectories, max(1, min(6, (os.cpu_count() or 2) - 1)))
    key = parameter_key(config, record_iterations)
    cache_path = cache_file(cache_dir, key)

    if args.use_cache and result_matches_parameter_key(data_output, key):
        data_for_plot = data_output
        print(f"Using parameter-matched Figure 2 data from {data_output}")
    elif args.use_cache and result_matches_parameter_key(cache_path, key):
        arrays, metadata = load_result(cache_path)
        save_result(data_output, arrays, metadata)
        data_for_plot = data_output
        print(f"Copied parameter-matched Figure 2 cache to {data_output}")
    else:
        arrays, metadata = run_experiment(
            config,
            record_iterations,
            workers=workers,
            verbose=True,
        )
        save_result(data_output, arrays, metadata)
        save_result(cache_path, arrays, metadata)
        data_for_plot = data_output
        print(f"Saved Figure 2 data to {data_output}")

    if not args.no_plot:
        plot_from_data(data_for_plot, output)
        print(f"Wrote {output} from {data_for_plot}")


if __name__ == "__main__":
    main()
