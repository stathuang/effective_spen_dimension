"""CLI for Figure 4: save RKHS KPCPE data and optionally plot it."""

from __future__ import annotations

import argparse
from pathlib import Path

from esd_experiments.core.results import save_result
from esd_experiments.experiments.figure4_rkhs_kpcpe import (
    arrays_for_records,
    build_config,
    metadata_for_config,
    simulate,
)
from esd_experiments.plots.figure4_rkhs_kpcpe import plot_from_data


DATA_OUTPUT = Path("outputs/figure_data/figure4_rkhs_kpcpe.npz")
FAST_DATA_OUTPUT = Path("outputs/smoke_figures/data/figure4_rkhs_kpcpe.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 RKHS KPCPE risk and ESD-bound plot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/figure4_rkhs_kpcpe.pdf"),
        help="Path where the generated PDF file is written.",
    )
    parser.add_argument("--data-output", type=Path, default=None, help="NPZ path for saved data.")
    parser.add_argument("--plot-from-data", type=Path, default=None, help="Plot from saved NPZ data.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--fast", action="store_true", help="Use a quick smoke configuration.")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_output = args.data_output or (FAST_DATA_OUTPUT if args.fast else DATA_OUTPUT)

    if args.plot_from_data is not None:
        if args.no_plot:
            raise ValueError("--plot-from-data cannot be combined with --no-plot")
        plot_from_data(args.plot_from_data, args.output)
        print(f"Wrote {args.output} from {args.plot_from_data}")
        return

    config = build_config(args.fast)
    records, f_sup_norm_sq, raw_arrays = simulate(config, args.seed)
    save_result(
        data_output,
        arrays_for_records(records, raw_arrays),
        metadata_for_config(config, args.seed, args.fast, f_sup_norm_sq),
    )
    print(f"Saved Figure 4 data to {data_output}")
    print(f"Estimated ||f*||_infty^2 = {f_sup_norm_sq:.6g}")
    if not args.no_plot:
        plot_from_data(data_output, args.output)
        print(f"Wrote {args.output} from {data_output}")


if __name__ == "__main__":
    main()
