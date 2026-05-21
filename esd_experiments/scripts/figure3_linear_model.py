"""CLI for Figure 3: save linear-model data and optionally plot it."""

from __future__ import annotations

import argparse
from pathlib import Path

from esd_experiments.core.results import save_result
from esd_experiments.experiments.figure3_linear_model import (
    arrays_for_case,
    build_case_config,
    metadata_for_case,
    run_case,
)
from esd_experiments.plots.figure3_linear_model import plot_from_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 linear-model PCR risk and ESD panels."
    )
    parser.add_argument(
        "--case",
        choices=("1", "2", "both"),
        default="both",
        help="Which Figure 3 case to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where generated PDF files are written.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory where per-case NPZ experiment data are saved.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--fast", action="store_true", help="Use a quick smoke configuration.")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = (1, 2) if args.case == "both" else (int(args.case),)
    data_dir = args.data_dir or (
        Path("outputs/smoke_figures/data")
        if args.fast
        else Path("outputs/figure_data")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        config = build_case_config(case_id, args.fast)
        records, raw_arrays = run_case(config, args.seed)
        data_output = data_dir / f"figure3_case_{case_id}.npz"
        save_result(
            data_output,
            arrays_for_case(records, raw_arrays),
            metadata_for_case(config, args.seed, args.fast),
        )
        print(f"Saved Figure 3 case {case_id} data to {data_output}")
        output = args.output_dir / config.output_name
        if not args.no_plot:
            plot_from_data(data_output, output)
            print(f"Wrote {output} from {data_output}")


if __name__ == "__main__":
    main()
