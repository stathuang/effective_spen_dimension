"""Plot Figure 2 from saved OP-GF experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esd_experiments.core.plot_style import (
    FIGSIZE_DOUBLE,
    add_terminal_label,
    apply_axes_style,
    apply_legend,
    configure_matplotlib,
    d_series_style,
    errorbar_kwargs,
    pad_xaxis_for_terminal_labels,
)
from esd_experiments.core.plotting import require_matplotlib, save_figure
from esd_experiments.core.results import load_result


def standard_error(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=float)
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def aggregate_data(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "sq_errors" not in arrays:
        return arrays

    sq_errors = arrays["sq_errors"]
    d_daggers = arrays["d_daggers"]
    return {
        "D_values": arrays["D_values"],
        "time_points": arrays["time_points"],
        "avg_sq_error": np.mean(sq_errors, axis=1),
        "avg_d_dagger": np.mean(d_daggers, axis=1),
        "se_sq_error": np.asarray([standard_error(row) for row in sq_errors]),
        "se_d_dagger": np.asarray([standard_error(row) for row in d_daggers]),
    }


def make_figure(arrays: dict[str, np.ndarray]):
    plt = require_matplotlib()
    configure_matplotlib(plt)
    data = aggregate_data(arrays)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, sharex=True)
    series_for_labels = []

    for idx, D in enumerate(data["D_values"]):
        style = d_series_style(int(D), markevery=7)
        error_style = dict(style)
        error_style.update(errorbar_kwargs())
        label = rf"$D={int(D)}$"
        time_points = data["time_points"][idx]
        axes[0].errorbar(
            time_points,
            data["avg_sq_error"][idx],
            yerr=data["se_sq_error"][idx],
            label=label,
            **error_style,
        )
        axes[1].errorbar(
            time_points,
            data["avg_d_dagger"][idx],
            yerr=data["se_d_dagger"][idx],
            label=label,
            **error_style,
        )
        series_for_labels.append(
            (
                int(D),
                time_points,
                data["avg_sq_error"][idx],
                data["avg_d_dagger"][idx],
                style["color"],
            )
        )

    axes[0].set_title("Oracle PC squared error")
    axes[0].set_ylabel("Squared error")
    axes[1].set_title("Effective span dimension")
    axes[1].set_ylabel(r"$d^\dagger$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Training time")
        apply_axes_style(ax, grid=True, which="both")
        apply_legend(ax, loc="best")
    pad_xaxis_for_terminal_labels(axes[0], factor=1.22)
    label_offsets = {0: (5.0, 7.0), 1: (5.0, 0.0), 3: (5.0, -7.0)}
    for D, time_points, sq_error, d_dagger, color in series_for_labels:
        add_terminal_label(
            axes[0],
            time_points,
            sq_error,
            rf"$D={D}$",
            color=color,
            xytext=label_offsets.get(D, (5.0, 0.0)),
        )
        add_terminal_label(
            axes[1],
            time_points,
            d_dagger,
            rf"$D={D}$",
            color=color,
            xytext=label_offsets.get(D, (5.0, 0.0)),
        )

    fig.tight_layout()
    return fig


def plot_from_data(data: str | Path, output: str | Path) -> None:
    arrays, _ = load_result(data)
    fig = make_figure(arrays)
    save_figure(fig, output)
    require_matplotlib().close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2 from saved data.")
    parser.add_argument("--data", type=Path, required=True, help="Saved Figure 2 NPZ data.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_data(args.data, args.output)
    print(f"Wrote {args.output} from {args.data}")


if __name__ == "__main__":
    main()
