"""Plot Figure 1 from saved span-profile experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esd_experiments.core.plot_style import (
    apply_axes_style,
    configure_matplotlib,
    multipanel_figsize,
    series_style,
)
from esd_experiments.core.plotting import require_matplotlib, save_figure
from esd_experiments.core.results import load_result


def make_figure(arrays: dict[str, np.ndarray]):
    plt = require_matplotlib()
    configure_matplotlib(plt)
    q_values = arrays["q_values"]
    tau_grid = arrays["tau_grid"]
    iterations = arrays["iterations"]
    time_values = arrays["time_values"]
    profiles = arrays["profiles"]

    n_panels = len(q_values)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=multipanel_figsize(n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    legend_handles = []
    legend_labels = []

    for panel, q in enumerate(q_values):
        ax = axes[panel // n_cols, panel % n_cols]
        for idx, profile in enumerate(profiles[panel]):
            label_time = time_values[idx] if time_values.ndim == 1 else time_values[panel, idx]
            (line,) = ax.loglog(
                tau_grid,
                profile,
                label=rf"$t={label_time:g}$",
                **series_style(idx, markevery=4),
            )
            if panel == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$t={label_time:g}$")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$d^\dagger(\tau)$")
        ax.set_title(rf"$q={q:g}$")
        apply_axes_style(ax, grid=False)

    for panel in range(n_panels, n_rows * n_cols):
        axes[panel // n_cols, panel % n_cols].axis("off")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def plot_from_data(data: str | Path, output: str | Path) -> None:
    arrays, _ = load_result(data)
    fig = make_figure(arrays)
    save_figure(fig, output)
    require_matplotlib().close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 1 from saved data.")
    parser.add_argument("--data", type=Path, required=True, help="Saved Figure 1 NPZ data.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_data(args.data, args.output)
    print(f"Wrote {args.output} from {args.data}")


if __name__ == "__main__":
    main()
