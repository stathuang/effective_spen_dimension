"""Plot Figure 5 from saved pathwise linear-network experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

from esd_experiments.core.plot_style import (
    AXIS_COLOR,
    FIGSIZE_WIDE,
    add_terminal_label,
    apply_axes_style,
    configure_matplotlib,
    legend_kwargs,
    pad_xaxis_for_terminal_labels,
    semantic_line_style,
)
from esd_experiments.core.plotting import require_matplotlib, save_figure
from esd_experiments.core.results import load_result


def make_figure(arrays: dict[str, object]):
    plt = require_matplotlib()
    configure_matplotlib(plt)
    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)

    esd_style = semantic_line_style("esd", linestyle="solid", marker="o", markevery=3)
    risk_style = semantic_line_style("risk", linestyle="dashed", marker="s", markevery=3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ESD")
    ax1.plot(
        arrays["epoch"],
        arrays["esd"],
        label="ESD",
        **esd_style,
    )
    apply_axes_style(ax1, grid=True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Risk")
    ax2.plot(
        arrays["epoch"],
        arrays["risk"],
        label="Risk",
        **risk_style,
    )
    apply_axes_style(ax2, grid=False)
    for ax in (ax1, ax2):
        ax.tick_params(axis="both", colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
        ax.yaxis.label.set_color(AXIS_COLOR)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
    pad_xaxis_for_terminal_labels(ax1, factor=1.12)
    add_terminal_label(
        ax1,
        arrays["epoch"],
        arrays["esd"],
        "ESD",
        color=esd_style["color"],
        xytext=(6.0, 8.0),
    )
    add_terminal_label(
        ax2,
        arrays["epoch"],
        arrays["risk"],
        "Risk",
        color=risk_style["color"],
        xytext=(6.0, -9.0),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, **legend_kwargs(loc="upper right"))
    fig.tight_layout()
    return fig


def plot_from_data(data: str | Path, output: str | Path) -> None:
    arrays, _ = load_result(data)
    fig = make_figure(arrays)
    save_figure(fig, output)
    require_matplotlib().close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 from saved data.")
    parser.add_argument("--data", type=Path, required=True, help="Saved Figure 5 NPZ data.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_data(args.data, args.output)
    print(f"Wrote {args.output} from {args.data}")


if __name__ == "__main__":
    main()
