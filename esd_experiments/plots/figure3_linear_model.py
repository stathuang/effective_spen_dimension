"""Plot one Figure 3 panel from saved linear-model experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esd_experiments.core.plot_style import (
    FIGSIZE_SINGLE,
    apply_axes_style,
    apply_legend,
    configure_matplotlib,
    errorbar_kwargs,
    semantic_line_style,
)
from esd_experiments.core.plotting import require_matplotlib, save_figure
from esd_experiments.core.results import load_result


def make_figure(arrays: dict[str, np.ndarray], metadata: dict[str, object]):
    plt = require_matplotlib()
    configure_matplotlib(plt)
    n = float(metadata["n"])
    sigma0_sq = float(metadata["sigma0_sq"])
    scale = n / sigma0_sq

    alpha = arrays["alpha"]
    esd = arrays["esd"]
    risk = arrays["risk"]
    risk_se = arrays["risk_se"]
    esd_style = semantic_line_style("esd", linestyle="solid", marker="o", markevery=2)
    risk_style = semantic_line_style("risk", linestyle="dashed", marker="s", markevery=2)
    risk_error_style = dict(risk_style)
    risk_error_style.update(errorbar_kwargs())

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(
        alpha,
        esd,
        label="ESD",
        **esd_style,
    )
    ax.errorbar(
        alpha,
        risk * scale,
        yerr=risk_se * scale,
        label=r"Rescaled risk $\pm$ SE",
        **risk_error_style,
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"ESD and risk $\cdot n/\sigma_0^2$")
    ax.set_ylim(bottom=0)
    apply_axes_style(ax, grid=True)
    apply_legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


def plot_from_data(data: str | Path, output: str | Path) -> None:
    arrays, metadata = load_result(data)
    fig = make_figure(arrays, metadata)
    save_figure(fig, output)
    require_matplotlib().close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one Figure 3 panel from saved data.")
    parser.add_argument("--data", type=Path, required=True, help="Saved Figure 3 case NPZ data.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_data(args.data, args.output)
    print(f"Wrote {args.output} from {args.data}")


if __name__ == "__main__":
    main()
