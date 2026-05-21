"""Plot Figure 4 from saved RKHS KPCPE experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def make_figure(arrays: dict[str, object]):
    plt = require_matplotlib()
    configure_matplotlib(plt)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    risk_style = semantic_line_style("risk", linestyle="solid", marker="o", markevery=2)
    lower_style = semantic_line_style("lower_bound", linestyle="dashed", marker="^", markevery=2)
    upper_style = semantic_line_style("upper_bound", linestyle="dotted", marker="D", markevery=2)
    risk_error_style = dict(risk_style)
    risk_error_style.update(errorbar_kwargs())
    ax.errorbar(
        arrays["alpha"],
        arrays["risk"],
        yerr=arrays["risk_se"],
        label=r"Optimal KPCPE risk $\pm$ SE",
        **risk_error_style,
    )
    ax.plot(
        arrays["alpha"],
        arrays["lower_bound"],
        label=r"$(d^\dagger-1)\sigma_0^2/n$",
        **lower_style,
    )
    ax.plot(
        arrays["alpha"],
        arrays["upper_bound"],
        label=r"$2d^\dagger\sigma_{\mathrm{eff}}^2$",
        **upper_style,
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Risk and ESD-based bounds")
    ax.set_ylim(bottom=0)
    apply_axes_style(ax, grid=True)
    apply_legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


def plot_from_data(data: str | Path, output: str | Path) -> None:
    arrays, _ = load_result(data)
    fig = make_figure(arrays)
    save_figure(fig, output)
    require_matplotlib().close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 from saved data.")
    parser.add_argument("--data", type=Path, required=True, help="Saved Figure 4 NPZ data.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_data(args.data, args.output)
    print(f"Wrote {args.output} from {args.data}")


if __name__ == "__main__":
    main()
