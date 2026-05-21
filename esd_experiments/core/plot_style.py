"""Shared visual style for manuscript figure plotting."""

from __future__ import annotations

import math


FIGSIZE_SINGLE = (5.2, 4.4)
FIGSIZE_DOUBLE = (10.8, 4.4)
FIGSIZE_WIDE = (7.2, 4.6)
PANEL_HEIGHT = 4.1
FONT_SCALE = 1.3
AXIS_LABEL_SIZE = 9.5 * FONT_SCALE
TICK_LABEL_SIZE = 8.5 * FONT_SCALE
LEGEND_FONT_SIZE = 8.5 * FONT_SCALE
TITLE_FONT_SIZE = 10.0 * FONT_SCALE

AXIS_COLOR = "#262626"

SEMANTIC_COLORS = {
    "esd": "#0072B2",
    "risk": "#D55E00",
    "upper_bound": "#009E73",
    "lower_bound": "#0072B2",
    "secondary": "#CC79A7",
    "neutral": "#4D4D4D",
}

CATEGORICAL_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#4D4D4D",
)

SERIES_LINESTYLES = (
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (5, 2, 1, 2)),
)
SERIES_MARKERS = ("o", "s", "^", "D", "v")

LINEWIDTH = 1.65
BOUND_LINEWIDTH = 1.55
ERROR_CAPSIZE = 3.0
ERROR_ELINEWIDTH = 1.05
ERROR_MARKEREDGEWIDTH = 1.0
MARKERSIZE = 5.0

GRID_STYLE = {
    "linestyle": "--",
    "linewidth": 0.5,
    "alpha": 0.55,
}

LEGEND_STYLE = {
    "frameon": False,
    "fontsize": LEGEND_FONT_SIZE,
}


def multipanel_figsize(n_rows: int) -> tuple[float, float]:
    """Return the shared figure size for two-column multi-panel figures."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    return FIGSIZE_DOUBLE[0], PANEL_HEIGHT * n_rows


def color_cycle(index: int) -> str:
    """Return a stable categorical color."""

    return CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]


def series_style(index: int, *, markevery: int | None = None) -> dict[str, object]:
    """Return a color-independent style for a repeated data series."""

    style: dict[str, object] = {
        "color": color_cycle(index),
        "linestyle": SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)],
        "marker": SERIES_MARKERS[index % len(SERIES_MARKERS)],
        "markersize": MARKERSIZE,
        "markerfacecolor": "white",
        "markeredgewidth": ERROR_MARKEREDGEWIDTH,
        "linewidth": LINEWIDTH,
    }
    if markevery is not None:
        style["markevery"] = markevery
    return style


def d_series_style(d_value: int, *, markevery: int | None = None) -> dict[str, object]:
    """Return the fixed visual encoding for a model dimension value."""

    d_to_index = {0: 0, 1: 1, 3: 2}
    return series_style(d_to_index.get(int(d_value), int(d_value)), markevery=markevery)


def semantic_line_style(
    key: str,
    *,
    linestyle: str | tuple[object, object] = "solid",
    marker: str = "o",
    markevery: int | None = None,
    linewidth: float | None = None,
) -> dict[str, object]:
    """Return an accessible line style for a named manuscript quantity."""

    style: dict[str, object] = {
        "color": SEMANTIC_COLORS[key],
        "linestyle": linestyle,
        "marker": marker,
        "markersize": MARKERSIZE,
        "markerfacecolor": "white",
        "markeredgewidth": ERROR_MARKEREDGEWIDTH,
        "linewidth": LINEWIDTH if linewidth is None else linewidth,
    }
    if markevery is not None:
        style["markevery"] = markevery
    return style


def configure_matplotlib(plt) -> None:
    """Apply manuscript plotting rcParams after importing matplotlib."""

    plt.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_STYLE["fontsize"],
            "lines.linewidth": LINEWIDTH,
            "savefig.bbox": "tight",
        }
    )


def apply_axes_style(ax, *, grid: bool = True, which: str = "major") -> None:
    """Apply shared axes styling."""

    ax.tick_params(direction="out", length=3.0, width=0.8, colors=AXIS_COLOR)
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    if grid:
        ax.grid(True, which=which, **GRID_STYLE)


def legend_kwargs(**overrides) -> dict[str, object]:
    """Return shared legend keyword arguments with optional overrides."""

    options = dict(LEGEND_STYLE)
    options.update(overrides)
    return options


def apply_legend(ax, **overrides) -> None:
    """Draw a legend with shared styling."""

    ax.legend(**legend_kwargs(**overrides))


def errorbar_kwargs(**overrides) -> dict[str, object]:
    """Return shared errorbar keyword arguments with optional overrides."""

    options: dict[str, object] = {
        "capsize": ERROR_CAPSIZE,
        "elinewidth": ERROR_ELINEWIDTH,
        "markeredgewidth": ERROR_MARKEREDGEWIDTH,
    }
    options.update(overrides)
    return options


def pad_xaxis_for_terminal_labels(ax, *, factor: float = 1.16) -> None:
    """Add room on the right side for endpoint labels."""

    left, right = ax.get_xlim()
    if factor <= 1.0:
        return
    if ax.get_xscale() == "log":
        if right > 0:
            ax.set_xlim(left, right * factor)
    else:
        ax.set_xlim(left, right + (right - left) * (factor - 1.0))


def add_terminal_label(
    ax,
    x_values,
    y_values,
    label: str,
    *,
    color: str = AXIS_COLOR,
    xytext: tuple[float, float] = (5.0, 0.0),
    va: str = "center",
) -> None:
    """Label the last finite point of a plotted series."""

    point = None
    for x_value, y_value in zip(reversed(list(x_values)), reversed(list(y_values))):
        x_float = float(x_value)
        y_float = float(y_value)
        if math.isfinite(x_float) and math.isfinite(y_float):
            point = (x_float, y_float)
            break
    if point is None:
        return
    ax.annotate(
        label,
        xy=point,
        xytext=xytext,
        textcoords="offset points",
        color=color,
        fontsize=LEGEND_FONT_SIZE,
        ha="left",
        va=va,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
        annotation_clip=False,
        clip_on=False,
    )
