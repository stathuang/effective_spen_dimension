"""Plotting helpers with explicit dependency errors."""

from __future__ import annotations

from pathlib import Path


def require_matplotlib():
    """Import matplotlib.pyplot or raise a clear dependency error."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for figure generation. Install requirements.txt before running figure scripts."
        ) from exc
    return plt


def save_figure(fig, output: str | Path) -> None:
    """Save a matplotlib figure, creating parent directories."""

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")

