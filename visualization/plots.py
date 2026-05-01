"""Small plotting helpers shared across dashboards."""

from __future__ import annotations

from matplotlib.axes import Axes


def apply_axis_style(ax: Axes, title: str) -> None:
    """Apply consistent title/grid style to a matplotlib axis."""
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
