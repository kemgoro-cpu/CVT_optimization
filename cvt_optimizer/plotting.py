from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .maps import BilinearMap
from .optimizer import OptimizationResult


def plot_outputs(output_dir: str | Path, cvt_map: BilinearMap, result: OptimizationResult) -> None:
    output = Path(output_dir)
    plot_heatmap(
        output / "cvt_current.png",
        cvt_map.x_axis,
        cvt_map.y_axis,
        cvt_map.values,
        "Current CVT RPM map",
        "Speed km/h",
        "Throttle %",
        "rpm",
    )
    plot_heatmap(
        output / "cvt_optimized.png",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.optimized_map,
        "Optimized CVT RPM map",
        "Speed km/h",
        "Throttle %",
        "rpm",
    )
    plot_heatmap(
        output / "cvt_delta.png",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.optimized_map - cvt_map.values,
        "Optimized - current RPM",
        "Speed km/h",
        "Throttle %",
        "rpm",
    )


def plot_heatmap(
    path: str | Path,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    mesh = ax.pcolormesh(x_axis, y_axis, values, shading="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)

