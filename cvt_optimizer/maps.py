from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BilinearMap:
    x_axis: np.ndarray
    y_axis: np.ndarray
    values: np.ndarray
    name: str = "map"

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.y_axis), len(self.x_axis)):
            raise ValueError(
                f"{self.name}: values shape {self.values.shape} does not match "
                f"axes ({len(self.y_axis)}, {len(self.x_axis)})"
            )
        if len(self.x_axis) < 2 or len(self.y_axis) < 2:
            raise ValueError(f"{self.name}: both axes need at least two points")

    @property
    def x_min(self) -> float:
        return float(self.x_axis[0])

    @property
    def x_max(self) -> float:
        return float(self.x_axis[-1])

    @property
    def y_min(self) -> float:
        return float(self.y_axis[0])

    @property
    def y_max(self) -> float:
        return float(self.y_axis[-1])

    def interpolate(
        self,
        x_query: np.ndarray | float,
        y_query: np.ndarray | float,
        *,
        clip: bool = False,
    ) -> np.ndarray:
        xq = np.asarray(x_query, dtype=float)
        yq = np.asarray(y_query, dtype=float)
        xq, yq = np.broadcast_arrays(xq, yq)

        valid = (
            np.isfinite(xq)
            & np.isfinite(yq)
            & (xq >= self.x_min)
            & (xq <= self.x_max)
            & (yq >= self.y_min)
            & (yq <= self.y_max)
        )
        if clip:
            x = np.clip(xq, self.x_min, self.x_max)
            y = np.clip(yq, self.y_min, self.y_max)
            valid = np.isfinite(xq) & np.isfinite(yq)
        else:
            x = xq
            y = yq

        xi1 = np.searchsorted(self.x_axis, x, side="right")
        yi1 = np.searchsorted(self.y_axis, y, side="right")
        xi1 = np.clip(xi1, 1, len(self.x_axis) - 1)
        yi1 = np.clip(yi1, 1, len(self.y_axis) - 1)
        xi0 = xi1 - 1
        yi0 = yi1 - 1

        x0 = self.x_axis[xi0]
        x1 = self.x_axis[xi1]
        y0 = self.y_axis[yi0]
        y1 = self.y_axis[yi1]
        wx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
        wy = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))

        z00 = self.values[yi0, xi0]
        z10 = self.values[yi0, xi1]
        z01 = self.values[yi1, xi0]
        z11 = self.values[yi1, xi1]
        z0 = z00 * (1.0 - wx) + z10 * wx
        z1 = z01 * (1.0 - wx) + z11 * wx
        result = z0 * (1.0 - wy) + z1 * wy
        return np.where(valid, result, np.nan)

