from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")


def read_text_fallback(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return raw.decode()


def read_drive_table(path: str | Path) -> pd.DataFrame:
    """Read a normal CSV or a TRN whitespace table with a two-line header."""
    path = Path(path)
    text = read_text_fallback(path)
    first_non_empty = next((line for line in text.splitlines() if line.strip()), "")
    if path.suffix.lower() == ".trn" or first_non_empty.lstrip().startswith("|"):
        return read_trn_table(path)
    return pd.read_csv(path)


def read_trn_table(path: str | Path) -> pd.DataFrame:
    text = read_text_fallback(path)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"TRN file has too few data lines: {path}")

    header_index = 0
    for i, line in enumerate(lines[:10]):
        if line.lstrip().startswith("|"):
            header_index = i
            break

    header = lines[header_index].lstrip("|").strip().split()
    data_lines = lines[header_index + 2 :]
    if not header:
        raise ValueError(f"Could not find TRN header columns: {path}")

    frame = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=header,
        engine="python",
    )
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        non_null = frame[column].notna()
        if converted[non_null].notna().all():
            frame[column] = converted
    return frame


def read_wide_map(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a 2D map CSV.

    Expected format:
        first row: blank or y-axis label, x-axis values
        first col: y-axis values
        cells: map values

    CVT map example: x=speed km/h, y=throttle %, values=rpm.
    BSFC map example: x=engine rpm, y=torque Nm, values=g/kWh.
    """
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        raise ValueError(f"Map is empty: {path}")

    x_axis = pd.to_numeric(pd.Index(frame.columns), errors="coerce").to_numpy(dtype=float)
    y_axis = pd.to_numeric(frame.index, errors="coerce").to_numpy(dtype=float)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if np.isnan(x_axis).any() or np.isnan(y_axis).any():
        raise ValueError(
            f"Map axes must be numeric. Check the first row/column in: {path}"
        )
    if np.isnan(values).all():
        raise ValueError(f"Map contains no numeric values: {path}")

    x_order = np.argsort(x_axis)
    y_order = np.argsort(y_axis)
    return x_axis[x_order], y_axis[y_order], values[np.ix_(y_order, x_order)]


def read_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column curve CSV such as engine RPM -> maximum torque."""
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"Curve needs at least two columns: {path}")
    x_axis = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_axis) & np.isfinite(values)
    x_axis = x_axis[valid]
    values = values[valid]
    if len(x_axis) < 2:
        raise ValueError(f"Curve contains fewer than two numeric points: {path}")
    order = np.argsort(x_axis)
    x_axis = x_axis[order]
    values = values[order]
    if np.any(np.diff(x_axis) <= 0):
        raise ValueError(f"Curve RPM values must be unique: {path}")
    return x_axis, values


def write_wide_map(
    path: str | Path,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    index_label: str,
) -> None:
    frame = pd.DataFrame(values, index=y_axis, columns=x_axis)
    frame.index.name = index_label
    frame.to_csv(path, float_format="%.6g")
