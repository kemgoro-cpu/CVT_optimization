from pathlib import Path

import numpy as np

from cvt_optimizer.app import (
    bsfc_contour_figure,
    parse_pasted_curve,
    parse_pasted_map,
)
from cvt_optimizer.io import read_drive_table
from cvt_optimizer.optimizer import ColumnConfig, OptimizationOptions, optimize_cvt_map


DEMO = Path(__file__).resolve().parents[1] / "demo_data"


def test_torque_curve_paste_and_bsfc_overlay_stay_inside_plot() -> None:
    drive = read_drive_table(DEMO / "demo_drive.csv")
    cvt_map = parse_pasted_map(
        (DEMO / "current_cvt_map_for_paste.tsv").read_text(encoding="utf-8"),
        "CVT",
    )
    bsfc_map = parse_pasted_map(
        (DEMO / "bsfc_map_for_paste.tsv").read_text(encoding="utf-8"),
        "BSFC",
    )
    torque_curve = parse_pasted_curve(
        (DEMO / "max_torque_curve_for_paste.tsv").read_text(encoding="utf-8"),
        "maximum torque",
    )
    result = optimize_cvt_map(
        drive,
        cvt_map,
        bsfc_map,
        ColumnConfig(),
        OptimizationOptions(),
        torque_curve,
    )

    figure = bsfc_contour_figure(
        bsfc_map,
        torque_curve,
        result.drive_evaluation,
        ColumnConfig(),
    )
    traces = {trace.name: trace for trace in figure.data}
    shade = np.asarray(traces["最大トルク超過領域"].y, dtype=float)

    assert np.nanmin(shade) >= bsfc_map.y_min
    assert np.nanmax(shade) <= bsfc_map.y_max
    assert result.summary["torque_constraint_applied"] is True
