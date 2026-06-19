import numpy as np
import pandas as pd

from cvt_optimizer.maps import BilinearMap, LinearCurve
from cvt_optimizer.optimizer import (
    ColumnConfig,
    OptimizationOptions,
    apply_map_update,
    enforce_monotonic_map,
    isotonic_increasing,
    optimize_cvt_map,
)


def test_isotonic_increasing_uses_least_squares_block_average() -> None:
    result = isotonic_increasing(np.array([1000.0, 1800.0, 1400.0, 2200.0]))

    np.testing.assert_allclose(result, [1000.0, 1600.0, 1600.0, 2200.0])


def test_enforce_monotonic_map_in_both_directions() -> None:
    values = np.array(
        [
            [1000.0, 1700.0, 1500.0],
            [900.0, 1600.0, 1400.0],
            [1300.0, 1200.0, 2000.0],
        ]
    )

    result = enforce_monotonic_map(
        values,
        speed_direction=True,
        throttle_direction=True,
    )

    assert np.all(np.diff(result, axis=1) >= -1e-8)
    assert np.all(np.diff(result, axis=0) >= -1e-8)


def test_apply_map_update_respects_selected_monotonic_direction() -> None:
    current = np.array([[1000.0, 1200.0, 1400.0], [1300.0, 1500.0, 1700.0]])
    target = np.array([[1000.0, 1800.0, 1300.0], [1600.0, 1400.0, 1900.0]])
    coverage = np.ones_like(current, dtype=int)

    result = apply_map_update(
        current,
        target,
        coverage,
        OptimizationOptions(monotonic_speed=True),
    )

    assert np.all(np.diff(result, axis=1) >= -1e-8)


def test_linear_curve_does_not_extrapolate() -> None:
    curve = LinearCurve(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([100.0, 150.0, 120.0]),
    )

    result = curve.interpolate(np.array([500.0, 1500.0, 3500.0]))

    assert np.isnan(result[0])
    assert result[1] == 125.0
    assert np.isnan(result[2])


def test_max_torque_curve_excludes_infeasible_low_rpm_candidate() -> None:
    drive = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Actual_Speed": [20.0, 20.0],
            "Throttle": [20.0, 20.0],
            "Engine_RPM": [2000.0, 2000.0],
            "Engine_Torque": [80.0, 80.0],
        }
    )
    cvt_map = BilinearMap(
        np.array([0.0, 40.0]),
        np.array([0.0, 40.0]),
        np.full((2, 2), 2000.0),
    )
    bsfc_map = BilinearMap(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([20.0, 80.0, 160.0]),
        np.array(
            [
                [180.0, 260.0, 340.0],
                [140.0, 220.0, 320.0],
                [100.0, 200.0, 300.0],
            ]
        ),
    )
    max_torque_curve = LinearCurve(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = optimize_cvt_map(
        drive,
        cvt_map,
        bsfc_map,
        ColumnConfig(),
        OptimizationOptions(min_rpm=1000.0, max_rpm=3000.0, rpm_step=1000.0),
        max_torque_curve,
    )

    assert np.all(result.drive_evaluation["Target_RPM_theoretical"] >= 2000.0)
    assert result.summary["target_unavailable_rows"] == 0


def test_infeasible_current_point_invalidates_fuel_result() -> None:
    drive = pd.DataFrame(
        {
            "Time": [0.0],
            "Actual_Speed": [20.0],
            "Throttle": [20.0],
            "Engine_RPM": [2000.0],
            "Engine_Torque": [120.0],
        }
    )
    cvt_map = BilinearMap(
        np.array([0.0, 40.0]),
        np.array([0.0, 40.0]),
        np.full((2, 2), 2000.0),
    )
    bsfc_map = BilinearMap(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([20.0, 80.0, 160.0]),
        np.full((3, 3), 240.0),
    )
    max_torque_curve = LinearCurve(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = optimize_cvt_map(
        drive,
        cvt_map,
        bsfc_map,
        ColumnConfig(),
        OptimizationOptions(min_rpm=1000.0, max_rpm=3000.0, rpm_step=1000.0),
        max_torque_curve,
    )

    assert result.summary["current_torque_infeasible_rows"] == 1
    assert np.isnan(result.summary["current_fuel_g"])
    assert result.summary["fuel_result_valid"] is False
