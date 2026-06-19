from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_wide_map
from .maps import BilinearMap, LinearCurve

RPM_TORQUE_TO_KW = 1.0 / 9549.29658551372


@dataclass(frozen=True)
class ColumnConfig:
    time: str = "Time"
    speed: str = "Actual_Speed"
    throttle: str = "Throttle"
    rpm: str = "Engine_RPM"
    torque: str = "Engine_Torque"
    mode: str | None = None
    mode_value: str | None = None


@dataclass(frozen=True)
class OptimizationOptions:
    min_rpm: float | None = None
    max_rpm: float | None = None
    rpm_step: float = 50.0
    min_speed_kmh: float = 1.0
    min_power_kw: float = 0.5
    map_update_gain: float = 1.0
    max_delta_rpm: float | None = None
    smooth_passes: int = 0
    smooth_weight: float = 0.15
    monotonic_speed: bool = False
    monotonic_throttle: bool = False


@dataclass
class OptimizationResult:
    optimized_map: np.ndarray
    target_map: np.ndarray
    coverage_count: np.ndarray
    drive_evaluation: pd.DataFrame
    summary: dict[str, float | int | str | dict]


def optimize_cvt_map(
    drive: pd.DataFrame,
    cvt_map: BilinearMap,
    bsfc_map: BilinearMap,
    columns: ColumnConfig,
    options: OptimizationOptions,
    max_torque_curve: LinearCurve | None = None,
) -> OptimizationResult:
    data = filter_and_validate_drive(drive, columns)

    speed = data[columns.speed].to_numpy(dtype=float)
    throttle = data[columns.throttle].to_numpy(dtype=float)
    current_rpm = data[columns.rpm].to_numpy(dtype=float)
    current_torque = data[columns.torque].to_numpy(dtype=float)
    dt = estimate_dt_seconds(data, columns.time)

    power_kw = current_rpm * current_torque * RPM_TORQUE_TO_KW
    active = (
        np.isfinite(speed)
        & np.isfinite(throttle)
        & np.isfinite(current_rpm)
        & np.isfinite(current_torque)
        & (speed >= options.min_speed_kmh)
        & (power_kw >= options.min_power_kw)
    )

    min_rpm = options.min_rpm if options.min_rpm is not None else bsfc_map.x_min
    max_rpm = options.max_rpm if options.max_rpm is not None else bsfc_map.x_max
    if min_rpm >= max_rpm:
        raise ValueError("min_rpm must be smaller than max_rpm")
    if options.rpm_step <= 0:
        raise ValueError("rpm_step must be positive")

    rpm_candidates = np.arange(min_rpm, max_rpm + options.rpm_step * 0.5, options.rpm_step)
    target_rpm = np.full(len(data), np.nan)
    target_torque = np.full(len(data), np.nan)
    target_bsfc = np.full(len(data), np.nan)
    theoretical_fuel_gps = np.full(len(data), np.nan)

    active_indices = np.where(active)[0]
    if len(active_indices):
        p = power_kw[active_indices]
        candidate_torque = p[:, None] / (rpm_candidates[None, :] * RPM_TORQUE_TO_KW)
        candidate_rpm = np.broadcast_to(rpm_candidates[None, :], candidate_torque.shape)
        candidate_bsfc = bsfc_map.interpolate(candidate_rpm, candidate_torque, clip=False)
        if max_torque_curve is not None:
            candidate_max_torque = max_torque_curve.interpolate(candidate_rpm, clip=False)
            candidate_feasible = (
                np.isfinite(candidate_max_torque)
                & (candidate_torque <= candidate_max_torque + 1e-9)
            )
            candidate_bsfc = np.where(candidate_feasible, candidate_bsfc, np.nan)
        candidate_fuel_gps = candidate_bsfc * p[:, None] / 3600.0

        all_nan = np.isnan(candidate_fuel_gps).all(axis=1)
        safe_fuel = np.where(np.isnan(candidate_fuel_gps), np.inf, candidate_fuel_gps)
        best_col = np.argmin(safe_fuel, axis=1)
        rows = np.arange(len(active_indices))
        valid_best = ~all_nan
        valid_indices = active_indices[valid_best]
        valid_cols = best_col[valid_best]

        target_rpm[valid_indices] = rpm_candidates[valid_cols]
        target_torque[valid_indices] = candidate_torque[rows[valid_best], valid_cols]
        target_bsfc[valid_indices] = candidate_bsfc[rows[valid_best], valid_cols]
        theoretical_fuel_gps[valid_indices] = candidate_fuel_gps[rows[valid_best], valid_cols]

    current_bsfc = bsfc_map.interpolate(current_rpm, current_torque, clip=False)
    current_fuel_gps = current_bsfc * power_kw / 3600.0
    current_max_torque, current_torque_feasible = torque_feasibility(
        current_rpm,
        current_torque,
        max_torque_curve,
    )

    target_map, coverage_count = aggregate_targets_to_map(
        cvt_map,
        speed,
        throttle,
        target_rpm,
        power_kw,
        dt,
    )
    optimized_map = apply_map_update(cvt_map.values, target_map, coverage_count, options)

    optimized_rpm = cvt_map.__class__(
        cvt_map.x_axis,
        cvt_map.y_axis,
        optimized_map,
        name="optimized CVT map",
    ).interpolate(speed, throttle, clip=True)
    optimized_torque = np.divide(
        power_kw,
        optimized_rpm * RPM_TORQUE_TO_KW,
        out=np.full_like(power_kw, np.nan),
        where=(optimized_rpm > 0),
    )
    optimized_bsfc = bsfc_map.interpolate(optimized_rpm, optimized_torque, clip=False)
    optimized_fuel_gps = optimized_bsfc * power_kw / 3600.0
    optimized_max_torque, optimized_torque_feasible = torque_feasibility(
        optimized_rpm,
        optimized_torque,
        max_torque_curve,
    )

    evaluation = data.copy()
    evaluation["Power_kW"] = power_kw
    evaluation["Current_BSFC_g_per_kWh"] = current_bsfc
    evaluation["Current_Fuel_gps"] = current_fuel_gps
    evaluation["Current_Max_Torque_Nm"] = current_max_torque
    evaluation["Current_Torque_Margin_Nm"] = current_max_torque - current_torque
    evaluation["Current_Torque_Feasible"] = current_torque_feasible
    evaluation["Target_RPM_theoretical"] = target_rpm
    evaluation["Target_Torque_Nm_theoretical"] = target_torque
    evaluation["Target_BSFC_g_per_kWh_theoretical"] = target_bsfc
    evaluation["Target_Fuel_gps_theoretical"] = theoretical_fuel_gps
    evaluation["Optimized_Map_RPM"] = optimized_rpm
    evaluation["Optimized_Map_Torque_Nm"] = optimized_torque
    evaluation["Optimized_Map_BSFC_g_per_kWh"] = optimized_bsfc
    evaluation["Optimized_Map_Fuel_gps"] = optimized_fuel_gps
    evaluation["Optimized_Max_Torque_Nm"] = optimized_max_torque
    evaluation["Optimized_Torque_Margin_Nm"] = optimized_max_torque - optimized_torque
    evaluation["Optimized_Torque_Feasible"] = optimized_torque_feasible

    distance_km = integrate_distance_km(speed, dt)
    current_valid = active & np.isfinite(current_fuel_gps) & current_torque_feasible
    optimized_valid = active & np.isfinite(optimized_fuel_gps) & optimized_torque_feasible
    theoretical_valid = active & np.isfinite(theoretical_fuel_gps)
    current_fuel_g = integrate_validated_fuel(current_fuel_gps, dt, active, current_valid)
    optimized_fuel_g = integrate_validated_fuel(
        optimized_fuel_gps,
        dt,
        active,
        optimized_valid,
    )
    theoretical_fuel_g = integrate_validated_fuel(
        theoretical_fuel_gps,
        dt,
        active,
        theoretical_valid,
    )

    current_infeasible = active & ~current_torque_feasible
    optimized_infeasible = active & ~optimized_torque_feasible
    target_unavailable = active & ~np.isfinite(target_rpm)

    summary = {
        "rows_total": int(len(drive)),
        "rows_after_filter": int(len(data)),
        "active_rows": int(np.count_nonzero(active)),
        "map_cells_total": int(cvt_map.values.size),
        "map_cells_updated": int(np.count_nonzero(coverage_count > 0)),
        "torque_constraint_applied": max_torque_curve is not None,
        "current_torque_infeasible_rows": int(np.count_nonzero(current_infeasible)),
        "optimized_torque_infeasible_rows": int(np.count_nonzero(optimized_infeasible)),
        "target_unavailable_rows": int(np.count_nonzero(target_unavailable)),
        "fuel_result_valid": bool(
            np.isfinite(current_fuel_g)
            and np.isfinite(optimized_fuel_g)
            and np.isfinite(theoretical_fuel_g)
        ),
        "distance_km": distance_km,
        "current_fuel_g": current_fuel_g,
        "optimized_map_fuel_g": optimized_fuel_g,
        "theoretical_pointwise_fuel_g": theoretical_fuel_g,
        "current_g_per_km": divide_or_nan(current_fuel_g, distance_km),
        "optimized_map_g_per_km": divide_or_nan(optimized_fuel_g, distance_km),
        "theoretical_pointwise_g_per_km": divide_or_nan(theoretical_fuel_g, distance_km),
        "optimized_map_improvement_percent": percent_improvement(current_fuel_g, optimized_fuel_g),
        "theoretical_pointwise_improvement_percent": percent_improvement(
            current_fuel_g, theoretical_fuel_g
        ),
        "columns": asdict(columns),
        "options": asdict(options),
    }

    return OptimizationResult(
        optimized_map=optimized_map,
        target_map=target_map,
        coverage_count=coverage_count,
        drive_evaluation=evaluation,
        summary=summary,
    )


def filter_and_validate_drive(drive: pd.DataFrame, columns: ColumnConfig) -> pd.DataFrame:
    required = [columns.speed, columns.throttle, columns.rpm, columns.torque]
    if columns.time in drive.columns:
        required.append(columns.time)
    missing = [column for column in required if column not in drive.columns]
    if missing:
        raise ValueError(
            "Drive data is missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {', '.join(map(str, drive.columns))}"
        )

    data = drive.copy()
    if columns.mode and columns.mode_value is not None:
        if columns.mode not in data.columns:
            raise ValueError(f"Mode column is missing: {columns.mode}")
        data = data[data[columns.mode].astype(str) == str(columns.mode_value)].copy()
    for column in set(required):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


def estimate_dt_seconds(data: pd.DataFrame, time_column: str) -> np.ndarray:
    if time_column in data.columns:
        time = pd.to_numeric(data[time_column], errors="coerce").to_numpy(dtype=float)
        diff = np.diff(time, prepend=np.nan)
        finite_diff = diff[np.isfinite(diff) & (diff > 0)]
        fallback = float(np.median(finite_diff)) if len(finite_diff) else 1.0
        diff[0] = fallback
        diff[~np.isfinite(diff) | (diff <= 0)] = fallback
        return diff
    return np.ones(len(data), dtype=float)


def integrate_distance_km(speed_kmh: np.ndarray, dt_seconds: np.ndarray) -> float:
    distance = np.nansum(speed_kmh * dt_seconds / 3600.0)
    return float(distance)


def integrate_fuel_g(fuel_gps: np.ndarray, dt_seconds: np.ndarray) -> float:
    return float(np.nansum(fuel_gps * dt_seconds))


def integrate_validated_fuel(
    fuel_gps: np.ndarray,
    dt_seconds: np.ndarray,
    required_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> float:
    if np.any(required_rows & ~valid_rows):
        return float("nan")
    return float(np.sum(fuel_gps[required_rows] * dt_seconds[required_rows]))


def torque_feasibility(
    rpm: np.ndarray,
    torque: np.ndarray,
    max_torque_curve: LinearCurve | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_torque_curve is None:
        maximum = np.full_like(np.asarray(torque, dtype=float), np.nan)
        feasible = np.isfinite(rpm) & np.isfinite(torque)
        return maximum, feasible
    maximum = max_torque_curve.interpolate(rpm, clip=False)
    feasible = np.isfinite(maximum) & np.isfinite(torque) & (torque <= maximum + 1e-9)
    return maximum, feasible


def divide_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def percent_improvement(current: float, candidate: float) -> float:
    if not np.isfinite(current) or current == 0 or not np.isfinite(candidate):
        return float("nan")
    return float((current - candidate) / current * 100.0)


def aggregate_targets_to_map(
    cvt_map: BilinearMap,
    speed: np.ndarray,
    throttle: np.ndarray,
    target_rpm: np.ndarray,
    power_kw: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_sum = np.zeros_like(cvt_map.values, dtype=float)
    weight_sum = np.zeros_like(cvt_map.values, dtype=float)
    coverage_count = np.zeros_like(cvt_map.values, dtype=int)

    valid = np.isfinite(speed) & np.isfinite(throttle) & np.isfinite(target_rpm)
    for row in np.where(valid)[0]:
        xi = int(np.argmin(np.abs(cvt_map.x_axis - speed[row])))
        yi = int(np.argmin(np.abs(cvt_map.y_axis - throttle[row])))
        weight = max(float(power_kw[row] * dt[row]), 1e-6)
        weighted_sum[yi, xi] += target_rpm[row] * weight
        weight_sum[yi, xi] += weight
        coverage_count[yi, xi] += 1

    target_map = np.full_like(cvt_map.values, np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=target_map, where=(weight_sum > 0))
    return target_map, coverage_count


def apply_map_update(
    current_map: np.ndarray,
    target_map: np.ndarray,
    coverage_count: np.ndarray,
    options: OptimizationOptions,
) -> np.ndarray:
    updated = current_map.copy()
    covered = coverage_count > 0
    gain = float(np.clip(options.map_update_gain, 0.0, 1.0))
    proposed = current_map + (target_map - current_map) * gain

    if options.max_delta_rpm is not None:
        delta = np.clip(
            proposed - current_map,
            -abs(options.max_delta_rpm),
            abs(options.max_delta_rpm),
        )
        proposed = current_map + delta

    if options.min_rpm is not None or options.max_rpm is not None:
        low = options.min_rpm if options.min_rpm is not None else -np.inf
        high = options.max_rpm if options.max_rpm is not None else np.inf
        proposed = np.clip(proposed, low, high)

    updated[covered] = proposed[covered]

    if options.smooth_passes > 0 and options.smooth_weight > 0:
        updated = smooth_map(updated, options.smooth_passes, options.smooth_weight)
    if options.monotonic_speed or options.monotonic_throttle:
        updated = enforce_monotonic_map(
            updated,
            speed_direction=options.monotonic_speed,
            throttle_direction=options.monotonic_throttle,
        )
        low = options.min_rpm if options.min_rpm is not None else -np.inf
        high = options.max_rpm if options.max_rpm is not None else np.inf
        updated = np.clip(updated, low, high)
    return updated


def smooth_map(values: np.ndarray, passes: int, weight: float) -> np.ndarray:
    result = values.copy()
    weight = float(np.clip(weight, 0.0, 1.0))
    for _ in range(max(0, passes)):
        padded = np.pad(result, 1, mode="edge")
        neighbors = (
            padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
        ) / 4.0
        result = result * (1.0 - weight) + neighbors * weight
    return result


def enforce_monotonic_map(
    values: np.ndarray,
    *,
    speed_direction: bool,
    throttle_direction: bool,
    max_iterations: int = 20,
) -> np.ndarray:
    """Project a CVT map onto nondecreasing speed/throttle directions."""
    result = np.asarray(values, dtype=float).copy()
    for _ in range(max_iterations):
        previous = result.copy()
        if speed_direction:
            for row in range(result.shape[0]):
                result[row, :] = isotonic_increasing(result[row, :])
        if throttle_direction:
            for column in range(result.shape[1]):
                result[:, column] = isotonic_increasing(result[:, column])
        if np.allclose(result, previous, rtol=0.0, atol=1e-8, equal_nan=True):
            break
    return result


def isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing projection using the pool-adjacent-violators algorithm."""
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if not finite.any():
        return result

    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    ends = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    for start, end in zip(starts, ends):
        segment = result[start:end]
        levels: list[float] = []
        weights: list[int] = []
        for value in segment:
            levels.append(float(value))
            weights.append(1)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                merged_weight = weights[-2] + weights[-1]
                merged_level = (
                    levels[-2] * weights[-2] + levels[-1] * weights[-1]
                ) / merged_weight
                levels[-2:] = [merged_level]
                weights[-2:] = [merged_weight]

        cursor = start
        for level, weight in zip(levels, weights):
            result[cursor : cursor + weight] = level
            cursor += weight
    return result


def write_outputs(
    output_dir: str | Path,
    cvt_map: BilinearMap,
    result: OptimizationResult,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_wide_map(
        output / "optimized_cvt_map.csv",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.optimized_map,
        "Throttle",
    )
    write_wide_map(
        output / "target_rpm_map.csv",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.target_map,
        "Throttle",
    )
    write_wide_map(
        output / "delta_rpm_map.csv",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.optimized_map - cvt_map.values,
        "Throttle",
    )
    write_wide_map(
        output / "coverage_count_map.csv",
        cvt_map.x_axis,
        cvt_map.y_axis,
        result.coverage_count,
        "Throttle",
    )
    result.drive_evaluation.to_csv(output / "drive_evaluation.csv", index=False)
