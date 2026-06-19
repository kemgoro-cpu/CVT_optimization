from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import read_curve, read_drive_table, read_wide_map
from .maps import BilinearMap, LinearCurve
from .optimizer import (
    ColumnConfig,
    OptimizationOptions,
    optimize_cvt_map,
    write_outputs,
)
from .plotting import plot_outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return inspect_file(args)
    if args.command == "optimize":
        return optimize(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvt-optimize",
        description="Optimize CVT speed-throttle RPM maps using drive data and a BSFC map.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect = subparsers.add_parser("inspect", help="Inspect a CSV/TRN drive data file.")
    inspect.add_argument("path")
    inspect.set_defaults(command="inspect")

    optimize_parser = subparsers.add_parser("optimize", help="Run CVT map optimization.")
    optimize_parser.add_argument("--drive-data", required=True)
    optimize_parser.add_argument("--cvt-map", required=True)
    optimize_parser.add_argument("--bsfc-map", required=True)
    optimize_parser.add_argument("--max-torque-curve", required=True)
    optimize_parser.add_argument("--output-dir", default="out")
    optimize_parser.add_argument("--time-col", default="Time")
    optimize_parser.add_argument("--speed-col", default="Actual_Speed")
    optimize_parser.add_argument("--throttle-col", default="Throttle")
    optimize_parser.add_argument("--rpm-col", default="Engine_RPM")
    optimize_parser.add_argument("--torque-col", default="Engine_Torque")
    optimize_parser.add_argument("--mode-col")
    optimize_parser.add_argument("--mode-value")
    optimize_parser.add_argument("--min-rpm", type=float)
    optimize_parser.add_argument("--max-rpm", type=float)
    optimize_parser.add_argument("--rpm-step", type=float, default=50.0)
    optimize_parser.add_argument("--min-speed-kmh", type=float, default=1.0)
    optimize_parser.add_argument("--min-power-kw", type=float, default=0.5)
    optimize_parser.add_argument("--map-update-gain", type=float, default=1.0)
    optimize_parser.add_argument("--max-delta-rpm", type=float)
    optimize_parser.add_argument("--smooth-passes", type=int, default=0)
    optimize_parser.add_argument("--smooth-weight", type=float, default=0.15)
    optimize_parser.add_argument("--monotonic-speed", action="store_true")
    optimize_parser.add_argument("--monotonic-throttle", action="store_true")
    optimize_parser.add_argument("--no-plots", action="store_true")
    optimize_parser.set_defaults(command="optimize")
    return parser


def inspect_file(args: argparse.Namespace) -> int:
    frame = read_drive_table(args.path)
    print(f"Rows: {len(frame)}")
    print("Columns:")
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            print(
                f"  {column}: numeric, min={series.min():.6g}, max={series.max():.6g}"
            )
        else:
            print(f"  {column}: text")
    return 0


def optimize(args: argparse.Namespace) -> int:
    drive = read_drive_table(args.drive_data)
    cvt_x, cvt_y, cvt_values = read_wide_map(args.cvt_map)
    bsfc_x, bsfc_y, bsfc_values = read_wide_map(args.bsfc_map)
    torque_rpm, torque_values = read_curve(args.max_torque_curve)
    cvt_map = BilinearMap(cvt_x, cvt_y, cvt_values, name="CVT map")
    bsfc_map = BilinearMap(bsfc_x, bsfc_y, bsfc_values, name="BSFC map")
    max_torque_curve = LinearCurve(
        torque_rpm,
        torque_values,
        name="maximum torque curve",
    )

    columns = ColumnConfig(
        time=args.time_col,
        speed=args.speed_col,
        throttle=args.throttle_col,
        rpm=args.rpm_col,
        torque=args.torque_col,
        mode=args.mode_col,
        mode_value=args.mode_value,
    )
    options = OptimizationOptions(
        min_rpm=args.min_rpm,
        max_rpm=args.max_rpm,
        rpm_step=args.rpm_step,
        min_speed_kmh=args.min_speed_kmh,
        min_power_kw=args.min_power_kw,
        map_update_gain=args.map_update_gain,
        max_delta_rpm=args.max_delta_rpm,
        smooth_passes=args.smooth_passes,
        smooth_weight=args.smooth_weight,
        monotonic_speed=args.monotonic_speed,
        monotonic_throttle=args.monotonic_throttle,
    )
    result = optimize_cvt_map(
        drive,
        cvt_map,
        bsfc_map,
        columns,
        options,
        max_torque_curve,
    )
    write_outputs(args.output_dir, cvt_map, result)
    if not args.no_plots:
        plot_outputs(args.output_dir, cvt_map, result)

    summary_path = Path(args.output_dir) / "optimization_summary.json"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
