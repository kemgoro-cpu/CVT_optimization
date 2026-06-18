from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cvt_optimizer.io import read_drive_table
from cvt_optimizer.maps import BilinearMap
from cvt_optimizer.optimizer import ColumnConfig, OptimizationOptions, optimize_cvt_map


def main() -> None:
    st.set_page_config(
        page_title="CVT Optimizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CVT変速線図最適化")

    with st.sidebar:
        uploaded_drive = st.file_uploader(
            "走行データ CSV / TRN",
            type=["csv", "trn", "txt"],
        )
        st.divider()
        time_col = st.text_input("時間列", "Time")
        speed_col = st.text_input("車速列", "Actual_Speed")
        throttle_col = st.text_input("アクセル開度列", "Throttle")
        rpm_col = st.text_input("エンジン回転数列", "Engine_RPM")
        torque_col = st.text_input("エンジントルク列", "Engine_Torque")
        mode_col = st.text_input("モード列", "")
        mode_value = st.text_input("燃費モード値", "")
        st.divider()
        min_rpm = optional_number("最小RPM", "")
        max_rpm = optional_number("最大RPM", "")
        rpm_step = st.number_input("RPM刻み", min_value=1.0, value=50.0, step=10.0)
        min_speed = st.number_input("最小車速 km/h", min_value=0.0, value=1.0, step=1.0)
        min_power = st.number_input("最小出力 kW", min_value=0.0, value=0.5, step=0.5)
        update_gain = st.slider("更新ゲイン", 0.0, 1.0, 1.0, 0.05)
        max_delta = optional_number("最大変更RPM", "")
        smooth_passes = st.number_input("平滑化回数", min_value=0, value=0, step=1)
        smooth_weight = st.slider("平滑化強さ", 0.0, 1.0, 0.15, 0.05)

    left, right = st.columns(2)
    with left:
        cvt_text = st.text_area(
            "現在のCVT変速線図",
            height=260,
            placeholder="Throttle\t0\t20\t40\t60\n0\t800\t900\t1000\t1100\n20\t1200\t1500\t1800\t2100",
        )
    with right:
        bsfc_text = st.text_area(
            "燃費率マップ",
            height=260,
            placeholder="Torque\t1000\t1500\t2000\t2500\n20\t360\t330\t310\t320\n40\t310\t270\t245\t250",
        )

    run = st.button("最適化を実行", type="primary", use_container_width=True)

    if not run:
        return

    if uploaded_drive is None:
        st.error("走行データを選択してください。")
        return
    if not cvt_text.strip() or not bsfc_text.strip():
        st.error("CVT変速線図と燃費率マップを貼り付けてください。")
        return

    try:
        drive = read_uploaded_drive(uploaded_drive)
        cvt_map = parse_pasted_map(cvt_text, "CVT変速線図")
        bsfc_map = parse_pasted_map(bsfc_text, "燃費率マップ")
        columns = ColumnConfig(
            time=time_col,
            speed=speed_col,
            throttle=throttle_col,
            rpm=rpm_col,
            torque=torque_col,
            mode=mode_col.strip() or None,
            mode_value=mode_value.strip() or None,
        )
        options = OptimizationOptions(
            min_rpm=min_rpm,
            max_rpm=max_rpm,
            rpm_step=float(rpm_step),
            min_speed_kmh=float(min_speed),
            min_power_kw=float(min_power),
            map_update_gain=float(update_gain),
            max_delta_rpm=max_delta,
            smooth_passes=int(smooth_passes),
            smooth_weight=float(smooth_weight),
        )
        result = optimize_cvt_map(drive, cvt_map, bsfc_map, columns, options)
    except Exception as exc:  # noqa: BLE001 - Streamlit should surface data issues.
        st.error(str(exc))
        return

    show_summary(result.summary)

    st.plotly_chart(
        bsfc_contour_figure(bsfc_map, result.drive_evaluation, columns),
        use_container_width=True,
    )

    st.plotly_chart(
        cvt_map_comparison_figure(cvt_map, result.optimized_map),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            map_heatmap_figure(
                cvt_map.x_axis,
                cvt_map.y_axis,
                result.optimized_map - cvt_map.values,
                "CVT変更量",
                "rpm",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            map_heatmap_figure(
                cvt_map.x_axis,
                cvt_map.y_axis,
                result.coverage_count,
                "走行点カバレッジ",
                "points",
            ),
            use_container_width=True,
        )

    st.subheader("出力")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "最適化後CVT CSV",
            wide_map_csv(cvt_map.x_axis, cvt_map.y_axis, result.optimized_map, "Throttle"),
            "optimized_cvt_map.csv",
            "text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "CVT差分 CSV",
            wide_map_csv(
                cvt_map.x_axis,
                cvt_map.y_axis,
                result.optimized_map - cvt_map.values,
                "Throttle",
            ),
            "delta_rpm_map.csv",
            "text/csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "走行点評価 CSV",
            result.drive_evaluation.to_csv(index=False).encode("utf-8-sig"),
            "drive_evaluation.csv",
            "text/csv",
            use_container_width=True,
        )


def optional_number(label: str, default: str) -> float | None:
    value = st.text_input(label, default)
    if not value.strip():
        return None
    return float(value)


def parse_pasted_map(text: str, name: str) -> BilinearMap:
    frame = parse_pasted_table(text)
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"{name}の行列サイズが小さすぎます。")

    x_axis = pd.to_numeric(pd.Index(frame.columns[1:]), errors="coerce").to_numpy(float)
    y_axis = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    if np.isnan(x_axis).any() or np.isnan(y_axis).any():
        raise ValueError(f"{name}の軸に数値ではない値があります。")
    if np.isnan(values).any():
        raise ValueError(f"{name}の値に空欄または数値ではない値があります。")

    x_order = np.argsort(x_axis)
    y_order = np.argsort(y_axis)
    return BilinearMap(
        x_axis[x_order],
        y_axis[y_order],
        values[np.ix_(y_order, x_order)],
        name=name,
    )


def parse_pasted_table(text: str) -> pd.DataFrame:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    delimiter = "\t" if "\t" in normalized.splitlines()[0] else ","
    if delimiter == "," and all("," not in line for line in normalized.splitlines()[:3]):
        delimiter = r"\s+"
    return pd.read_csv(StringIO(normalized), sep=delimiter, engine="python")


def read_uploaded_drive(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name
    try:
        return read_drive_table(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def show_summary(summary: dict) -> None:
    current = summary.get("current_g_per_km", float("nan"))
    optimized = summary.get("optimized_map_g_per_km", float("nan"))
    improvement = summary.get("optimized_map_improvement_percent", float("nan"))
    distance = summary.get("distance_km", float("nan"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現状 g/km", format_metric(current))
    m2.metric("最適化後 g/km", format_metric(optimized))
    m3.metric("改善率", format_metric(improvement, suffix="%"))
    m4.metric("走行距離 km", format_metric(distance))


def format_metric(value: object, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.3f}{suffix}"


def bsfc_contour_figure(
    bsfc_map: BilinearMap,
    evaluation: pd.DataFrame,
    columns: ColumnConfig,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=bsfc_map.x_axis,
            y=bsfc_map.y_axis,
            z=bsfc_map.values,
            contours=dict(
                showlabels=True,
                coloring="heatmap",
                labelfont=dict(size=15, color="#25332f"),
            ),
            colorscale=[
                [0.0, "#f4f6f5"],
                [0.25, "#dce4e1"],
                [0.5, "#b7c9c3"],
                [0.75, "#789c91"],
                [1.0, "#3f665d"],
            ],
            line=dict(color="rgba(38, 54, 49, 0.42)", width=1),
            colorbar=dict(title="g/kWh", thickness=16),
            hovertemplate="%{x:.0f} rpm<br>%{y:.1f} Nm<br>%{z:.1f} g/kWh<extra></extra>",
            name="BSFC",
        )
    )

    for power_kw in choose_power_lines(evaluation):
        rpm = np.linspace(bsfc_map.x_min, bsfc_map.x_max, 180)
        torque = power_kw * 9549.29658551372 / rpm
        mask = (torque >= bsfc_map.y_min) & (torque <= bsfc_map.y_max)
        if np.count_nonzero(mask) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=rpm[mask],
                y=torque[mask],
                mode="lines",
                line=dict(width=1, color="rgba(35, 43, 41, 0.22)", dash="dot"),
                name=f"{power_kw:.0f} kW",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    sampled = downsample(evaluation, 5000)
    fig.add_trace(
        go.Scattergl(
            x=sampled[columns.rpm],
            y=sampled[columns.torque],
            mode="markers",
            marker=dict(
                size=7,
                color="#b73d52",
                opacity=0.78,
                line=dict(color="rgba(255,255,255,0.9)", width=0.7),
            ),
            name="変更前",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=sampled["Optimized_Map_RPM"],
            y=sampled["Optimized_Map_Torque_Nm"],
            mode="markers",
            marker=dict(
                size=7,
                color="#176b93",
                opacity=0.78,
                line=dict(color="rgba(255,255,255,0.9)", width=0.7),
            ),
            name="変更後",
        )
    )
    fig.update_layout(
        title="燃費率マップ上の走行点",
        xaxis_title="エンジン回転数 rpm",
        yaxis_title="エンジントルク Nm",
        height=680,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(size=13, color="#27322f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(55, 68, 64, 0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(55, 68, 64, 0.10)")
    return fig


def cvt_map_comparison_figure(
    current_map: BilinearMap,
    optimized_values: np.ndarray,
) -> go.Figure:
    z_min = float(np.nanmin([np.nanmin(current_map.values), np.nanmin(optimized_values)]))
    z_max = float(np.nanmax([np.nanmax(current_map.values), np.nanmax(optimized_values)]))
    colorscale = [
        [0.0, "#f3f5f4"],
        [0.25, "#d6dfdc"],
        [0.5, "#a8bdb7"],
        [0.75, "#6f948c"],
        [1.0, "#315f58"],
    ]
    contour_style = dict(
        showlabels=True,
        coloring="heatmap",
        labelfont=dict(size=12, color="#26332f"),
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("変更前", "変更後"),
    )
    fig.add_trace(
        go.Contour(
            x=current_map.x_axis,
            y=current_map.y_axis,
            z=current_map.values,
            zmin=z_min,
            zmax=z_max,
            colorscale=colorscale,
            contours=contour_style,
            line=dict(color="rgba(40, 55, 51, 0.38)", width=0.9),
            showscale=False,
            hovertemplate="%{x:.1f} km/h<br>%{y:.1f} %<br>%{z:.0f} rpm<extra>変更前</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            x=current_map.x_axis,
            y=current_map.y_axis,
            z=optimized_values,
            zmin=z_min,
            zmax=z_max,
            colorscale=colorscale,
            contours=contour_style,
            line=dict(color="rgba(40, 55, 51, 0.38)", width=0.9),
            colorbar=dict(title="rpm", thickness=16),
            hovertemplate="%{x:.1f} km/h<br>%{y:.1f} %<br>%{z:.0f} rpm<extra>変更後</extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="CVT変速線図マップ比較",
        height=520,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(size=13, color="#27322f"),
        margin=dict(l=55, r=35, t=85, b=55),
    )
    fig.update_xaxes(title_text="車速 km/h", showgrid=True, gridcolor="rgba(55, 68, 64, 0.10)")
    fig.update_yaxes(title_text="アクセル開度 %", showgrid=True, gridcolor="rgba(55, 68, 64, 0.10)", row=1, col=1)
    return fig


def map_heatmap_figure(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    title: str,
    unit: str,
) -> go.Figure:
    heatmap_args = dict(
        x=x_axis,
        y=y_axis,
        z=values,
        colorscale="RdBu",
        colorbar=dict(title=unit),
    )
    if np.nanmin(values) < 0 < np.nanmax(values):
        heatmap_args["zmid"] = 0
    fig = go.Figure(
        data=go.Heatmap(**heatmap_args)
    )
    fig.update_layout(
        title=title,
        xaxis_title="車速 km/h",
        yaxis_title="アクセル開度 %",
        height=420,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def choose_power_lines(evaluation: pd.DataFrame) -> list[float]:
    power = pd.to_numeric(evaluation["Power_kW"], errors="coerce")
    power = power[np.isfinite(power) & (power > 0)]
    if power.empty:
        return []
    high = float(np.nanpercentile(power, 95))
    step = 5.0 if high <= 40 else 10.0
    return [p for p in np.arange(step, high + step, step)]


def downsample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, max_rows).astype(int)].copy()


def wide_map_csv(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    index_label: str,
) -> bytes:
    frame = pd.DataFrame(values, index=y_axis, columns=x_axis)
    frame.index.name = index_label
    return frame.to_csv(float_format="%.6g").encode("utf-8-sig")


if __name__ == "__main__":
    main()
