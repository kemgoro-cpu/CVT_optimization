from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cvt_optimizer.io import read_drive_table
from cvt_optimizer.maps import BilinearMap, LinearCurve
from cvt_optimizer.optimizer import ColumnConfig, OptimizationOptions, optimize_cvt_map


def main() -> None:
    st.set_page_config(
        page_title="CVT Optimizer",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_app_theme()

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span>CVT CALIBRATION</span>'
            '<strong>Optimization Setup</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-label">走行データ</div>', unsafe_allow_html=True)
        uploaded_drive = st.file_uploader(
            "走行データ CSV / TRN",
            type=["csv", "trn", "txt"],
        )

        with st.expander("列マッピング", expanded=False):
            time_col = st.text_input("時間列", "Time")
            speed_col = st.text_input("車速列", "Actual_Speed")
            throttle_col = st.text_input("アクセル開度列", "Throttle")
            rpm_col = st.text_input("エンジン回転数列", "Engine_RPM")
            torque_col = st.text_input("エンジントルク列", "Engine_Torque")
            mode_col = st.text_input("モード列", "")
            mode_value = st.text_input("燃費モード値", "")

        with st.expander("最適化条件", expanded=False):
            min_rpm = optional_number("最小RPM", "")
            max_rpm = optional_number("最大RPM", "")
            rpm_step = st.number_input("RPM刻み", min_value=1.0, value=50.0, step=10.0)
            min_speed = st.number_input(
                "最小車速 km/h", min_value=0.0, value=1.0, step=1.0
            )
            min_power = st.number_input(
                "最小出力 kW", min_value=0.0, value=0.5, step=0.5
            )
            update_gain = st.slider("更新ゲイン", 0.0, 1.0, 1.0, 0.05)
            max_delta = optional_number("最大変更RPM", "")
            smooth_passes = st.number_input("平滑化回数", min_value=0, value=0, step=1)
            smooth_weight = st.slider("平滑化強さ", 0.0, 1.0, 0.15, 0.05)
            monotonic_enabled = st.toggle("単調増加制約", value=False)
            monotonic_speed_selected = st.checkbox(
                "車速方向", value=True, disabled=not monotonic_enabled
            )
            monotonic_throttle_selected = st.checkbox(
                "アクセル方向", value=True, disabled=not monotonic_enabled
            )
            monotonic_speed = monotonic_enabled and monotonic_speed_selected
            monotonic_throttle = monotonic_enabled and monotonic_throttle_selected

    render_page_header()
    render_section_header("INPUT MAPS", "マップ入力")

    left, right = st.columns(2)
    with left:
        cvt_text = st.text_area(
            "現在のCVT変速線図",
            height=290,
            placeholder="Throttle\t0\t20\t40\t60\n0\t800\t900\t1000\t1100\n20\t1200\t1500\t1800\t2100",
        )
    with right:
        bsfc_text = st.text_area(
            "燃費率マップ",
            height=290,
            placeholder="Torque\t1000\t1500\t2000\t2500\n20\t360\t330\t310\t320\n40\t310\t270\t245\t250",
        )
    max_torque_text = st.text_area(
        "エンジン最大トルクカーブ",
        height=175,
        placeholder="Engine_RPM\tMax_Torque_Nm\n800\t80\n1500\t135\n2500\t175\n3500\t175\n4500\t140",
    )

    run = st.button("最適化を実行", type="primary", use_container_width=True)

    if not run:
        return

    if uploaded_drive is None:
        st.error("走行データを選択してください。")
        return
    if not cvt_text.strip() or not bsfc_text.strip() or not max_torque_text.strip():
        st.error(
            "CVT変速線図、燃費率マップ、エンジン最大トルクカーブを"
            "貼り付けてください。"
        )
        return

    try:
        with st.spinner("最適化を実行中"):
            drive = read_uploaded_drive(uploaded_drive)
            cvt_map = parse_pasted_map(cvt_text, "CVT変速線図")
            bsfc_map = parse_pasted_map(bsfc_text, "燃費率マップ")
            max_torque_curve = parse_pasted_curve(
                max_torque_text,
                "エンジン最大トルクカーブ",
            )
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
                monotonic_speed=monotonic_speed,
                monotonic_throttle=monotonic_throttle,
            )
            result = optimize_cvt_map(
                drive,
                cvt_map,
                bsfc_map,
                columns,
                options,
                max_torque_curve,
            )
    except Exception as exc:  # noqa: BLE001 - Streamlit should surface data issues.
        st.error(str(exc))
        return

    render_section_header("RESULTS", "最適化結果")
    show_torque_validation(result.summary)
    show_summary(result.summary)

    operating_tab, cvt_tab, diagnostics_tab = st.tabs(
        ["燃費率と走行点", "CVT線図比較", "変更量とカバレッジ"]
    )
    with operating_tab:
        st.plotly_chart(
            bsfc_contour_figure(
                bsfc_map,
                max_torque_curve,
                result.drive_evaluation,
                columns,
            ),
            use_container_width=True,
        )
    with cvt_tab:
        st.plotly_chart(
            cvt_map_comparison_figure(cvt_map, result.optimized_map),
            use_container_width=True,
        )
    with diagnostics_tab:
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
            st.markdown(
                '<p class="chart-note"><strong>走行点カバレッジ</strong>は、各セルに'
                '割り当てられた有効走行点の件数です。濃いほど、その速度×アクセル開度領域を'
                '走行した回数が多いことを示します。</p>',
                unsafe_allow_html=True,
            )

    render_section_header("EXPORT", "出力")
    map_export_enabled = bool(result.summary.get("fuel_result_valid", False))
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "最適化後CVT CSV",
            wide_map_csv(cvt_map.x_axis, cvt_map.y_axis, result.optimized_map, "Throttle"),
            "optimized_cvt_map.csv",
            "text/csv",
            use_container_width=True,
            disabled=not map_export_enabled,
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
            disabled=not map_export_enabled,
        )
    with d3:
        st.download_button(
            "走行点評価 CSV",
            result.drive_evaluation.to_csv(index=False).encode("utf-8-sig"),
            "drive_evaluation.csv",
            "text/csv",
            use_container_width=True,
        )


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --canvas: #f2f5f5;
            --surface: #ffffff;
            --surface-2: #e8eeee;
            --ink: #172328;
            --muted: #607076;
            --line: #cbd5d8;
            --accent: #0b7a75;
            --accent-hover: #08645f;
            --sidebar: #1a2529;
            --sidebar-2: #223136;
        }

        html, body, [class*="css"] {
            font-family: "Bahnschrift", "Yu Gothic UI", sans-serif;
            letter-spacing: 0;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--canvas);
            color: var(--ink);
        }

        [data-testid="stMain"],
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span {
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(242, 245, 245, 0.94);
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1500px;
            padding: 1.4rem 2.5rem 4rem;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid #33454b;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] summary,
        [data-testid="stSidebar"] span {
            color: #e8efef;
        }

        [data-testid="stSidebar"] input {
            color: var(--ink) !important;
            background: #f8fafa !important;
        }

        [data-testid="stSidebar"] [data-baseweb="input"] {
            background: #f8fafa;
            border-color: #53666c;
            border-radius: 4px;
        }

        [data-testid="stSidebar"] details {
            background: var(--sidebar-2);
            border: 1px solid #3a4c52;
            border-radius: 4px;
            margin-top: 0.7rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--sidebar-2);
            border: 1px dashed #71858b;
            border-radius: 4px;
        }

        .sidebar-brand {
            border-bottom: 1px solid #3a4a4f;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
        }

        .sidebar-brand > span,
        .page-title > div > span,
        .section-heading > span,
        .sidebar-label {
            color: #5aaea8 !important;
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .sidebar-brand strong {
            color: #ffffff;
            display: block;
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 0.28rem;
        }

        .sidebar-label {
            margin-bottom: 0.35rem;
        }

        .page-title {
            align-items: end;
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            margin-bottom: 1.4rem;
            padding: 0.45rem 0 1.15rem;
        }

        .page-title h1 {
            color: var(--ink) !important;
            font-size: 2.2rem;
            font-weight: 650;
            letter-spacing: 0;
            line-height: 1.08;
            margin: 0.3rem 0 0;
            white-space: nowrap;
        }

        .page-title code {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 3px;
            color: #3f5157;
            font-family: "Cascadia Mono", monospace;
            font-size: 0.72rem;
            padding: 0.32rem 0.48rem;
        }

        .section-heading {
            margin: 1.35rem 0 0.65rem;
        }

        .section-heading h2 {
            color: var(--ink) !important;
            font-size: 1.35rem;
            font-weight: 650;
            letter-spacing: 0;
            line-height: 1.2;
            margin: 0.2rem 0 0;
        }

        [data-testid="stTextArea"] textarea {
            background: var(--surface);
            border: 1px solid #b8c5c8;
            border-radius: 4px;
            color: #243238 !important;
            font-family: "Cascadia Mono", "Consolas", monospace;
            font-size: 0.84rem;
            line-height: 1.55;
            -webkit-text-fill-color: #243238;
        }

        [data-testid="stTextArea"] label p {
            color: #34464d !important;
            font-size: 0.86rem;
            font-weight: 650;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(11, 122, 117, 0.14);
        }

        button[kind="primary"], [data-testid="stBaseButton-primary"] {
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 4px !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            min-height: 46px;
        }

        button[kind="primary"] p,
        [data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
        }

        button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
            background: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 4px;
            min-height: 105px;
            padding: 0.9rem 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.82rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: "Bahnschrift", sans-serif;
            font-size: 1.75rem;
        }

        [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--line);
            gap: 1.5rem;
        }

        [data-baseweb="tab"] {
            color: var(--muted);
            font-weight: 600;
            padding-left: 0;
            padding-right: 0;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            color: var(--accent);
        }

        [data-testid="stDownloadButton"] button {
            background: var(--surface);
            border: 1px solid #aebdc1;
            border-radius: 4px;
            color: var(--ink);
            min-height: 44px;
        }

        [data-testid="stDownloadButton"] button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        .chart-note {
            background: #e7efef;
            border: 1px solid #c2d0d2;
            border-radius: 4px;
            color: #40545b !important;
            font-size: 0.82rem;
            line-height: 1.55;
            margin: -0.25rem 0 0;
            padding: 0.7rem 0.8rem;
        }

        .chart-note strong {
            color: #1f373d;
        }

        @media (max-width: 800px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1rem 1rem 3rem;
            }
            .page-title {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.8rem;
            }
            .page-title h1 {
                font-size: 1.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown(
        """
        <div class="page-title">
            <div><span>CVT CALIBRATION</span><h1>変速線図最適化</h1></div>
            <code>DRIVE-CYCLE / BSFC</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="section-heading"><span>{kicker}</span><h2>{title}</h2></div>',
        unsafe_allow_html=True,
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


def parse_pasted_curve(text: str, name: str) -> LinearCurve:
    frame = parse_pasted_table(text)
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"{name}はRPMとトルクの2列、2点以上が必要です。")
    x_axis = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    if np.isnan(x_axis).any() or np.isnan(values).any():
        raise ValueError(f"{name}に空欄または数値ではない値があります。")
    order = np.argsort(x_axis)
    return LinearCurve(x_axis[order], values[order], name=name)


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


def show_torque_validation(summary: dict) -> None:
    current_invalid = int(summary.get("current_torque_infeasible_rows", 0))
    optimized_invalid = int(summary.get("optimized_torque_infeasible_rows", 0))
    target_unavailable = int(summary.get("target_unavailable_rows", 0))
    if current_invalid:
        st.error(
            f"現在の走行点のうち {current_invalid} 点が最大トルクカーブを超えています。"
            "トルクカーブまたは走行データの単位・列を確認してください。"
        )
    if optimized_invalid:
        st.warning(
            f"最適化後マップに最大トルク制約外の走行点が {optimized_invalid} 点あります。"
            "燃費改善値は無効として表示しています。"
        )
    if target_unavailable:
        st.warning(
            f"BSFC範囲と最大トルク制約を同時に満たす候補がない走行点が "
            f"{target_unavailable} 点あります。"
        )


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
    max_torque_curve: LinearCurve,
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
                labelfont=dict(
                    family="Bahnschrift, Yu Gothic UI, sans-serif",
                    size=14,
                    color="#1d3035",
                ),
            ),
            colorscale=[
                [0.0, "#7ca8a1"],
                [0.25, "#abc4bd"],
                [0.5, "#d9ddd4"],
                [0.75, "#ddbea0"],
                [1.0, "#bd826d"],
            ],
            line=dict(color="rgba(32, 48, 53, 0.62)", width=1.25),
            colorbar=dict(
                title=dict(text="g/kWh", side="top"),
                thickness=18,
                tickfont=dict(size=12, color="#34484f"),
                outlinecolor="#9aabad",
                outlinewidth=1,
            ),
            hovertemplate="%{x:.0f} rpm<br>%{y:.1f} Nm<br>%{z:.1f} g/kWh<extra></extra>",
            name="BSFC",
        )
    )

    curve_min_rpm = max(bsfc_map.x_min, max_torque_curve.x_min)
    curve_max_rpm = min(bsfc_map.x_max, max_torque_curve.x_max)
    curve_rpm = np.linspace(curve_min_rpm, curve_max_rpm, 260)
    curve_torque = max_torque_curve.interpolate(curve_rpm)
    upper_torque = np.full_like(curve_torque, bsfc_map.y_max)
    shaded_curve_torque = np.clip(curve_torque, bsfc_map.y_min, bsfc_map.y_max)
    visible_curve_torque = np.where(
        (curve_torque >= bsfc_map.y_min) & (curve_torque <= bsfc_map.y_max),
        curve_torque,
        np.nan,
    )
    fig.add_trace(
        go.Scatter(
            x=np.r_[curve_rpm, curve_rpm[::-1]],
            y=np.r_[shaded_curve_torque, upper_torque[::-1]],
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(124, 73, 58, 0.12)",
            name="最大トルク超過領域",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_rpm,
            y=visible_curve_torque,
            mode="lines",
            line=dict(color="#733f32", width=2.6),
            name="最大トルク",
            hovertemplate="%{x:.0f} rpm<br>最大 %{y:.1f} Nm<extra></extra>",
        )
    )

    for line_index, power_kw in enumerate(choose_power_lines(evaluation)):
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
                line=dict(width=1.8, color="rgba(36, 47, 51, 0.66)", dash="dash"),
                name=f"{power_kw:.0f} kW",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        line_rpm = rpm[mask]
        line_torque = torque[mask]
        label_fraction = 0.58 + 0.08 * (line_index % 3)
        label_index = min(int(len(line_rpm) * label_fraction), len(line_rpm) - 1)
        fig.add_annotation(
            x=float(line_rpm[label_index]),
            y=float(line_torque[label_index]),
            text=f"{power_kw:.0f} kW",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.88)",
            bordercolor="rgba(73, 87, 92, 0.48)",
            borderwidth=1,
            borderpad=2,
            font=dict(
                family="Bahnschrift, Yu Gothic UI, sans-serif",
                size=11,
                color="#27383e",
            ),
        )

    sampled = downsample(evaluation, 5000)
    fig.add_trace(
        go.Scattergl(
            x=sampled[columns.rpm],
            y=sampled[columns.torque],
            mode="markers",
            marker=dict(
                size=9,
                symbol="x",
                color="#c3432e",
                opacity=0.9,
                line=dict(color="#ffffff", width=1.2),
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
                size=8,
                symbol="circle",
                color="#006d8d",
                opacity=0.88,
                line=dict(color="#ffffff", width=1.1),
            ),
            name="変更後",
        )
    )
    fig.update_layout(
        title="燃費率マップ上の走行点",
        xaxis_title="エンジン回転数 rpm",
        yaxis_title="エンジントルク Nm",
        height=700,
        plot_bgcolor="#f9fbfb",
        paper_bgcolor="#ffffff",
        font=dict(
            family="Bahnschrift, Yu Gothic UI, sans-serif",
            size=13,
            color="#27383e",
        ),
        title_font=dict(size=19, color="#17282e"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#a9b7ba",
            borderwidth=1,
            font=dict(size=13, color="#27383e"),
        ),
        margin=dict(l=72, r=42, t=92, b=70),
    )
    apply_chart_axes(fig)
    add_map_gridlines(fig, bsfc_map.x_axis, bsfc_map.y_axis)
    return fig


def cvt_map_comparison_figure(
    current_map: BilinearMap,
    optimized_values: np.ndarray,
) -> go.Figure:
    z_min = float(np.nanmin([np.nanmin(current_map.values), np.nanmin(optimized_values)]))
    z_max = float(np.nanmax([np.nanmax(current_map.values), np.nanmax(optimized_values)]))
    contour_step = nice_contour_step((z_max - z_min) / 8.0)
    contour_start = np.floor(z_min / contour_step) * contour_step
    contour_end = np.ceil(z_max / contour_step) * contour_step
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=current_map.x_axis,
            y=current_map.y_axis,
            z=current_map.values,
            zmin=z_min,
            zmax=z_max,
            contours=dict(
                start=contour_start,
                end=contour_end,
                size=contour_step,
                coloring="none",
                showlabels=False,
            ),
            line=dict(color="#c2603f", width=1.8, dash="dash"),
            showscale=False,
            showlegend=False,
            hovertemplate="%{x:.1f} km/h<br>%{y:.1f} %<br>%{z:.0f} rpm<extra>変更前</extra>",
        )
    )
    fig.add_trace(
        go.Contour(
            x=current_map.x_axis,
            y=current_map.y_axis,
            z=optimized_values,
            zmin=z_min,
            zmax=z_max,
            contours=dict(
                start=contour_start,
                end=contour_end,
                size=contour_step,
                coloring="none",
                showlabels=True,
                labelfont=dict(
                    family="Bahnschrift, Yu Gothic UI, sans-serif",
                    size=12,
                    color="#075f5c",
                ),
            ),
            line=dict(color="#087b76", width=2.4),
            showscale=False,
            showlegend=False,
            hovertemplate="%{x:.1f} km/h<br>%{y:.1f} %<br>%{z:.0f} rpm<extra>変更後</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#c2603f", width=1.8, dash="dash"),
            name="変更前",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#087b76", width=2.4),
            name="変更後",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="CVT変速線図マップ比較（重ね表示）",
        height=620,
        plot_bgcolor="#f9fbfb",
        paper_bgcolor="#ffffff",
        font=dict(
            family="Bahnschrift, Yu Gothic UI, sans-serif",
            size=13,
            color="#27383e",
        ),
        title_font=dict(size=19, color="#17282e"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#a9b7ba",
            borderwidth=1,
        ),
        margin=dict(l=72, r=42, t=92, b=70),
    )
    apply_chart_axes(fig)
    fig.update_xaxes(title_text="車速 km/h")
    fig.update_yaxes(title_text="アクセル開度 %")
    add_map_gridlines(fig, current_map.x_axis, current_map.y_axis)
    return fig


def nice_contour_step(raw_step: float) -> float:
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 100.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def map_heatmap_figure(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    title: str,
    unit: str,
) -> go.Figure:
    diverging = [
        [0.0, "#ad5948"],
        [0.5, "#f1f3f1"],
        [1.0, "#247584"],
    ]
    sequential = [
        [0.0, "#f1f4f3"],
        [0.35, "#c5d9d5"],
        [0.7, "#7eaaa3"],
        [1.0, "#346f69"],
    ]
    heatmap_args = dict(
        x=x_axis,
        y=y_axis,
        z=values,
        colorscale=diverging if unit == "rpm" else sequential,
        colorbar=dict(
            title=dict(text=unit, side="top"),
            thickness=16,
            outlinecolor="#9aabad",
            outlinewidth=1,
        ),
        hovertemplate="%{x:.1f} km/h<br>%{y:.1f} %<br>%{z:.0f} "
        + unit
        + "<extra></extra>",
    )
    if np.nanmin(values) < 0 < np.nanmax(values):
        heatmap_args["zmid"] = 0
    fig = go.Figure(data=go.Heatmap(**heatmap_args))
    fig.update_layout(
        title=title,
        xaxis_title="車速 km/h",
        yaxis_title="アクセル開度 %",
        height=460,
        plot_bgcolor="#f9fbfb",
        paper_bgcolor="#ffffff",
        font=dict(
            family="Bahnschrift, Yu Gothic UI, sans-serif",
            size=12,
            color="#27383e",
        ),
        title_font=dict(size=17, color="#17282e"),
        margin=dict(l=65, r=35, t=75, b=65),
    )
    apply_chart_axes(fig)
    add_map_gridlines(fig, x_axis, y_axis)
    return fig


def apply_chart_axes(fig: go.Figure) -> None:
    axis_style = dict(
        showgrid=True,
        gridcolor="#ccd6d9",
        gridwidth=1.15,
        showline=True,
        linecolor="#607279",
        linewidth=1.4,
        mirror=True,
        ticks="outside",
        tickcolor="#607279",
        ticklen=6,
        tickwidth=1.2,
        tickfont=dict(size=12, color="#34484f"),
        title_font=dict(size=15, color="#1f3339"),
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)


def add_map_gridlines(
    fig: go.Figure,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    columns: int = 1,
) -> None:
    x_grid = sample_grid_axis(x_axis)
    y_grid = sample_grid_axis(y_axis)
    line = dict(color="rgba(63, 79, 85, 0.23)", width=0.8)
    for column in range(1, columns + 1):
        for x_value in x_grid:
            fig.add_vline(x=float(x_value), line=line, layer="above", row=1, col=column)
        for y_value in y_grid:
            fig.add_hline(y=float(y_value), line=line, layer="above", row=1, col=column)


def sample_grid_axis(axis: np.ndarray, max_lines: int = 16) -> np.ndarray:
    if len(axis) <= max_lines:
        return axis
    indices = np.linspace(0, len(axis) - 1, max_lines).round().astype(int)
    return axis[np.unique(indices)]


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
