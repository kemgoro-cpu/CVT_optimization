# CVT変速線図最適化ツール

走行ログ、現行CVT変速線図、燃費率マップから、走行データ全体の燃料消費量が下がるようにCVTの速度×アクセル開度RPMマップを作る初版です。

## 考え方

各走行点について、現在のエンジン回転数とトルクからエンジン出力を計算します。

```text
出力[kW] = エンジン回転数[rpm] * トルク[Nm] / 9549.2966
```

CVTで回転数を変えても、その瞬間に車両が要求している出力は同じと仮定します。候補回転数ごとに必要トルクを計算し、燃費率マップからBSFCを補間して燃料消費を比較します。

```text
候補トルク[Nm] = 出力[kW] * 9549.2966 / 候補回転数[rpm]
燃料消費[g/s] = BSFC[g/kWh] * 出力[kW] / 3600
```

その後、走行ログ上で求めた最適回転数を速度×アクセル開度のCVTマップ格子へ集約し、更新後のCVTマップとして出力します。

## 入力ファイル

### 走行ログ

CSVまたは`.trn`を読み込めます。`.trn`は以下のような2行ヘッダと空白区切りデータに対応しています。

```text
| Time    Target_Speed    Actual_Speed    Throttle    Engine_RPM    Engine_Torque
  s       km/h            km/h            %           rpm           Nm
  0.0     0.00            0.00            0.00        805           35.0
```

必要列:

- 時間: 既定 `Time`
- 車速: 既定 `Actual_Speed`
- アクセル開度: 既定 `Throttle`
- エンジン回転数: 既定 `Engine_RPM`
- エンジントルク: 既定 `Engine_Torque`

列名はCLI引数で変更できます。

### CVT変速線図

CSVのワイド形式です。X軸が車速、Y軸がアクセル開度、値がエンジン回転数です。

```csv
Throttle,0,20,40,60,80,100
0,800,900,1000,1100,1200,1300
20,900,1100,1300,1500,1700,1900
40,1200,1500,1800,2100,2400,2700
```

### 燃費率マップ

CSVのワイド形式です。X軸がエンジン回転数、Y軸がエンジントルク、値がBSFC `g/kWh` です。

```csv
Torque,1000,1500,2000,2500,3000,3500
20,360,330,310,320,340,370
40,310,270,245,250,270,300
80,290,245,225,230,250,280
```

## 使い方

アプリ起動:

```powershell
rtk python -m streamlit run cvt_optimizer/app.py --server.port 8501
```

ブラウザで以下を開きます。

```text
http://localhost:8501
```

アプリでは、現在のCVT変速線図と燃費率マップをExcelからそのまま貼り付けできます。走行データはCSVまたは`.trn`をアップロードします。

サンプル`.trn`の列確認:

```powershell
rtk python -m cvt_optimizer.cli inspect "C:\Users\kemgo\Documents\Program\antigravity\csv_viewer\NEDC_sample_A.trn"
```

最適化:

```powershell
rtk python -m cvt_optimizer.cli optimize `
  --drive-data "drive_fuel_mode.trn" `
  --cvt-map "current_cvt_map.csv" `
  --bsfc-map "bsfc_map.csv" `
  --output-dir "out" `
  --torque-col "Engine_Torque"
```

燃費モードだけを抽出する列がある場合:

```powershell
rtk python -m cvt_optimizer.cli optimize `
  --drive-data "drive.csv" `
  --cvt-map "current_cvt_map.csv" `
  --bsfc-map "bsfc_map.csv" `
  --output-dir "out" `
  --mode-col "Fuel_Mode" `
  --mode-value "ECO"
```

## 出力

- `optimized_cvt_map.csv`: 最適化後のCVT変速線図
- `target_rpm_map.csv`: 走行ログから直接求めた目標RPMの集約値
- `delta_rpm_map.csv`: 現行マップとの差分
- `coverage_count_map.csv`: 各マップセルに入った走行点数
- `drive_evaluation.csv`: 各走行点の現状、理論最適、最適化後マップ評価
- `optimization_summary.json`: 燃料消費量、g/km、改善率
- `cvt_current.png`, `cvt_optimized.png`, `cvt_delta.png`: 確認用ヒートマップ

アプリ画面では、燃費率マップを等高線で表示し、変更前と変更後の走行点をRPM×トルク平面に重ねて表示します。

## 後から追加しやすい制約案

- `--max-delta-rpm`: 現行マップからの変更量を±rpmで制限
- `--min-rpm`, `--max-rpm`: 使用回転数範囲を制限
- `--smooth-passes`, `--smooth-weight`: 隣接セルの段差をなだらかにする
- アクセル開度方向の単調性: 高アクセルで回転数が低くなりすぎないようにする
- 車速方向の連続性: 車速が少し変わっただけで回転数が急変しないようにする
- 高アクセル領域の性能優先: 例えばアクセル80%以上は更新しない、または改善幅を小さくする
- 低速/発進領域の固定: 発進フィーリングに関わる領域を更新対象外にする
