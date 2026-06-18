# Demo data

Use these files to check the app immediately.

## App workflow

1. Start the app.

```powershell
rtk python -m streamlit run cvt_optimizer/app.py --server.port 8501
```

2. Upload `demo_drive.csv`.
3. Open `current_cvt_map_for_paste.tsv`, select all, copy, and paste it into "現在のCVT変速線図".
4. Open `bsfc_map_for_paste.tsv`, select all, copy, and paste it into "燃費率マップ".
5. Click "最適化を実行".

## CLI workflow

```powershell
rtk python -m cvt_optimizer.cli optimize `
  --drive-data demo_data/demo_drive.csv `
  --cvt-map demo_data/current_cvt_map.csv `
  --bsfc-map demo_data/bsfc_map.csv `
  --output-dir out/demo
```

`demo_drive.trn` contains the same demo drive data in TRN format.

