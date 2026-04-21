# Realty Income Chronos Live

This repository runs a scheduled GitHub Actions workflow every 5 minutes to:

1. Fetch the latest 5-minute candles for `O` (Realty Income) from Yahoo Finance.
2. Build a zero-shot Chronos-2 forecast with:
   - context length = 1048
   - horizon length = 24
3. Store each forecast block.
4. Score completed forecast blocks once the next 24 observed bars are available.

## Files

- `run_realtime_o.py` — main forecasting script
- `requirements.txt` — Python dependencies
- `.github/workflows/realtime_o.yml` — scheduled workflow

## Output directory

The workflow writes all state and results to:

`realtime_o_chronos2_state/`

Expected files:
- `forecasts.json`
- `latest_forecast.csv`
- `evaluations.csv`
- `evaluation_summary.csv`
- `latest_forecast_plot.png`
- `metrics_history.png`

## First run

You can trigger the workflow manually from the **Actions** tab using **workflow_dispatch**.
