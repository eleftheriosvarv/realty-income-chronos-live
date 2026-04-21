import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt

# =========================
# User configuration
# =========================
TICKER = "O"
INTERVAL = "5m"
PERIOD = "60d"
PRICE_COL = "Close"

CONTEXT_LEN = 1048
HORIZON_LEN = 24
POLL_EVERY_MINUTES = 5

# IMPORTANT:
# For GitHub Actions, keep RUN_FOREVER=False and MAX_CYCLES=1.
RUN_FOREVER = False
MAX_CYCLES = 1

STATE_DIR = Path("realtime_o_chronos2_state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

FORECASTS_FILE = STATE_DIR / "forecasts.json"
EVALUATIONS_CSV = STATE_DIR / "evaluations.csv"
LATEST_FORECAST_CSV = STATE_DIR / "latest_forecast.csv"
SUMMARY_CSV = STATE_DIR / "evaluation_summary.csv"
LATEST_PLOT = STATE_DIR / "latest_forecast_plot.png"
METRICS_PLOT = STATE_DIR / "metrics_history.png"

# =========================
# Helpers
# =========================
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100.0)

def fetch_close_series(ticker=TICKER, period=PERIOD, interval=INTERVAL, price_col=PRICE_COL):
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        prepost=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found. Available columns: {list(df.columns)}")

    out = df[[price_col]].copy()
    out = out.dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.rename(columns={price_col: "target"})
    return out

def get_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Torch device: {device}")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    return pipeline

def extract_median_forecast(raw_forecast, horizon):
    if isinstance(raw_forecast, torch.Tensor):
        arr = raw_forecast.detach().cpu().numpy()
    else:
        arr = np.asarray(raw_forecast)

    arr = np.squeeze(arr)

    # Expected common cases:
    # 1D: [prediction_length]
    # 2D: [num_samples, prediction_length] or [prediction_length, num_quantiles]
    # 3D: [num_series, num_samples, prediction_length]
    if arr.ndim == 1:
        preds = arr[:horizon]

    elif arr.ndim == 2:
        if arr.shape[1] == horizon and arr.shape[0] != horizon:
            preds = np.median(arr, axis=0)
        elif arr.shape[0] == horizon and arr.shape[1] != horizon:
            preds = np.median(arr, axis=1)
        else:
            preds = np.median(arr, axis=0)[:horizon]

    elif arr.ndim == 3:
        arr0 = arr[0]
        if arr0.shape[1] == horizon:
            preds = np.median(arr0, axis=0)
        elif arr0.shape[0] == horizon:
            preds = np.median(arr0, axis=1)
        else:
            preds = np.median(arr0, axis=0)[:horizon]
    else:
        raise ValueError(f"Unexpected forecast output shape: {arr.shape}")

    preds = np.asarray(preds, dtype=float).reshape(-1)
    if len(preds) < horizon:
        raise ValueError(f"Forecast shorter than horizon: got {len(preds)}, expected {horizon}")
    return preds[:horizon]

def latest_observation_forecast(pipeline, series_df):
    if len(series_df) < CONTEXT_LEN:
        raise ValueError(f"Need at least {CONTEXT_LEN} rows, got {len(series_df)}.")

    context_df = series_df.iloc[-CONTEXT_LEN:].copy()
    origin_ts = context_df.index[-1]

    context_values = torch.tensor(
        context_df["target"].astype("float32").values,
        dtype=torch.float32
    )

    raw_forecast = pipeline.predict(
        context=context_values,
        prediction_length=HORIZON_LEN,
    )

    preds = extract_median_forecast(raw_forecast, HORIZON_LEN)

    return {
        "origin_ts": origin_ts.isoformat(),
        "created_at_utc": datetime.utcnow().isoformat(),
        "context_len": CONTEXT_LEN,
        "horizon_len": HORIZON_LEN,
        "price_col": PRICE_COL,
        "interval": INTERVAL,
        "ticker": TICKER,
        "predictions": [float(x) for x in preds],
    }

def score_ready_forecasts(series_df, forecasts_store):
    evaluations = []
    index_list = list(series_df.index)

    for forecast in forecasts_store:
        if forecast.get("evaluated", False):
            continue

        origin_ts = pd.Timestamp(forecast["origin_ts"])
        if origin_ts not in series_df.index:
            continue

        origin_pos = index_list.index(origin_ts)
        end_pos = origin_pos + HORIZON_LEN

        if end_pos >= len(series_df):
            continue

        actual_block = series_df.iloc[origin_pos + 1 : origin_pos + 1 + HORIZON_LEN]["target"].to_numpy(dtype=float)
        pred_block = np.asarray(forecast["predictions"], dtype=float)

        if len(actual_block) != HORIZON_LEN or len(pred_block) != HORIZON_LEN:
            continue

        forecast["evaluated"] = True
        forecast["actual_end_ts"] = series_df.index[origin_pos + HORIZON_LEN].isoformat()
        forecast["mae"] = mae(actual_block, pred_block)
        forecast["rmse"] = rmse(actual_block, pred_block)
        forecast["smape"] = smape(actual_block, pred_block)

        evaluations.append({
            "origin_ts": forecast["origin_ts"],
            "actual_end_ts": forecast["actual_end_ts"],
            "mae": forecast["mae"],
            "rmse": forecast["rmse"],
            "smape": forecast["smape"],
            "ticker": forecast["ticker"],
            "interval": forecast["interval"],
            "context_len": forecast["context_len"],
            "horizon_len": forecast["horizon_len"],
        })

    return evaluations

def save_evaluations_csv(all_forecasts):
    rows = []
    for f in all_forecasts:
        if f.get("evaluated", False):
            rows.append({
                "origin_ts": f["origin_ts"],
                "actual_end_ts": f.get("actual_end_ts"),
                "mae": f.get("mae"),
                "rmse": f.get("rmse"),
                "smape": f.get("smape"),
                "ticker": f.get("ticker"),
                "interval": f.get("interval"),
                "context_len": f.get("context_len"),
                "horizon_len": f.get("horizon_len"),
            })

    if rows:
        eval_df = pd.DataFrame(rows).sort_values("origin_ts")
    else:
        eval_df = pd.DataFrame(
            columns=["origin_ts", "actual_end_ts", "mae", "rmse", "smape", "ticker", "interval", "context_len", "horizon_len"]
        )

    eval_df.to_csv(EVALUATIONS_CSV, index=False)

    if not eval_df.empty:
        summary = pd.DataFrame([{
            "ticker": TICKER,
            "interval": INTERVAL,
            "n_evaluated_blocks": len(eval_df),
            "mean_mae": eval_df["mae"].mean(),
            "mean_rmse": eval_df["rmse"].mean(),
            "mean_smape": eval_df["smape"].mean(),
            "median_mae": eval_df["mae"].median(),
            "median_rmse": eval_df["rmse"].median(),
            "median_smape": eval_df["smape"].median(),
        }])
    else:
        summary = pd.DataFrame([{
            "ticker": TICKER,
            "interval": INTERVAL,
            "n_evaluated_blocks": 0,
            "mean_mae": np.nan,
            "mean_rmse": np.nan,
            "mean_smape": np.nan,
            "median_mae": np.nan,
            "median_rmse": np.nan,
            "median_smape": np.nan,
        }])

    summary.to_csv(SUMMARY_CSV, index=False)
    return eval_df, summary

def save_latest_forecast_csv(forecast_obj):
    origin_ts = pd.Timestamp(forecast_obj["origin_ts"])
    pred_df = pd.DataFrame({
        "step_ahead": list(range(1, HORIZON_LEN + 1)),
        "predicted": forecast_obj["predictions"],
    })
    pred_df.insert(0, "forecast_origin_ts", origin_ts.isoformat())
    pred_df.to_csv(LATEST_FORECAST_CSV, index=False)

def save_latest_plot(series_df, forecast_obj):
    preds = np.asarray(forecast_obj["predictions"], dtype=float)
    hist = series_df.iloc[-200:].copy()

    plt.figure(figsize=(12, 5))
    plt.plot(hist.index, hist["target"].values, label="Observed Close", linewidth=1.8)
    future_x = list(range(len(hist), len(hist) + HORIZON_LEN))
    plt.plot(future_x, preds, label="Chronos-2 forecast (next 24 bars)", linewidth=2.0)
    plt.axvline(x=len(hist) - 1, linestyle="--", linewidth=1.2)
    plt.title(f"{TICKER} | {INTERVAL} | Context={CONTEXT_LEN}, Horizon={HORIZON_LEN}")
    plt.xlabel("Recent history + forecast steps")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LATEST_PLOT, dpi=160)
    plt.close()

def save_metrics_plot(eval_df):
    if eval_df.empty:
        return

    plot_df = eval_df.copy()
    plot_df["origin_ts"] = pd.to_datetime(plot_df["origin_ts"])

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["origin_ts"], plot_df["mae"], label="MAE", marker="o")
    plt.plot(plot_df["origin_ts"], plot_df["rmse"], label="RMSE", marker="o")
    plt.plot(plot_df["origin_ts"], plot_df["smape"], label="sMAPE", marker="o")
    plt.title(f"{TICKER} | Forecast accuracy by completed 24-bar block")
    plt.xlabel("Forecast origin timestamp")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_PLOT, dpi=160)
    plt.close()

def run_cycle(pipeline):
    print("\n" + "=" * 80)
    print(f"Cycle started at UTC: {datetime.utcnow().isoformat()}")

    series_df = fetch_close_series()
    latest_ts = series_df.index[-1]
    print(f"Rows fetched: {len(series_df)}")
    print(f"Latest observed bar: {latest_ts}")

    if len(series_df) < CONTEXT_LEN:
        print("Not enough rows for the requested context length yet.")
        return

    forecasts_store = load_json(FORECASTS_FILE, default=[])

    latest_forecast = latest_observation_forecast(pipeline, series_df)
    latest_origin_ts = latest_forecast["origin_ts"]

    already_exists = any(f["origin_ts"] == latest_origin_ts for f in forecasts_store)

    if not already_exists:
        forecasts_store.append(latest_forecast)
        save_latest_forecast_csv(latest_forecast)
        save_latest_plot(series_df, latest_forecast)
        print(f"New forecast stored for origin: {latest_origin_ts}")
    else:
        print(f"Forecast for origin {latest_origin_ts} already exists. No duplicate added.")

    newly_scored = score_ready_forecasts(series_df, forecasts_store)
    save_json(FORECASTS_FILE, forecasts_store)

    eval_df, summary_df = save_evaluations_csv(forecasts_store)
    save_metrics_plot(eval_df)

    print(f"Newly evaluated forecast blocks this cycle: {len(newly_scored)}")
    print("\nLatest evaluation summary:")
    print(summary_df.to_string(index=False))

    if not eval_df.empty:
        print("\nMost recent evaluated blocks:")
        print(eval_df.tail(5).to_string(index=False))

    print("\nSaved files:")
    print(f"- {FORECASTS_FILE}")
    print(f"- {LATEST_FORECAST_CSV}")
    print(f"- {EVALUATIONS_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {LATEST_PLOT}")
    print(f"- {METRICS_PLOT}")

def main():
    pipeline = get_pipeline()

    cycle = 0
    while True:
        run_cycle(pipeline)
        cycle += 1

        if not RUN_FOREVER:
            break

        if MAX_CYCLES is not None and cycle >= MAX_CYCLES:
            break

        sleep_seconds = POLL_EVERY_MINUTES * 60
        print(f"Sleeping for {sleep_seconds} seconds...")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    main()
