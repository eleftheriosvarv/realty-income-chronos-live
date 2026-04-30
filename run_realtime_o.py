# =========================================================
# REAL-TIME REALTY INCOME (O) FORECASTING WITH CHRONOS-2
# - Fetches latest 5-minute data from Yahoo Finance
# - Creates a 24-step forecast
# - Stores forecast blocks in JSON
# - Scores completed forecast blocks once actuals are available
# - Saves:
#     1) latest_forecast.csv
#     2) latest_forecast_plot.png
#     3) latest_evaluated_forecast_plot.png
#     4) evaluations.csv
#     5) evaluation_summary.csv
#     6) metrics_history.png
# =========================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import yfinance as yf
from chronos import Chronos2Pipeline

warnings.filterwarnings("ignore")

# =========================================================
# USER SETTINGS
# =========================================================
TICKER = "O"
INTERVAL = "5m"
YF_PERIOD = "60d"

MODEL_NAME = "amazon/chronos-2"

CONTEXT_LEN = 2048
HORIZON_LEN = 24
NUM_SAMPLES = 20

STATE_DIR = Path("realtime_o_chronos2_state")
FORECASTS_JSON = STATE_DIR / "forecasts.json"
LATEST_FORECAST_CSV = STATE_DIR / "latest_forecast.csv"
LATEST_PLOT = STATE_DIR / "latest_forecast_plot.png"
LATEST_EVALUATED_PLOT = STATE_DIR / "latest_evaluated_forecast_plot.png"
EVALUATIONS_CSV = STATE_DIR / "evaluations.csv"
EVALUATION_SUMMARY_CSV = STATE_DIR / "evaluation_summary.csv"
METRICS_PLOT = STATE_DIR / "metrics_history.png"

EPS = 1e-8

_PIPELINE = None


# =========================================================
# BASIC HELPERS
# =========================================================
def ensure_state_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return pd.Timestamp.utcnow().isoformat()


def load_json_list(path: Path):
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []
    except Exception:
        return []


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def normalize_timestamp(ts):
    ts = pd.Timestamp(ts)

    if ts.tz is not None:
        ts = ts.tz_convert(None)

    return ts


def smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    denom = np.abs(actual) + np.abs(pred) + EPS
    return float(np.mean(200.0 * np.abs(pred - actual) / denom))


def mae(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    return float(np.mean(np.abs(actual - pred)))


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def save_placeholder_plot(path: Path, title: str, message: str):
    plt.figure(figsize=(12, 5))
    plt.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# =========================================================
# DATA FETCHING
# =========================================================
def fetch_price_series():
    df = yf.download(
        TICKER,
        period=YF_PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        prepost=False,
        threads=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    # Robust Close extraction for both normal and MultiIndex yfinance outputs.
    # Sometimes yfinance returns:
    #   normal columns: Open, High, Low, Close, Volume
    # and sometimes:
    #   MultiIndex columns: Close / O, Open / O, etc.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        elif "Close" in df.columns.get_level_values(-1):
            close = df.xs("Close", axis=1, level=-1)
        else:
            raise RuntimeError(f"Close column not found. Columns are: {df.columns}")
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"Close column not found. Columns are: {df.columns}")
        close = df["Close"]

    # If Close is still a DataFrame, keep the first column.
    # This fixes: AttributeError: 'DataFrame' object has no attribute 'to_frame'
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    series_df = pd.DataFrame({"target": close}).copy()
    series_df.index = pd.to_datetime(series_df.index)

    if getattr(series_df.index, "tz", None) is not None:
        series_df.index = series_df.index.tz_convert(None)

    series_df = series_df[~series_df.index.duplicated(keep="last")]
    series_df = series_df.sort_index()
    series_df = series_df.dropna()

    if len(series_df) < CONTEXT_LEN:
        raise RuntimeError(
            f"Not enough observations. Need at least {CONTEXT_LEN}, got {len(series_df)}."
        )

    return series_df


# =========================================================
# MODEL
# =========================================================
def get_pipeline():
    global _PIPELINE

    if _PIPELINE is None:
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "cuda" if use_cuda else "cpu"

        _PIPELINE = Chronos2Pipeline.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            torch_dtype=dtype,
        )

    return _PIPELINE


def extract_predictions_from_chronos_output(forecast_raw):
    """
    Makes Chronos output robust.

    Expected possible shapes:
    - Tensor/array with shape: (num_samples, horizon)
    - Tensor/array with shape: (1, num_samples, horizon)
    - Tensor/array with shape: (horizon,)
    - Tuple/list where first element is forecast tensor
    """

    if isinstance(forecast_raw, (tuple, list)):
        forecast_raw = forecast_raw[0]

    arr = to_numpy(forecast_raw)

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim == 2:
        preds = np.median(arr, axis=0)
    elif arr.ndim == 1:
        preds = arr
    else:
        raise RuntimeError(f"Unexpected forecast output shape: {arr.shape}")

    preds = np.asarray(preds, dtype=float).reshape(-1)

    if len(preds) < HORIZON_LEN:
        raise RuntimeError(
            f"Forecast shorter than expected. Expected {HORIZON_LEN}, got {len(preds)}."
        )

    preds = preds[:HORIZON_LEN]
    return preds


def generate_forecast(series_df):
    pipeline = get_pipeline()

    context_values = series_df["target"].values.astype(np.float32)[-CONTEXT_LEN:]
    context_tensor = torch.tensor(context_values)

    forecast_raw = pipeline.predict(
        context=context_tensor,
        prediction_length=HORIZON_LEN,
        num_samples=NUM_SAMPLES,
    )

    preds = extract_predictions_from_chronos_output(forecast_raw)

    origin_ts = normalize_timestamp(series_df.index[-1])

    forecast_obj = {
        "ticker": TICKER,
        "interval": INTERVAL,
        "origin_ts": origin_ts.isoformat(),
        "created_at": now_iso(),
        "context_len": CONTEXT_LEN,
        "horizon_len": HORIZON_LEN,
        "predictions": [float(x) for x in preds],
        "evaluated": False,
    }

    return forecast_obj


# =========================================================
# FORECAST STORE
# =========================================================
def deduplicate_forecasts(forecasts_store):
    keep = {}

    for f in forecasts_store:
        key = f.get("origin_ts")

        if key is None:
            continue

        keep[key] = f

    out = list(keep.values())
    out = sorted(out, key=lambda x: x["origin_ts"])

    return out


def maybe_append_new_forecast(forecasts_store, new_forecast):
    existing_keys = {f.get("origin_ts") for f in forecasts_store}

    if new_forecast["origin_ts"] not in existing_keys:
        forecasts_store.append(new_forecast)

    return deduplicate_forecasts(forecasts_store)


def get_latest_forecast(forecasts_store):
    if not forecasts_store:
        return None

    return sorted(forecasts_store, key=lambda x: x["origin_ts"])[-1]


# =========================================================
# EVALUATION
# =========================================================
def score_ready_forecasts(series_df, forecasts_store):
    index_list = list(series_df.index)
    index_map = {normalize_timestamp(ts): i for i, ts in enumerate(index_list)}

    for forecast in forecasts_store:
        if forecast.get("evaluated", False):
            continue

        if "origin_ts" not in forecast:
            continue

        origin_ts = normalize_timestamp(forecast["origin_ts"])

        if origin_ts not in index_map:
            continue

        origin_pos = index_map[origin_ts]

        # Need full actual block of length HORIZON_LEN after the forecast origin.
        if origin_pos + HORIZON_LEN >= len(series_df):
            continue

        actual_series = series_df.iloc[
            origin_pos + 1 : origin_pos + 1 + HORIZON_LEN
        ]["target"]

        actual_block = actual_series.to_numpy(dtype=float)
        pred_block = np.asarray(forecast["predictions"], dtype=float)

        n = min(len(actual_block), len(pred_block))

        if n == 0:
            continue

        actual_block = actual_block[:n]
        pred_block = pred_block[:n]
        actual_timestamps = actual_series.index[:n]

        forecast["actuals"] = [float(x) for x in actual_block]
        forecast["actual_timestamps"] = [
            normalize_timestamp(ts).isoformat() for ts in actual_timestamps
        ]

        forecast["mae"] = mae(actual_block, pred_block)
        forecast["rmse"] = rmse(actual_block, pred_block)
        forecast["smape"] = smape(actual_block, pred_block)
        forecast["evaluated"] = True
        forecast["evaluated_at"] = now_iso()

    return deduplicate_forecasts(forecasts_store)


def build_evaluations_df(forecasts_store):
    rows = []

    for f in forecasts_store:
        if f.get("evaluated", False):
            rows.append(
                {
                    "ticker": f.get("ticker", TICKER),
                    "interval": f.get("interval", INTERVAL),
                    "origin_ts": f.get("origin_ts"),
                    "created_at": f.get("created_at"),
                    "evaluated_at": f.get("evaluated_at"),
                    "mae": f.get("mae"),
                    "rmse": f.get("rmse"),
                    "smape": f.get("smape"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "interval",
                "origin_ts",
                "created_at",
                "evaluated_at",
                "mae",
                "rmse",
                "smape",
            ]
        )

    df = pd.DataFrame(rows)
    df["origin_ts"] = pd.to_datetime(df["origin_ts"])
    df = df.sort_values("origin_ts").reset_index(drop=True)

    return df


def save_evaluation_summary(eval_df):
    if eval_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "metric": "mae",
                    "mean": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "latest": np.nan,
                    "n_evaluated": 0,
                },
                {
                    "metric": "rmse",
                    "mean": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "latest": np.nan,
                    "n_evaluated": 0,
                },
                {
                    "metric": "smape",
                    "mean": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "latest": np.nan,
                    "n_evaluated": 0,
                },
            ]
        )
        summary.to_csv(EVALUATION_SUMMARY_CSV, index=False)
        return

    rows = []

    for metric in ["mae", "rmse", "smape"]:
        vals = eval_df[metric].dropna().astype(float)

        rows.append(
            {
                "metric": metric,
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
                "latest": float(vals.iloc[-1]) if len(vals) else np.nan,
                "n_evaluated": int(len(vals)),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(EVALUATION_SUMMARY_CSV, index=False)


# =========================================================
# CSV OUTPUTS
# =========================================================
def save_latest_forecast_csv(forecast_obj):
    if forecast_obj is None:
        pd.DataFrame(
            columns=["timestamp", "predicted_close"]
        ).to_csv(LATEST_FORECAST_CSV, index=False)
        return

    origin_ts = normalize_timestamp(forecast_obj["origin_ts"])
    preds = np.asarray(forecast_obj["predictions"], dtype=float)

    future_index = pd.date_range(
        start=origin_ts + pd.Timedelta(minutes=5),
        periods=len(preds),
        freq="5min",
    )

    out = pd.DataFrame(
        {
            "timestamp": future_index,
            "predicted_close": preds,
        }
    )

    out.to_csv(LATEST_FORECAST_CSV, index=False)


# =========================================================
# PLOTS
# =========================================================
def save_latest_plot(series_df, forecast_obj):
    if forecast_obj is None:
        save_placeholder_plot(
            LATEST_PLOT,
            f"{TICKER} | Latest Forecast",
            "No forecast available yet.",
        )
        return

    preds = np.asarray(forecast_obj["predictions"], dtype=float)
    hist = series_df.iloc[-200:].copy()
    origin_ts = normalize_timestamp(forecast_obj["origin_ts"])

    future_index = pd.date_range(
        start=origin_ts + pd.Timedelta(minutes=5),
        periods=len(preds),
        freq="5min",
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        hist.index,
        hist["target"].values,
        label="Observed Close",
        linewidth=1.8,
    )

    plt.plot(
        future_index,
        preds,
        label="Chronos-2 forecast next 24 bars",
        linewidth=2.0,
        marker="o",
    )

    plt.axvline(
        x=origin_ts,
        linestyle="--",
        linewidth=1.2,
        label="Forecast origin",
    )

    plt.title(
        f"{TICKER} | {INTERVAL} | Latest forecast | "
        f"Context={CONTEXT_LEN}, Horizon={HORIZON_LEN}"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(LATEST_PLOT, dpi=160)
    plt.close()


def save_metrics_plot(eval_df):
    if eval_df.empty:
        save_placeholder_plot(
            METRICS_PLOT,
            f"{TICKER} | Forecast accuracy by completed 24-bar block",
            "No completed forecast blocks yet.\nWait until enough future actual bars are available.",
        )
        return

    plot_df = eval_df.copy()
    plot_df["origin_ts"] = pd.to_datetime(plot_df["origin_ts"])

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df["origin_ts"],
        plot_df["mae"],
        marker="o",
        label="MAE",
    )
    plt.plot(
        plot_df["origin_ts"],
        plot_df["rmse"],
        marker="o",
        label="RMSE",
    )
    plt.plot(
        plot_df["origin_ts"],
        plot_df["smape"],
        marker="o",
        label="sMAPE",
    )

    plt.title(f"{TICKER} | Forecast accuracy by completed 24-bar block")
    plt.xlabel("Forecast origin timestamp")
    plt.ylabel("Metric value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(METRICS_PLOT, dpi=160)
    plt.close()


def save_latest_evaluated_forecast_plot(forecasts_store):
    evaluated = [
        f
        for f in forecasts_store
        if f.get("evaluated", False)
        and "actuals" in f
        and "actual_timestamps" in f
        and "predictions" in f
    ]

    if not evaluated:
        save_placeholder_plot(
            LATEST_EVALUATED_PLOT,
            f"{TICKER} | Latest evaluated forecast",
            "No evaluated forecast yet.\nMetrics will appear after the next 24 real bars become available.",
        )
        return

    latest = sorted(evaluated, key=lambda x: x["origin_ts"])[-1]

    actuals = np.asarray(latest["actuals"], dtype=float)
    preds = np.asarray(latest["predictions"], dtype=float)
    timestamps = pd.to_datetime(latest["actual_timestamps"])

    n = min(len(actuals), len(preds), len(timestamps))

    if n == 0:
        save_placeholder_plot(
            LATEST_EVALUATED_PLOT,
            f"{TICKER} | Latest evaluated forecast",
            "Evaluated forecast exists, but no plottable data was found.",
        )
        return

    actuals = actuals[:n]
    preds = preds[:n]
    timestamps = timestamps[:n]

    latest_smape = latest.get("smape", np.nan)
    latest_mae = latest.get("mae", np.nan)
    latest_rmse = latest.get("rmse", np.nan)

    plt.figure(figsize=(12, 5))

    plt.plot(
        timestamps,
        actuals,
        marker="o",
        linewidth=2,
        label="Actual Close",
    )

    plt.plot(
        timestamps,
        preds,
        marker="o",
        linewidth=2,
        label="Predicted Close",
    )

    plt.title(
        f"{TICKER} | Latest evaluated forecast | "
        f"sMAPE={latest_smape:.3f}% | "
        f"MAE={latest_mae:.4f} | "
        f"RMSE={latest_rmse:.4f}"
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(LATEST_EVALUATED_PLOT, dpi=160)
    plt.close()


# =========================================================
# MAIN CYCLE
# =========================================================
def run_cycle():
    ensure_state_dir()

    series_df = fetch_price_series()

    forecasts_store = load_json_list(FORECASTS_JSON)
    forecasts_store = deduplicate_forecasts(forecasts_store)

    current_origin_ts = normalize_timestamp(series_df.index[-1]).isoformat()
    existing_origins = {f.get("origin_ts") for f in forecasts_store}

    if current_origin_ts not in existing_origins:
        new_forecast = generate_forecast(series_df)
        forecasts_store = maybe_append_new_forecast(
            forecasts_store,
            new_forecast,
        )

    forecasts_store = score_ready_forecasts(
        series_df,
        forecasts_store,
    )

    save_json(FORECASTS_JSON, forecasts_store)

    latest_forecast = get_latest_forecast(forecasts_store)

    save_latest_forecast_csv(latest_forecast)
    save_latest_plot(series_df, latest_forecast)

    eval_df = build_evaluations_df(forecasts_store)
    eval_df.to_csv(EVALUATIONS_CSV, index=False)

    save_evaluation_summary(eval_df)
    save_metrics_plot(eval_df)
    save_latest_evaluated_forecast_plot(forecasts_store)

    print(f"Ticker: {TICKER}")
    print(f"Rows fetched: {len(series_df)}")
    print(f"Forecast blocks stored: {len(forecasts_store)}")
    print(f"Completed/evaluated forecast blocks: {len(eval_df)}")

    if not eval_df.empty:
        latest_eval = eval_df.iloc[-1]
        print(
            "Latest metrics -> "
            f"MAE={latest_eval['mae']:.4f}, "
            f"RMSE={latest_eval['rmse']:.4f}, "
            f"sMAPE={latest_eval['smape']:.3f}%"
        )


if __name__ == "__main__":
    run_cycle()
