"""
services/nse_service.py
━━━━━━━━━━━━━━━━━━━━━━━
Wraps all existing NSE-Neuron model / utility functions and exposes
clean dict-based results suitable for the REST API.
"""
import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from datetime import date

import config
from utils.data_fetcher import _load_or_fetch
from utils.preprocessor  import preprocess_nse_df
from utils.regime_detector import detect_regime, apply_regime_confidence
from utils.pattern_detector import detect_patterns, combine_regime_patterns

from models.lstm      import lstm
from models.bilstm    import bilstm
from models.gru       import gru
from models.cnn_lstm  import cnn_lstm
from models.classifiers.lstm     import lstm_classifier
from models.classifiers.bilstm   import bilstm_classifier
from models.classifiers.gru      import gru_classifier
from models.classifiers.cnn_lstm import cnn_lstm_classifier


# ── helpers ───────────────────────────────────────────────────────────────────

ALGO_FUNCS = {
    "lstm":     lstm,
    "bilstm":   bilstm,
    "gru":      gru,
    "cnn_lstm": cnn_lstm,
}

CLASSIFIER_FUNCS = {
    "lstm":     lstm_classifier,
    "bilstm":   bilstm_classifier,
    "gru":      gru_classifier,
    "cnn_lstm": cnn_lstm_classifier,
}

ALGO_DISPLAY = {
    "lstm": "LSTM", "bilstm": "BiLSTM",
    "gru": "GRU",   "cnn_lstm": "CNN-LSTM",
}


def fetch_data(symbol: str):
    """Fetch + preprocess data for a given NSE symbol."""
    from nselib import capital_market

    ticker_info = capital_market.equity_list()
    ticker_info = ticker_info[ticker_info["SYMBOL"] == symbol]

    if ticker_info.empty:
        raise ValueError(f"Symbol '{symbol}' not found in NSE equity list.")

    from_date = ticker_info[" DATE OF LISTING"].values[0]
    from_date = pd.to_datetime(from_date, dayfirst=True).strftime("%d-%m-%Y")
    to_date   = date.today().strftime("%d-%m-%Y")

    df      = _load_or_fetch(symbol, from_date, to_date)
    df      = preprocess_nse_df(df)          # also sets config.HISTORIC_DATA

    details = {
        "scheme_name": ticker_info["NAME OF COMPANY"].values[0],
        "scheme_code": str(symbol),
    }
    return df, details


def _historical_ohlc(n: int = 120) -> list:
    """Return last n rows of OHLC from config.HISTORIC_DATA (includes open)."""
    hist_df = config.HISTORIC_DATA
    if hist_df is None:
        return []
    # Mandatory columns — rows without these are useless for the chart
    mandatory = [c for c in ["date", "high", "low", "close"] if c in hist_df.columns]
    # Optional: include open only when the column exists
    optional  = [c for c in ["open"] if c in hist_df.columns]
    cols = mandatory + optional

    tail = hist_df[cols].tail(n).copy()
    # Only drop rows where the mandatory columns are missing
    tail = tail.dropna(subset=mandatory)

    records = []
    for _, row in tail.iterrows():
        rec: dict = {}
        for k in mandatory:
            rec[k] = str(row[k]) if k == "date" else (float(row[k]) if pd.notna(row[k]) else None)
        for k in optional:
            rec[k] = float(row[k]) if pd.notna(row[k]) else None   # None → omitted on JS side
        records.append(rec)
    return records


def _build_forecast_days(pred: np.ndarray, signals: list | None) -> list:
    """Convert raw prediction array to list of dicts with signal info."""
    last_date    = pd.to_datetime(config.HISTORIC_DATA["Date"]).max()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
    )
    result = []
    for i in range(config.FORECAST_DAYS):
        item = {
            "date":       future_dates[i].strftime("%Y-%m-%d"),
            "high":       round(float(pred[i][0]), 2),
            "low":        round(float(pred[i][1]), 2),
            "close":      round(float(pred[i][2]), 2),
            "prev_close": round(float(pred[i][3]), 2),
            "signal":     None,
        }
        if signals and i < len(signals):
            sig = signals[i]
            item["signal"] = {
                "label":            sig.get("label"),
                "confidence":       sig.get("confidence"),
                "confidence_orig":  sig.get("confidence_orig"),
                "confidence_delta": sig.get("confidence_delta"),
                "regime_adjusted":  sig.get("regime_adjusted", False),
                "regime_direction": sig.get("regime_direction", "—"),
            }
        result.append(item)
    return result


def _regime_to_dict(regime: dict) -> dict:
    """Clean regime dict for JSON serialisation."""
    return {
        "regime":            regime.get("regime", "UNKNOWN"),
        "recommended_model": regime.get("recommended_model", "LSTM"),
        "sufficient_data":   regime.get("sufficient_data", False),
        "rows":              regime.get("rows", 0),
        "sma_fast":          regime.get("sma_fast"),
        "sma_slow":          regime.get("sma_slow"),
        "description":       regime.get("description", ""),
    }


# ── Public service functions ──────────────────────────────────────────────────

def run_single_forecast(job, symbol: str, algorithm: str) -> dict:
    """Train one model, run classifier, apply regime → return result dict."""
    job.progress = f"Fetching data for {symbol}…"
    df, details  = fetch_data(symbol)

    if algorithm not in ALGO_FUNCS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    job.progress = f"Training {ALGO_DISPLAY[algorithm]} model…"
    pred, rmse   = ALGO_FUNCS[algorithm](df)
    rmse_val     = rmse["close"] if isinstance(rmse, dict) else float(rmse)

    job.progress = f"Running {ALGO_DISPLAY[algorithm]} classifier…"
    signals      = CLASSIFIER_FUNCS[algorithm](df, pred)

    job.progress = "Detecting market regime…"
    regime       = detect_regime(df)
    if regime["sufficient_data"] and signals:
        signals = apply_regime_confidence(signals, regime)

    return {
        "symbol":       symbol,
        "company_name": details["scheme_name"],
        "algorithm":    algorithm,
        "display_name": ALGO_DISPLAY[algorithm],
        "forecast":     _build_forecast_days(pred, signals),
        "rmse":         round(rmse_val, 6),
        "regime":       _regime_to_dict(regime),
        "historical":   _historical_ohlc(),
    }


def run_all_forecast(job, symbol: str) -> dict:
    """Train all four models, compare RMSE → return result dict."""
    job.progress = f"Fetching data for {symbol}…"
    df, details  = fetch_data(symbol)

    algos        = ["lstm", "bilstm", "gru", "cnn_lstm"]
    all_preds    = {}
    all_rmse     = {}

    for algo in algos:
        job.progress = f"Training {ALGO_DISPLAY[algo]}…"
        pred, rmse   = ALGO_FUNCS[algo](df)
        all_preds[algo] = pred.tolist()
        all_rmse[algo]  = round(rmse["close"] if isinstance(rmse, dict) else float(rmse), 6)

    job.progress = "Detecting market regime…"
    regime = detect_regime(df)

    # Build per-algo forecast day lists (no signals in "all" mode)
    last_date    = pd.to_datetime(config.HISTORIC_DATA["Date"]).max()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
    )

    algo_forecasts = {}
    for algo in algos:
        pred_arr = np.array(all_preds[algo])
        days = []
        for i in range(config.FORECAST_DAYS):
            days.append({
                "date":       future_dates[i].strftime("%Y-%m-%d"),
                "high":       round(float(pred_arr[i][0]), 2),
                "low":        round(float(pred_arr[i][1]), 2),
                "close":      round(float(pred_arr[i][2]), 2),
                "prev_close": round(float(pred_arr[i][3]), 2),
                "signal":     None,
            })
        algo_forecasts[algo] = days

    best_algo = min(all_rmse, key=all_rmse.get)

    return {
        "symbol":          symbol,
        "company_name":    details["scheme_name"],
        "algorithm":       "all",
        "display_name":    "All Algorithms",
        "algo_forecasts":  algo_forecasts,
        "all_rmse":        all_rmse,
        "best_algo":       best_algo,
        "regime":          _regime_to_dict(regime),
        "historical":      _historical_ohlc(),
    }


def run_regime_analysis(symbol: str) -> dict:
    """Fetch data, detect regime + candlestick patterns."""
    df, details = fetch_data(symbol)

    regime = detect_regime(df)

    pattern_df, active_pats = detect_patterns(config.HISTORIC_DATA)
    patterns = [
        {
            "name":      pat.replace("_", " "),
            "value":     val,
            "direction": "Bullish" if val > 0 else "Bearish",
            "date":      date_str,
        }
        for pat, val, date_str in active_pats
    ]

    # Combined insight
    insight = _get_combined_insight(regime, active_pats)

    return {
        "symbol":       symbol,
        "company_name": details["scheme_name"],
        "regime":       _regime_to_dict(regime),
        "patterns":     patterns,
        "insight":      insight,
        "historical":   _historical_ohlc(),
    }


def _get_combined_insight(regime: dict, active_pats: list) -> str | None:
    if not regime["sufficient_data"] or not active_pats:
        return None
    bullish = [p for p, v, d in active_pats if v > 0]
    bearish = [p for p, v, d in active_pats if v < 0]
    reg     = regime["regime"]
    if reg == "BEAR" and bullish:
        return "⚠️ Bullish pattern inside BEAR trend — possible reversal or dead-cat bounce. Wait for confirmation."
    if reg == "BULL" and bearish:
        return "⚠️ Bearish pattern inside BULL trend — possible short-term pullback. Trend is still up."
    if reg == "BULL" and bullish:
        return "✅ Bullish pattern confirms BULL regime — trend and candles are aligned. Strong buy signal."
    if reg == "BEAR" and bearish:
        return "🔴 Bearish pattern confirms BEAR regime — trend and candles are aligned. Strong sell signal."
    if reg == "SIDEWAYS" and bullish:
        return "🟡 Bullish pattern in SIDEWAYS market — possible breakout upward. Watch volume."
    if reg == "SIDEWAYS" and bearish:
        return "🟡 Bearish pattern in SIDEWAYS market — possible breakout downward. Watch volume."
    return None


def search_symbols(query: str, limit: int = 20) -> list:
    """Search NSE equity symbols by ticker or company name."""
    from nselib import capital_market
    eq = capital_market.equity_list()
    q  = query.upper().strip()
    mask = (
        eq["SYMBOL"].str.contains(q, na=False) |
        eq["NAME OF COMPANY"].str.upper().str.contains(q, na=False)
    )
    results = eq[mask].head(limit)
    return [
        {"symbol": row["SYMBOL"], "name": row["NAME OF COMPANY"]}
        for _, row in results.iterrows()
    ]

