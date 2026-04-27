"""
utils/regime_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━
Rule-Based Market Regime Detector — No ML, pure price logic.

Regime Logic (SMA Cross only):
───────────────────────────────
  SMA_FAST > SMA_SLOW  AND  close > SMA_SLOW              →  BULL
  SMA_FAST < SMA_SLOW  AND  close < SMA_SLOW              →  BEAR
  |close - SMA_SLOW| / SMA_SLOW < SIDEWAYS_TOLERANCE      →  SIDEWAYS
  Ambiguous cross                                          →  SIDEWAYS

Data-Safety Guard:
──────────────────
  If len(df) < MIN_REGIME_ROWS  →  returns a FALLBACK result dict
  so the caller never crashes; it simply uses the default model.

Confidence Adjustment:
──────────────────────
  BULL  + BUY  signal  → boost
  BULL  + SELL signal  → penalty
  BEAR  + SELL signal  → boost
  BEAR  + BUY  signal  → penalty
  SIDEWAYS / UNKNOWN   → no change
"""

import pandas as pd
from config import (
    MIN_REGIME_ROWS,
    REGIME_SMA_FAST,
    REGIME_SMA_SLOW,
    REGIME_SIDEWAYS_TOLERANCE,
    REGIME_CONFIDENCE_BOOST,
    REGIME_CONFIDENCE_PENALTY,
    REGIME_MODEL_MAP,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _classify_trend(close_last: float, sma_fast_last: float, sma_slow_last: float) -> str:
    """Classify trend as BULL / BEAR / SIDEWAYS based on SMA positions."""
    sideways_band = abs(close_last - sma_slow_last) / sma_slow_last
    if sideways_band < REGIME_SIDEWAYS_TOLERANCE:
        return 'SIDEWAYS'
    if sma_fast_last > sma_slow_last and close_last > sma_slow_last:
        return 'BULL'
    if sma_fast_last < sma_slow_last and close_last < sma_slow_last:
        return 'BEAR'
    return 'SIDEWAYS'   # ambiguous cross


# ── Public API ────────────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> dict:
    """
    Detect the current market regime from a preprocessed OHLC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column (numeric, sorted ascending by date).

    Returns
    -------
    dict with keys:
        regime            (str)  — 'BULL' | 'BEAR' | 'SIDEWAYS' | 'UNKNOWN'
        recommended_model (str)  — best model for this regime
        sufficient_data   (bool) — False means fallback was triggered
        rows              (int)  — how many rows were available
        sma_fast          (float)
        sma_slow          (float)
        description       (str)  — human-readable summary
    """
    rows = len(df)

    # ── Data-safety guard ─────────────────────────────────────────────────────
    if rows < MIN_REGIME_ROWS:
        print(f"\n  [Regime] Only {rows} rows available "
              f"(minimum required: {MIN_REGIME_ROWS}).")
        print(f"  [Regime] Insufficient data — falling back to standard prediction.\n")
        return {
            'regime':            'UNKNOWN',
            'recommended_model': 'LSTM',
            'sufficient_data':   False,
            'rows':              rows,
            'sma_fast':          None,
            'sma_slow':          None,
            'description':       f'Insufficient data ({rows} rows < {MIN_REGIME_ROWS} required). '
                                  'Using standard LSTM prediction.',
        }

    close = df['close'].astype(float)

    # ── Compute SMAs ──────────────────────────────────────────────────────────
    sma_fast_last = float(_compute_sma(close, REGIME_SMA_FAST).iloc[-1])
    sma_slow_last = float(_compute_sma(close, REGIME_SMA_SLOW).iloc[-1])
    close_last    = float(close.iloc[-1])

    # ── Classify ──────────────────────────────────────────────────────────────
    regime            = _classify_trend(close_last, sma_fast_last, sma_slow_last)
    recommended_model = REGIME_MODEL_MAP.get(regime, 'LSTM')

    description = (
        f"Market is {regime} | "
        f"SMA{REGIME_SMA_FAST}={sma_fast_last:.2f}, "
        f"SMA{REGIME_SMA_SLOW}={sma_slow_last:.2f} | "
        f"Close={close_last:.2f} | "
        f"Recommended model: {recommended_model}"
    )

    print(f"\n  [Regime] {description}\n")

    return {
        'regime':            regime,
        'recommended_model': recommended_model,
        'sufficient_data':   True,
        'rows':              rows,
        'sma_fast':          sma_fast_last,
        'sma_slow':          sma_slow_last,
        'description':       description,
    }


def apply_regime_confidence(signals: list, regime: dict) -> list:
    """
    Boost or penalise classifier signal confidence based on regime agreement.

    BULL  + BUY  → boost  |  BULL  + SELL → penalty
    BEAR  + SELL → boost  |  BEAR  + BUY  → penalty
    SIDEWAYS / UNKNOWN    → no change
    """
    if not signals or not regime.get('sufficient_data'):
        return signals

    trend    = regime.get('regime', 'UNKNOWN')
    adjusted = []

    for sig in signals:
        label      = sig['label']
        confidence = float(sig['confidence'])
        raw_conf   = confidence
        boosted    = False
        penalised  = False

        if trend == 'BULL':
            if label == 'BUY':
                confidence = min(100.0, confidence + REGIME_CONFIDENCE_BOOST)
                boosted = True
            elif label == 'SELL':
                confidence = max(0.0, confidence - REGIME_CONFIDENCE_PENALTY)
                penalised = True

        elif trend == 'BEAR':
            if label == 'SELL':
                confidence = min(100.0, confidence + REGIME_CONFIDENCE_BOOST)
                boosted = True
            elif label == 'BUY':
                confidence = max(0.0, confidence - REGIME_CONFIDENCE_PENALTY)
                penalised = True

        # SIDEWAYS / UNKNOWN — no change

        adjusted.append({
            **sig,
            'confidence':       round(confidence, 1),
            'raw_confidence':   round(raw_conf, 1),
            'regime_adjusted':  boosted or penalised,
            'regime_direction': '↑ boosted' if boosted else ('↓ penalised' if penalised else '—'),
        })

    return adjusted


def regime_summary_line(regime: dict) -> str:
    """Return a one-line string suitable for printing below the forecast table."""
    if not regime.get('sufficient_data'):
        return "  [Regime] Standard prediction (insufficient data for regime analysis)"
    emoji = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡'}.get(regime['regime'], '⚪')
    return (
        f"  [Regime] {emoji}  {regime['regime']}  |  "
        f"Recommended: {regime['recommended_model']}  |  "
        f"SMA{REGIME_SMA_FAST}={regime['sma_fast']:.2f}  "
        f"SMA{REGIME_SMA_SLOW}={regime['sma_slow']:.2f}"
    )
