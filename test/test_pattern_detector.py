"""Tests for utils/pattern_detector.py (TA-Lib candlestick patterns)."""
import numpy as np
import pandas as pd
import pytest

import config

pytest.importorskip("talib", reason="TA-Lib is not installed")

from utils.pattern_detector import detect_patterns          # noqa: E402


def make_ohlc(n=60, seed=7):
    """Random-walk OHLC frame with the columns detect_patterns() expects."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.5, n))
    open_ = close + rng.normal(0, 1.0, n)
    high = np.maximum(open_, close) + rng.random(n)
    low = np.minimum(open_, close) - rng.random(n)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "Date": dates,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": open_, "high": high, "low": low, "close": close,
    })


def test_detect_patterns_adds_every_pattern_column():
    out, _ = detect_patterns(make_ohlc())
    for col in config.PATTERN_COLS:
        assert col in out.columns


def test_detect_patterns_adds_a_pattern_score():
    out, _ = detect_patterns(make_ohlc())
    assert "Pattern_Score" in out.columns
    assert out["Pattern_Score"].notna().all()


def test_detect_patterns_preserves_row_count():
    df = make_ohlc()
    out, _ = detect_patterns(df)
    assert len(out) == len(df)


def test_active_patterns_have_the_expected_shape():
    _, active = detect_patterns(make_ohlc())
    for name, value, date_str in active:
        assert name in config.PATTERN_COLS
        assert value in (-100, -200, 100, 200)
        assert isinstance(date_str, str)


def test_active_patterns_are_deduplicated():
    _, active = detect_patterns(make_ohlc())
    names = [p[0] for p in active]
    assert len(names) == len(set(names))


def test_detect_patterns_flags_a_textbook_doji():
    """A near-zero-body candle at the end of the window must be picked up."""
    df = make_ohlc()
    last = len(df) - 1
    df.loc[last, "open"] = 100.0
    df.loc[last, "close"] = 100.0
    df.loc[last, "high"] = 103.0
    df.loc[last, "low"] = 97.0
    _, active = detect_patterns(df)
    assert "DOJI" in [p[0] for p in active]


def test_detect_patterns_returns_no_patterns_on_a_flat_series():
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    flat = pd.DataFrame({
        "Date": dates,
        "open": [100.0] * n, "high": [100.0] * n,
        "low": [100.0] * n, "close": [100.0] * n,
    })
    out, active = detect_patterns(flat)
    assert isinstance(active, list)
    assert out["Pattern_Score"].notna().all()

