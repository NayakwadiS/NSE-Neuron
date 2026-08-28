"""
test/conftest.py
Shared pytest fixtures + import-path bootstrap for the NSE-Neuron test suite.

The tests are deliberately dependency-light: nothing here touches the network,
the NSE API, or TensorFlow unless a test explicitly opts in. That keeps
`pytest` fast enough to run on every commit.
"""
import os
import sys

import pandas as pd
import pytest

# Make the project root importable (config, utils, models, src ...)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


RAW_COLUMNS = [
    "Date", "ClosePrice", "PrevClose", "HighPrice", "LowPrice", "OpenPrice",
    "TotalTradedQuantity", "DeliverableQty",
]


def make_raw_row(date, close, prev_close, high, low, open_, volume="1,000", deliv="500"):
    return {
        "Date":                date,
        "ClosePrice":          close,
        "PrevClose":           prev_close,
        "HighPrice":           high,
        "LowPrice":            low,
        "OpenPrice":           open_,
        "TotalTradedQuantity": volume,
        "DeliverableQty":      deliv,
    }


@pytest.fixture
def raw_nse_df():
    """A small raw NSE-style frame: unsorted, with a duplicate date and commas."""
    rows = [
        make_raw_row("03-Jan-2024", "1,010.50", "1,000.00", "1,020.00", "995.00", "1,000.00", "1,20,000"),
        make_raw_row("01-Jan-2024", "1,000.00", "990.00", "1,005.00", "985.00", "990.00", "80,000"),
        make_raw_row("02-Jan-2024", "1,005.25", "1,000.00", "1,010.00", "1,000.00", "1,000.00", "95,000"),
        # duplicate of 02-Jan - the LAST occurrence must win
        make_raw_row("02-Jan-2024", "1,006.00", "1,000.00", "1,011.00", "1,001.00", "1,000.00", "96,000"),
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def raw_nse_df_no_volume(raw_nse_df):
    """Older cached CSVs may not carry the volume columns at all."""
    return raw_nse_df.drop(columns=["TotalTradedQuantity", "DeliverableQty"])


def make_price_df(closes, start="2015-01-01"):
    """Build a preprocessed-style frame (Date/date/close/high/low/prev_close)."""
    dates = pd.bdate_range(start=start, periods=len(closes))
    closes = [float(c) for c in closes]
    return pd.DataFrame({
        "Date":       dates,
        "date":       [d.strftime("%Y-%m-%d") for d in dates],
        "open":       closes,
        "high":       [c * 1.01 for c in closes],
        "low":        [c * 0.99 for c in closes],
        "close":      closes,
        "prev_close": [closes[0]] + closes[:-1],
        "volume":     [10000 + i * 10 for i in range(len(closes))],
    })


@pytest.fixture
def price_df_factory():
    return make_price_df


@pytest.fixture
def bull_df():
    """Steadily rising series - golden-cross BULL regime."""
    return make_price_df([100 + i * 0.5 for i in range(1200)])


@pytest.fixture
def bear_df():
    """Steadily falling series - death-cross BEAR regime."""
    return make_price_df([700 - i * 0.5 for i in range(1200)])


@pytest.fixture
def sideways_df():
    """Flat series - SIDEWAYS regime (close sits on top of the slow SMA)."""
    return make_price_df([100.0 for _ in range(1200)])


@pytest.fixture(autouse=True)
def reset_historic_data():
    """config.HISTORIC_DATA is a module-level global - never leak it between tests."""
    import config
    original = config.HISTORIC_DATA
    yield
    config.HISTORIC_DATA = original

