"""
Tests for src/backend/services/nse_service.py.

Only the pure, deterministic helpers are exercised here - no NSE API calls and
no model training. Importing the module pulls in TensorFlow, so the whole file
is marked `slow`.
"""
import numpy as np
import pandas as pd
import pytest

import config

pytestmark = pytest.mark.slow

nse_service = pytest.importorskip(
    "src.backend.services.nse_service",
    reason="backend service dependencies (tensorflow / nselib / talib) unavailable",
)


# ── _historical_ohlc ──────────────────────────────────────────────────────────

def test_historical_ohlc_returns_empty_for_none():
    assert nse_service._historical_ohlc(None) == []


def test_historical_ohlc_returns_ohlcv_records(price_df_factory):
    df = price_df_factory([100.0, 101.0, 102.0])
    out = nse_service._historical_ohlc(df)
    assert len(out) == 3
    assert set(out[0]) == {"date", "high", "low", "close", "open", "volume"}
    assert isinstance(out[0]["date"], str)
    assert isinstance(out[0]["close"], float)


def test_historical_ohlc_includes_volume(price_df_factory):
    df = price_df_factory([100.0, 101.0])
    assert nse_service._historical_ohlc(df)[0]["volume"] == pytest.approx(10000)


def test_historical_ohlc_omits_volume_when_column_missing(price_df_factory):
    df = price_df_factory([100.0, 101.0]).drop(columns=["volume"])
    assert "volume" not in nse_service._historical_ohlc(df)[0]


def test_historical_ohlc_omits_open_when_column_missing(price_df_factory):
    df = price_df_factory([100.0, 101.0]).drop(columns=["open"])
    assert "open" not in nse_service._historical_ohlc(df)[0]


def test_historical_ohlc_respects_the_tail_limit(price_df_factory):
    df = price_df_factory([100.0 + i for i in range(50)])
    assert len(nse_service._historical_ohlc(df, n=10)) == 10


def test_historical_ohlc_keeps_rows_with_missing_open(price_df_factory):
    """A NaN 'open' must never drop an otherwise-chartable row."""
    df = price_df_factory([100.0, 101.0, 102.0])
    df.loc[1, "open"] = np.nan
    out = nse_service._historical_ohlc(df)
    assert len(out) == 3
    assert out[1]["open"] is None


def test_historical_ohlc_keeps_rows_with_missing_volume(price_df_factory):
    df = price_df_factory([100.0, 101.0, 102.0])
    df["volume"] = df["volume"].astype(float)
    df.loc[1, "volume"] = np.nan
    out = nse_service._historical_ohlc(df)
    assert len(out) == 3
    assert out[1]["volume"] is None


def test_historical_ohlc_drops_rows_missing_close(price_df_factory):
    df = price_df_factory([100.0, 101.0, 102.0])
    df.loc[1, "close"] = np.nan
    assert len(nse_service._historical_ohlc(df)) == 2


def test_historical_ohlc_deduplicates_dates(price_df_factory):
    """Duplicate timestamps make lightweight-charts throw and blank the chart."""
    df = price_df_factory([100.0, 101.0, 102.0])
    dup = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    out = nse_service._historical_ohlc(dup)
    dates = [r["date"] for r in out]
    assert len(dates) == len(set(dates))


def test_historical_ohlc_dates_are_ascending(price_df_factory):
    df = price_df_factory([100.0 + i for i in range(20)])
    dates = [r["date"] for r in nse_service._historical_ohlc(df)]
    assert dates == sorted(dates)


def test_historical_ohlc_is_json_serialisable(price_df_factory):
    import json
    json.dumps(nse_service._historical_ohlc(price_df_factory([100.0, 101.0])))


# ── _build_forecast_days ──────────────────────────────────────────────────────

def _pred(n=None):
    n = n or config.FORECAST_DAYS
    # columns: [high, low, close, prev_close]
    return np.array([[102.0 + i, 98.0 + i, 100.0 + i, 99.0 + i] for i in range(n)])


def test_build_forecast_days_length(price_df_factory):
    out = nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10))
    assert len(out) == config.FORECAST_DAYS


def test_build_forecast_days_starts_after_last_historical_date(price_df_factory):
    hist = price_df_factory([100.0] * 10)
    out = nse_service._build_forecast_days(_pred(), None, hist)
    assert out[0]["date"] > hist["date"].max()


def test_build_forecast_days_uses_business_days(price_df_factory):
    out = nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10))
    for day in out:
        assert pd.Timestamp(day["date"]).weekday() < 5


def test_build_forecast_days_dates_are_unique_and_ascending(price_df_factory):
    dates = [d["date"] for d in
             nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10))]
    assert dates == sorted(dates) == sorted(set(dates))


def test_build_forecast_days_maps_prediction_columns(price_df_factory):
    out = nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10))
    assert out[0]["high"] == pytest.approx(102.0)
    assert out[0]["low"] == pytest.approx(98.0)
    assert out[0]["close"] == pytest.approx(100.0)
    assert out[0]["prev_close"] == pytest.approx(99.0)


def test_build_forecast_days_signal_is_none_without_classifier(price_df_factory):
    out = nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10))
    assert all(d["signal"] is None for d in out)


def test_build_forecast_days_attaches_signals(price_df_factory):
    signals = [{"label": "BUY", "confidence": 71.5}] * config.FORECAST_DAYS
    out = nse_service._build_forecast_days(_pred(), signals, price_df_factory([100.0] * 10))
    assert out[0]["signal"]["label"] == "BUY"
    assert out[0]["signal"]["confidence"] == 71.5
    assert out[0]["signal"]["regime_adjusted"] is False


def test_build_forecast_days_tolerates_short_signal_list(price_df_factory):
    signals = [{"label": "SELL", "confidence": 60.0}]
    out = nse_service._build_forecast_days(_pred(), signals, price_df_factory([100.0] * 10))
    assert out[0]["signal"]["label"] == "SELL"
    assert out[-1]["signal"] is None


def test_build_forecast_days_is_json_serialisable(price_df_factory):
    import json
    json.dumps(nse_service._build_forecast_days(_pred(), None, price_df_factory([100.0] * 10)))


# ── _regime_to_dict ───────────────────────────────────────────────────────────

def test_regime_to_dict_passes_known_keys_through():
    out = nse_service._regime_to_dict({
        "regime": "BULL", "recommended_model": "CNN-LSTM", "sufficient_data": True,
        "rows": 1200, "sma_fast": 1.0, "sma_slow": 2.0, "description": "d",
    })
    assert out["regime"] == "BULL"
    assert out["rows"] == 1200


def test_regime_to_dict_supplies_safe_defaults():
    out = nse_service._regime_to_dict({})
    assert out["regime"] == "UNKNOWN"
    assert out["sufficient_data"] is False
    assert out["recommended_model"] == "LSTM"


def test_regime_to_dict_drops_unexpected_keys():
    out = nse_service._regime_to_dict({"regime": "BEAR", "secret": "x"})
    assert "secret" not in out


# ── _get_combined_insight ─────────────────────────────────────────────────────

BULL = {"regime": "BULL", "sufficient_data": True}
BEAR = {"regime": "BEAR", "sufficient_data": True}
FLAT = {"regime": "SIDEWAYS", "sufficient_data": True}

BULLISH_PAT = [("HAMMER", 100, "01-Jan-2024")]
BEARISH_PAT = [("SHOOTING_STAR", -100, "01-Jan-2024")]


def test_insight_none_without_patterns():
    assert nse_service._get_combined_insight(BULL, []) is None


def test_insight_none_without_sufficient_data():
    assert nse_service._get_combined_insight({"regime": "BULL", "sufficient_data": False},
                                             BULLISH_PAT) is None


@pytest.mark.parametrize("regime,pats", [
    (BULL, BULLISH_PAT), (BULL, BEARISH_PAT),
    (BEAR, BULLISH_PAT), (BEAR, BEARISH_PAT),
    (FLAT, BULLISH_PAT), (FLAT, BEARISH_PAT),
])
def test_insight_returned_for_every_regime_pattern_combo(regime, pats):
    assert isinstance(nse_service._get_combined_insight(regime, pats), str)


def test_insight_flags_conflict_in_bear_trend():
    assert "reversal" in nse_service._get_combined_insight(BEAR, BULLISH_PAT).lower()


def test_insight_confirms_aligned_bull_signal():
    assert "confirms" in nse_service._get_combined_insight(BULL, BULLISH_PAT).lower()


# ── _cache_info ───────────────────────────────────────────────────────────────

def test_cache_info_returns_matching_entry(monkeypatch):
    monkeypatch.setattr(nse_service.model_registry, "list_cached", lambda: [
        {"symbol": "SUZLON", "model": "LSTM", "trained_at": "2024-06-01T10:00:00", "n_rows": 1000},
    ])
    assert nse_service._cache_info("lstm", "SUZLON")["n_rows"] == 1000


def test_cache_info_empty_when_not_cached(monkeypatch):
    monkeypatch.setattr(nse_service.model_registry, "list_cached", lambda: [])
    assert nse_service._cache_info("lstm", "SUZLON") == {}


def test_cache_info_does_not_match_a_different_model(monkeypatch):
    monkeypatch.setattr(nse_service.model_registry, "list_cached", lambda: [
        {"symbol": "SUZLON", "model": "GRU", "trained_at": "x", "n_rows": 5},
    ])
    assert nse_service._cache_info("lstm", "SUZLON") == {}


# ── constants / wiring ────────────────────────────────────────────────────────

def test_every_algorithm_has_a_classifier_and_labels():
    for algo in nse_service.ALGO_FUNCS:
        assert algo in nse_service.CLASSIFIER_FUNCS
        assert algo in nse_service.ALGO_DISPLAY
        assert algo in nse_service.REGISTRY_NAMES


def test_cache_status_labels_cover_every_status():
    assert set(nse_service.CACHE_STATUS_LABEL) == {"fresh", "warm", "miss"}


# ── fetch_data: race-safety contract ──────────────────────────────────────────

def _stub_nse(monkeypatch, raw_df, symbol="SUZLON"):
    from nselib import capital_market
    monkeypatch.setattr(capital_market, "equity_list", lambda: pd.DataFrame({
        "SYMBOL": [symbol],
        "NAME OF COMPANY": ["Test Company Ltd"],
        " DATE OF LISTING": ["01-Jan-1996"],
    }))
    monkeypatch.setattr(nse_service, "_load_or_fetch", lambda *a, **k: raw_df.copy())


def test_fetch_data_returns_frame_details_and_hist(monkeypatch, raw_nse_df):
    _stub_nse(monkeypatch, raw_nse_df)
    df, details, hist_df = nse_service.fetch_data("SUZLON")
    assert details["scheme_name"] == "Test Company Ltd"
    assert details["scheme_code"] == "SUZLON"
    assert "close" in df.columns
    assert "open" in hist_df.columns and "volume" in hist_df.columns


def test_fetch_data_raises_for_unknown_symbol(monkeypatch, raw_nse_df):
    _stub_nse(monkeypatch, raw_nse_df, symbol="SUZLON")
    with pytest.raises(ValueError, match="not found"):
        nse_service.fetch_data("NOPE")


def test_fetch_data_snapshot_survives_a_concurrent_job(monkeypatch, raw_nse_df):
    """
    config.HISTORIC_DATA is a shared global. The snapshot returned by
    fetch_data() must stay valid even after another job overwrites the global -
    this is what stopped the candlestick chart rendering another symbol's data.
    """
    _stub_nse(monkeypatch, raw_nse_df)
    _, _, hist_df = nse_service.fetch_data("SUZLON")
    before = hist_df["close"].tolist()

    config.HISTORIC_DATA = pd.DataFrame({"close": [0.0]})   # simulate another job

    assert hist_df["close"].tolist() == before
    assert nse_service._historical_ohlc(hist_df)


# ── search_symbols ────────────────────────────────────────────────────────────

@pytest.fixture
def stub_equity_list(monkeypatch):
    from nselib import capital_market
    monkeypatch.setattr(capital_market, "equity_list", lambda: pd.DataFrame({
        "SYMBOL": ["SUZLON", "NTPC", "SBIN"],
        "NAME OF COMPANY": ["Suzlon Energy Limited", "NTPC Limited",
                            "State Bank of India"],
    }))


def test_search_symbols_matches_ticker(stub_equity_list):
    assert [r["symbol"] for r in nse_service.search_symbols("SUZ")] == ["SUZLON"]


def test_search_symbols_matches_company_name(stub_equity_list):
    assert [r["symbol"] for r in nse_service.search_symbols("ntpc")] == ["NTPC"]


def test_search_symbols_is_case_insensitive(stub_equity_list):
    assert nse_service.search_symbols("suzlon") == nse_service.search_symbols("SUZLON")


def test_search_symbols_respects_limit(stub_equity_list):
    assert len(nse_service.search_symbols("a", limit=1)) <= 1


def test_search_symbols_returns_empty_on_no_match(stub_equity_list):
    assert nse_service.search_symbols("ZZZZZ") == []

