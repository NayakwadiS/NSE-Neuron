"""Tests for utils/preprocessor.py - the data-cleaning pipeline."""
import numpy as np
import pandas as pd
import pytest

import config
from utils import preprocessor as pp


# ── select_columns ────────────────────────────────────────────────────────────

def test_select_columns_keeps_required_and_optional(raw_nse_df):
    out = pp.select_columns(raw_nse_df)
    for col in pp.REQUIRED_RAW_COLUMNS:
        assert col in out.columns
    assert "TotalTradedQuantity" in out.columns
    assert "DeliverableQty" in out.columns


def test_select_columns_tolerates_missing_optional(raw_nse_df_no_volume):
    out = pp.select_columns(raw_nse_df_no_volume)
    assert list(out.columns) == pp.REQUIRED_RAW_COLUMNS


def test_select_columns_raises_on_missing_required(raw_nse_df):
    broken = raw_nse_df.drop(columns=["ClosePrice"])
    with pytest.raises(KeyError, match="ClosePrice"):
        pp.select_columns(broken)


def test_select_columns_ignores_unknown_extra_columns(raw_nse_df):
    raw_nse_df["Series"] = "EQ"
    out = pp.select_columns(raw_nse_df)
    assert "Series" not in out.columns


# ── rename_columns ────────────────────────────────────────────────────────────

def test_rename_columns_maps_all_known_names(raw_nse_df):
    out = pp.rename_columns(raw_nse_df)
    for standard in ("close", "prev_close", "high", "low", "open",
                     "volume", "deliverable_qty"):
        assert standard in out.columns


def test_rename_columns_leaves_date_untouched(raw_nse_df):
    assert "Date" in pp.rename_columns(raw_nse_df).columns


# ── numeric conversion ────────────────────────────────────────────────────────

def test_convert_price_columns_strips_commas(raw_nse_df):
    df = pp.convert_price_columns(pp.rename_columns(raw_nse_df))
    assert df["close"].dtype.kind == "f"
    assert df["close"].iloc[0] == pytest.approx(1010.50)


def test_convert_price_columns_coerces_dashes_to_nan(raw_nse_df):
    df = pp.rename_columns(raw_nse_df)
    df.loc[0, "high"] = "-"
    out = pp.convert_price_columns(df)
    assert np.isnan(out["high"].iloc[0])


def test_convert_price_columns_does_not_mutate_input(raw_nse_df):
    df = pp.rename_columns(raw_nse_df)
    before = df["close"].tolist()
    pp.convert_price_columns(df)
    assert df["close"].tolist() == before


def test_convert_volume_columns_strips_indian_grouping(raw_nse_df):
    df = pp.convert_volume_columns(pp.rename_columns(raw_nse_df))
    # "1,20,000" (Indian grouping) must become 120000
    assert df["volume"].iloc[0] == pytest.approx(120000)
    assert df["volume"].dtype.kind in "if"


def test_convert_volume_columns_handles_missing_column(raw_nse_df_no_volume):
    df = pp.rename_columns(raw_nse_df_no_volume)
    out = pp.convert_volume_columns(df)      # must not raise
    assert "volume" not in out.columns


def test_convert_volume_columns_coerces_dash_to_nan(raw_nse_df):
    df = pp.rename_columns(raw_nse_df)
    df.loc[0, "volume"] = "-"
    out = pp.convert_volume_columns(df)
    assert np.isnan(out["volume"].iloc[0])


# ── parse_and_sort_dates ──────────────────────────────────────────────────────

def _prepared(raw):
    return pp.convert_volume_columns(pp.convert_price_columns(pp.rename_columns(pp.select_columns(raw))))


def test_parse_and_sort_dates_sorts_ascending(raw_nse_df):
    out = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    assert out["Date"].is_monotonic_increasing


def test_parse_and_sort_dates_drops_duplicate_dates(raw_nse_df):
    out = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    assert out["Date"].duplicated().sum() == 0
    assert len(out) == 3          # 4 raw rows, one duplicate collapsed


def test_parse_and_sort_dates_keeps_last_duplicate(raw_nse_df):
    """The most recently fetched row for a day is the authoritative one."""
    out = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    jan2 = out[out["date"] == "2024-01-02"].iloc[0]
    assert jan2["close"] == pytest.approx(1006.00)   # not 1005.25


def test_parse_and_sort_dates_adds_formatted_date_string(raw_nse_df):
    out = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    assert out["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_parse_and_sort_dates_drops_unparseable_dates(raw_nse_df):
    df = _prepared(raw_nse_df)
    df.loc[0, "Date"] = "not-a-date"
    out = pp.parse_and_sort_dates(df)
    assert len(out) == 2
    assert out["Date"].notna().all()


def test_parse_and_sort_dates_does_not_leak_index_column(raw_nse_df):
    """A stray 'index' column used to leak in from reset_index()."""
    out = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    assert "index" not in out.columns
    assert out.index.tolist() == list(range(len(out)))


def test_parse_and_sort_dates_does_not_mutate_input(raw_nse_df):
    df = _prepared(raw_nse_df)
    n_before = len(df)
    pp.parse_and_sort_dates(df)
    assert len(df) == n_before


# ── column stripping ──────────────────────────────────────────────────────────

def test_remove_non_feature_columns_drops_open_and_volume(raw_nse_df):
    df = pp.parse_and_sort_dates(_prepared(raw_nse_df))
    out = pp.remove_non_feature_columns(df)
    for col in ("open", "volume", "deliverable_qty"):
        assert col not in out.columns
    for col in ("close", "high", "low", "prev_close", "Date", "date"):
        assert col in out.columns


def test_remove_non_feature_columns_is_a_noop_when_absent():
    df = pd.DataFrame({"close": [1.0], "high": [1.0]})
    assert list(pp.remove_non_feature_columns(df).columns) == ["close", "high"]


def test_removed_open_coulmn_backward_compatible():
    df = pd.DataFrame({"open": [1.0], "close": [1.0], "volume": [10]})
    out = pp.removed_open_coulmn(df)
    assert "open" not in out.columns
    assert "volume" in out.columns      # legacy helper only drops 'open'


# ── full pipeline ─────────────────────────────────────────────────────────────

def test_preprocess_returns_model_ready_frame(raw_nse_df):
    out = pp.preprocess_nse_df(raw_nse_df)
    for col in config.FEATURE_COLUMNS:
        assert col in out.columns
    assert "open" not in out.columns
    assert "volume" not in out.columns


def test_preprocess_publishes_historic_data_with_open_and_volume(raw_nse_df):
    pp.preprocess_nse_df(raw_nse_df)
    hist = config.HISTORIC_DATA
    assert hist is not None
    assert "open" in hist.columns
    assert "volume" in hist.columns
    assert "date" in hist.columns


def test_preprocess_historic_data_is_a_detached_copy(raw_nse_df):
    """Downstream mutation of the returned frame must not corrupt the global."""
    out = pp.preprocess_nse_df(raw_nse_df)
    hist = config.HISTORIC_DATA
    out.loc[0, "close"] = -999.0
    assert hist["close"].iloc[0] != -999.0


def test_preprocess_assigns_a_new_object_each_call(raw_nse_df):
    """Race-safety: snapshotting the global must be enough to isolate a job."""
    pp.preprocess_nse_df(raw_nse_df)
    first = config.HISTORIC_DATA
    pp.preprocess_nse_df(raw_nse_df)
    assert config.HISTORIC_DATA is not first


def test_preprocess_historic_data_has_unique_ascending_dates(raw_nse_df):
    """lightweight-charts blanks out entirely on duplicate timestamps."""
    pp.preprocess_nse_df(raw_nse_df)
    hist = config.HISTORIC_DATA
    assert hist["date"].duplicated().sum() == 0
    assert hist["Date"].is_monotonic_increasing


def test_preprocess_works_without_volume_columns(raw_nse_df_no_volume):
    out = pp.preprocess_nse_df(raw_nse_df_no_volume)
    assert "volume" not in config.HISTORIC_DATA.columns
    assert "close" in out.columns


def test_preprocess_volume_is_numeric(raw_nse_df):
    pp.preprocess_nse_df(raw_nse_df)
    vols = config.HISTORIC_DATA["volume"]
    assert vols.dtype.kind in "if"
    assert (vols.dropna() > 0).all()



