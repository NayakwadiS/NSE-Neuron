"""
Tests for utils/model_registry.py - the per-symbol weight cache.

TensorFlow is stubbed out so these run in milliseconds; what we care about is
the *decision* logic (fresh / warm / miss) and the on-disk contract, not Keras.
"""
import json
import os
import sys
import types
from datetime import datetime, timedelta

import pytest

import config
from utils import model_registry as mr


# ── Test doubles ──────────────────────────────────────────────────────────────

class FakeModel:
    """Stands in for a compiled Keras model."""

    def __init__(self, tag="model"):
        self.tag = tag
        self.fit_calls = []

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.tag)

    def fit(self, X, y, **kwargs):
        self.fit_calls.append({"n": len(X), **kwargs})


class FakeScaler:
    def __init__(self, tag="scaler"):
        self.tag = tag


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point the registry at a throwaway directory."""
    monkeypatch.setattr(mr, "REGISTRY_DIR", str(tmp_path / "saved_models"))
    monkeypatch.setattr(config, "ENABLE_MODEL_CACHE", True)
    return mr


@pytest.fixture
def stub_tensorflow(monkeypatch):
    """Install a fake tensorflow.keras.models.load_model that returns a marker."""
    loaded = FakeModel("loaded-from-disk")

    models_mod = types.ModuleType("tensorflow.keras.models")
    models_mod.load_model = lambda path: loaded

    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.models = models_mod

    tf_mod = types.ModuleType("tensorflow")
    tf_mod.keras = keras_mod

    monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models_mod)
    return loaded


def make_sig(n_rows=1000, last_date="2024-06-01", **extra):
    sig = {
        "model":       "LSTM",
        "features":    list(config.FEATURE_COLUMNS),
        "split":       config.TRAIN_TEST_SPLIT,
        "lib_version": "1",
        "n_rows":      n_rows,
        "last_date":   last_date,
        "time_step":   10,
        "units":       64,
        "n_features":  4,
        "arch":        "stacked-lstm-3x",
    }
    sig.update(extra)
    return sig


def seed_cache(registry, symbol="SUZLON", model="LSTM", sig=None, trained_at=None):
    """Write a complete artifact set to disk and return its signature."""
    sig = sig or make_sig()
    registry.save(symbol, model, FakeModel(), FakeScaler(), sig, {"rmse": 0.01})
    if trained_at is not None:
        _, _, _, jpath = registry.get_paths(symbol, model)
        with open(jpath, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["trained_at"] = trained_at.isoformat(timespec="seconds")
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    return sig


# ── path helpers ──────────────────────────────────────────────────────────────

def test_safe_sanitises_path_separators():
    assert "/" not in mr._safe("a/b")
    assert "\\" not in mr._safe("a\\b")
    assert mr._safe("CNN-LSTM") == "CNN-LSTM"


def test_get_paths_uppercases_symbol_and_returns_four_paths(registry):
    d, m, s, j = registry.get_paths("SUZLON", "LSTM")
    assert d.endswith(os.path.join("SUZLON", "LSTM"))
    assert m.endswith("model.keras")
    assert s.endswith("scaler.pkl")
    assert j.endswith("meta.json")


# ── signature ─────────────────────────────────────────────────────────────────

def test_build_signature_contains_core_keys(bull_df):
    sig = mr.build_signature("LSTM", bull_df, {"units": 64})
    assert sig["model"] == "LSTM"
    assert sig["features"] == list(config.FEATURE_COLUMNS)
    assert sig["n_rows"] == len(bull_df)
    assert sig["units"] == 64
    assert sig["last_date"]


def test_signature_hash_ignores_row_count_and_last_date():
    a = mr.signature_hash(make_sig(n_rows=1000, last_date="2024-01-01"))
    b = mr.signature_hash(make_sig(n_rows=9999, last_date="2025-12-31"))
    assert a == b


def test_signature_hash_changes_with_architecture():
    assert mr.signature_hash(make_sig(units=64)) != mr.signature_hash(make_sig(units=128))


def test_signature_hash_changes_with_cache_version():
    assert mr.signature_hash(make_sig(lib_version="1")) != mr.signature_hash(make_sig(lib_version="2"))


def test_signature_hash_is_key_order_independent():
    sig = make_sig()
    shuffled = dict(reversed(list(sig.items())))
    assert mr.signature_hash(sig) == mr.signature_hash(shuffled)


# ── save ──────────────────────────────────────────────────────────────────────

def test_save_writes_all_three_artifacts(registry):
    sig = seed_cache(registry)
    d, m, s, j = registry.get_paths("SUZLON", "LSTM")
    assert os.path.exists(m) and os.path.exists(s) and os.path.exists(j)
    with open(j, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["symbol"] == "SUZLON"
    assert meta["signature"]["n_rows"] == sig["n_rows"]
    assert meta["metrics"]["rmse"] == 0.01


def test_save_leaves_no_temp_files_behind(registry):
    seed_cache(registry)
    d, *_ = registry.get_paths("SUZLON", "LSTM")
    assert not [f for f in os.listdir(d) if ".tmp" in f]


def test_save_is_a_noop_when_cache_disabled(registry, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_MODEL_CACHE", False)
    registry.save("SUZLON", "LSTM", FakeModel(), FakeScaler(), make_sig())
    assert not os.path.exists(registry.get_paths("SUZLON", "LSTM")[1])


def test_save_is_a_noop_without_a_symbol(registry):
    registry.save(None, "LSTM", FakeModel(), FakeScaler(), make_sig())
    assert not os.path.isdir(registry.REGISTRY_DIR)


# ── load: miss paths (no TensorFlow needed) ───────────────────────────────────

def test_load_miss_when_nothing_cached(registry):
    model, scaler, status, meta = registry.load("SUZLON", "LSTM", make_sig())
    assert (model, scaler, status) == (None, None, "miss")


def test_load_miss_on_force_retrain(registry, stub_tensorflow):
    sig = seed_cache(registry)
    _, _, status, _ = registry.load("SUZLON", "LSTM", sig, force_retrain=True)
    assert status == "miss"


def test_load_miss_when_cache_disabled(registry, monkeypatch, stub_tensorflow):
    sig = seed_cache(registry)
    monkeypatch.setattr(config, "ENABLE_MODEL_CACHE", False)
    _, _, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "miss"


def test_load_miss_when_symbol_is_empty(registry):
    _, _, status, _ = registry.load("", "LSTM", make_sig())
    assert status == "miss"


def test_load_miss_on_architecture_change(registry):
    seed_cache(registry, sig=make_sig(units=64))
    _, _, status, _ = registry.load("SUZLON", "LSTM", make_sig(units=128))
    assert status == "miss"


def test_load_miss_on_corrupt_metadata(registry):
    sig = seed_cache(registry)
    _, _, _, jpath = registry.get_paths("SUZLON", "LSTM")
    with open(jpath, "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    _, _, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "miss"


def test_load_miss_when_artifact_file_is_deleted(registry):
    sig = seed_cache(registry)
    os.remove(registry.get_paths("SUZLON", "LSTM")[2])   # drop the scaler
    _, _, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "miss"


def test_load_miss_past_hard_expiry(registry):
    old = datetime.now() - timedelta(days=config.CACHE_MAX_AGE_DAYS + 5)
    sig = seed_cache(registry, trained_at=old)
    _, _, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "miss"


def test_load_miss_when_too_many_new_bars(registry):
    sig = seed_cache(registry, sig=make_sig(n_rows=1000))
    newer = make_sig(n_rows=1000 + config.CACHE_MAX_STALE_DAYS + 1)
    _, _, status, _ = registry.load("SUZLON", "LSTM", newer)
    assert status == "miss"


def test_load_miss_when_dataset_shrank(registry):
    seed_cache(registry, sig=make_sig(n_rows=1000))
    _, _, status, _ = registry.load("SUZLON", "LSTM", make_sig(n_rows=900))
    assert status == "miss"


# ── load: hit paths (stubbed TensorFlow) ──────────────────────────────────────

def test_load_fresh_when_no_new_bars(registry, stub_tensorflow):
    sig = seed_cache(registry, sig=make_sig(n_rows=1000))
    model, scaler, status, meta = registry.load("SUZLON", "LSTM", make_sig(n_rows=1000))
    assert status == "fresh"
    assert model is stub_tensorflow
    assert isinstance(scaler, FakeScaler)
    assert meta["symbol"] == "SUZLON"


def test_load_warm_within_stale_window(registry, stub_tensorflow):
    seed_cache(registry, sig=make_sig(n_rows=1000))
    newer = make_sig(n_rows=1000 + config.CACHE_MAX_STALE_DAYS)
    _, _, status, _ = registry.load("SUZLON", "LSTM", newer)
    assert status == "warm"


def test_load_returns_the_exact_persisted_scaler(registry, stub_tensorflow):
    """A cached model MUST come back with its own scaler or predictions corrupt."""
    sig = make_sig(n_rows=1000)
    registry.save("SUZLON", "LSTM", FakeModel(), FakeScaler("scaler-v7"), sig)
    _, scaler, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "fresh"
    assert scaler.tag == "scaler-v7"


def test_load_miss_when_keras_file_is_unreadable(registry, monkeypatch):
    sig = seed_cache(registry)
    models_mod = types.ModuleType("tensorflow.keras.models")

    def boom(path):
        raise OSError("corrupt file")

    models_mod.load_model = boom
    keras_mod = types.ModuleType("tensorflow.keras"); keras_mod.models = models_mod
    tf_mod = types.ModuleType("tensorflow"); tf_mod.keras = keras_mod
    monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models_mod)

    _, _, status, _ = registry.load("SUZLON", "LSTM", sig)
    assert status == "miss"


# ── warm_start ────────────────────────────────────────────────────────────────

def test_warm_start_uses_only_the_newest_samples(registry, monkeypatch):
    monkeypatch.setattr(config, "WARM_START_SAMPLES", 25)
    monkeypatch.setattr(config, "WARM_START_EPOCHS", 3)
    model = FakeModel()
    X = list(range(200))
    y = list(range(200))
    mr.warm_start(model, X, y, batch_size=8)
    call = model.fit_calls[0]
    assert call["n"] == 25
    assert call["epochs"] == 3
    assert call["batch_size"] == 8


def test_warm_start_survives_a_model_without_an_optimizer(registry):
    model = FakeModel()
    assert mr.warm_start(model, list(range(50)), list(range(50))) is model


# ── introspection / management ────────────────────────────────────────────────

def test_list_cached_is_empty_for_a_fresh_registry(registry):
    assert registry.list_cached() == []


def test_list_cached_reports_each_artifact(registry):
    seed_cache(registry, "SUZLON", "LSTM")
    seed_cache(registry, "SBIN", "GRU")
    entries = registry.list_cached()
    assert {(e["symbol"], e["model"]) for e in entries} == {("SUZLON", "LSTM"), ("SBIN", "GRU")}
    assert all(e["size_kb"] >= 0 for e in entries)
    assert all(e["trained_at"] for e in entries)


def test_list_cached_is_newest_first(registry):
    seed_cache(registry, "OLD", "LSTM", trained_at=datetime.now() - timedelta(days=5))
    seed_cache(registry, "NEW", "LSTM", trained_at=datetime.now())
    assert registry.list_cached()[0]["symbol"] == "NEW"


def test_delete_single_model(registry):
    seed_cache(registry, "SUZLON", "LSTM")
    seed_cache(registry, "SUZLON", "GRU")
    assert registry.delete("SUZLON", "LSTM") == 1
    assert [e["model"] for e in registry.list_cached()] == ["GRU"]


def test_delete_all_models_for_symbol(registry):
    seed_cache(registry, "SUZLON", "LSTM")
    seed_cache(registry, "SUZLON", "GRU")
    assert registry.delete("SUZLON") == 2
    assert registry.list_cached() == []


def test_delete_unknown_symbol_returns_zero(registry):
    assert registry.delete("NOPE") == 0


def test_delete_unknown_model_returns_zero(registry):
    seed_cache(registry, "SUZLON", "LSTM")
    assert registry.delete("SUZLON", "GRU") == 0


def test_delete_is_case_insensitive_on_symbol(registry):
    seed_cache(registry, "SUZLON", "LSTM")
    assert registry.delete("SUZLON") == 1


def test_purge_expired_removes_only_old_artifacts(registry):
    seed_cache(registry, "OLD", "LSTM",
               trained_at=datetime.now() - timedelta(days=config.CACHE_MAX_AGE_DAYS + 1))
    seed_cache(registry, "NEW", "LSTM", trained_at=datetime.now())
    assert registry.purge_expired() == 1
    assert [e["symbol"] for e in registry.list_cached()] == ["NEW"]


def test_purge_expired_on_empty_registry(registry):
    assert registry.purge_expired() == 0

