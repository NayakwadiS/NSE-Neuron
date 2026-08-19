"""
utils/model_registry.py
━━━━━━━━━━━━━━━━━━━━━━━
Persistent per-symbol / per-model weight cache.

Every trained Keras model is stored together with the *exact* scaler that was
fitted alongside it plus a signature describing the architecture and the data
window it was trained on.  On the next run the registry decides whether the
artifact can be re-used as-is ("fresh"), fine-tuned on the newest bars
("warm") or must be discarded and rebuilt ("miss").

Disk layout
-----------
    saved_models/
        INFY/
            LSTM/
                model.keras
                scaler.pkl
                meta.json
            CNN-LSTM/
                ...
        SBIN/
            GRU/ ...

Status semantics
----------------
    'miss'  → nothing usable on disk; caller must build + fully train
    'fresh' → cached model is up to date; caller may skip training entirely
    'warm'  → cached model is a few bars behind; caller should fine-tune
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import threading
from datetime import datetime

import config

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(_PROJECT_ROOT, getattr(config, "MODEL_CACHE_DIR", "saved_models"))

# One lock per symbol/model key so concurrent backend jobs never interleave writes
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def _safe(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def get_paths(symbol: str, model_name: str):
    d = os.path.join(REGISTRY_DIR, _safe(symbol).upper(), _safe(model_name))
    return (
        d,
        os.path.join(d, "model.keras"),
        os.path.join(d, "scaler.pkl"),
        os.path.join(d, "meta.json"),
    )


# ── Signature ─────────────────────────────────────────────────────────────────

def _last_date(df) -> str:
    """Best-effort extraction of the most recent date in the dataframe."""
    for col in ("Date", "date"):
        if col in getattr(df, "columns", []):
            try:
                return str(df[col].max())
            except Exception:
                pass
    return ""


def build_signature(model_name: str, df, extra: dict | None = None) -> dict:
    """
    Describe everything that must match for a cached artifact to be reusable.

    `extra` lets each model contribute its own architecture-defining values
    (time_step, units, n_features, sub-sequence layout, ...).
    """
    sig = {
        "model":       model_name,
        "features":    list(config.FEATURE_COLUMNS),
        "split":       config.TRAIN_TEST_SPLIT,
        "lib_version": getattr(config, "MODEL_CACHE_VERSION", "1"),
        # data-window keys (not architecture — used for staleness, not identity)
        "n_rows":      int(len(df)),
        "last_date":   _last_date(df),
    }
    if extra:
        sig.update(extra)
    return sig


# Keys that define *identity* — any mismatch invalidates the cache outright.
# `n_rows` / `last_date` are deliberately excluded: they drive the staleness
# tiering instead.
_ARCH_KEYS_EXCLUDED = {"n_rows", "last_date"}


def signature_hash(sig: dict) -> str:
    arch = {k: v for k, v in sig.items() if k not in _ARCH_KEYS_EXCLUDED}
    return hashlib.md5(json.dumps(arch, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ── Load ──────────────────────────────────────────────────────────────────────

def load(symbol: str, model_name: str, sig: dict, force_retrain: bool = False):
    """
    Try to load a cached (model, scaler) pair.

    Returns
    -------
    (keras_model | None, scaler | None, status, meta)
        status ∈ {'miss', 'fresh', 'warm'}
    """
    if force_retrain or not getattr(config, "ENABLE_MODEL_CACHE", True) or not symbol:
        return None, None, "miss", {}

    d, mpath, spath, jpath = get_paths(symbol, model_name)
    if not (os.path.exists(mpath) and os.path.exists(spath) and os.path.exists(jpath)):
        return None, None, "miss", {}

    try:
        with open(jpath, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None, None, "miss", {}

    old = meta.get("signature", {})

    # 1. Architecture / preprocessing must match exactly
    if signature_hash(old) != signature_hash(sig):
        print(f"  [Cache] {symbol}/{model_name}: signature changed -> full retrain")
        return None, None, "miss", {}

    # 2. Hard expiry
    try:
        age_days = (datetime.now() - datetime.fromisoformat(meta["trained_at"])).days
    except Exception:
        age_days = 10 ** 6
    if age_days > getattr(config, "CACHE_MAX_AGE_DAYS", 30):
        print(f"  [Cache] {symbol}/{model_name}: {age_days} days old -> full retrain")
        return None, None, "miss", {}

    # 3. Staleness tiering on newly appended bars
    new_bars = int(sig.get("n_rows", 0)) - int(old.get("n_rows", 0))
    if new_bars < 0:
        # dataset shrank — something changed upstream, don't trust the cache
        return None, None, "miss", {}
    if new_bars > getattr(config, "CACHE_MAX_STALE_DAYS", 10):
        print(f"  [Cache] {symbol}/{model_name}: {new_bars} new bars -> full retrain")
        return None, None, "miss", {}

    # 4. Load artifacts
    try:
        from tensorflow.keras.models import load_model as _load_model
        model = _load_model(mpath)
        with open(spath, "rb") as fh:
            scaler = pickle.load(fh)
    except Exception as exc:  # corrupted / incompatible artifact
        print(f"  [Cache] {symbol}/{model_name}: failed to load ({exc}) -> full retrain")
        return None, None, "miss", {}

    status = "fresh" if new_bars == 0 else "warm"
    print(f"  [Cache] {symbol}/{model_name}: HIT ({status}, {new_bars} new bars, "
          f"trained {age_days}d ago)")
    return model, scaler, status, meta


# ── Save ──────────────────────────────────────────────────────────────────────

def save(symbol: str, model_name: str, model, scaler, sig: dict, metrics: dict | None = None):
    """Atomically persist model + scaler + metadata."""
    if not getattr(config, "ENABLE_MODEL_CACHE", True) or not symbol:
        return

    key = f"{symbol}/{model_name}"
    with _lock_for(key):
        d, mpath, spath, jpath = get_paths(symbol, model_name)
        os.makedirs(d, exist_ok=True)
        tmp_m = mpath + ".tmp.keras"
        tmp_s = spath + ".tmp"
        tmp_j = jpath + ".tmp"
        try:
            model.save(tmp_m)
            with open(tmp_s, "wb") as fh:
                pickle.dump(scaler, fh)
            with open(tmp_j, "w", encoding="utf-8") as fh:
                json.dump({
                    "symbol":     str(symbol).upper(),
                    "model":      model_name,
                    "signature":  sig,
                    "metrics":    metrics or {},
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                }, fh, indent=2, default=str)

            os.replace(tmp_m, mpath)
            os.replace(tmp_s, spath)
            os.replace(tmp_j, jpath)
        except Exception as exc:
            print(f"  [Cache] WARNING: could not save {symbol}/{model_name}: {exc}")
            for tmp in (tmp_m, tmp_s, tmp_j):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            return

    # Printed outside the try so a console-encoding hiccup can never be
    # mistaken for a failed save (keep this message pure ASCII).
    print(f"  [Cache] Saved {symbol}/{model_name} -> {d}")


# ── Warm-start helper ─────────────────────────────────────────────────────────

def warm_start(model, X, y, **fit_kwargs):
    """
    Fine-tune an already-trained model on the newest windows at a much lower
    learning rate so previously learned structure isn't destroyed.
    """
    lr = getattr(config, "WARM_START_LR", 1e-4)
    try:
        model.optimizer.learning_rate.assign(lr)
    except Exception:
        try:
            import tensorflow as tf
            model.optimizer.learning_rate = tf.Variable(lr)
        except Exception:
            pass  # keep the original LR rather than failing the run

    n = getattr(config, "WARM_START_SAMPLES", 250)
    Xw, yw = X[-n:], y[-n:]
    model.fit(
        Xw, yw,
        epochs=getattr(config, "WARM_START_EPOCHS", 15),
        verbose=1,
        **fit_kwargs,
    )
    return model


# ── Introspection / management ────────────────────────────────────────────────

def list_cached() -> list[dict]:
    """Return metadata for every cached artifact, newest first."""
    out: list[dict] = []
    if not os.path.isdir(REGISTRY_DIR):
        return out

    for sym in sorted(os.listdir(REGISTRY_DIR)):
        sym_dir = os.path.join(REGISTRY_DIR, sym)
        if not os.path.isdir(sym_dir):
            continue
        for mdl in sorted(os.listdir(sym_dir)):
            _, mpath, _, jpath = get_paths(sym, mdl)
            if not os.path.exists(jpath):
                continue
            try:
                with open(jpath, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                continue
            size = os.path.getsize(mpath) if os.path.exists(mpath) else 0
            out.append({
                "symbol":     meta.get("symbol", sym),
                "model":      meta.get("model", mdl),
                "trained_at": meta.get("trained_at"),
                "n_rows":     meta.get("signature", {}).get("n_rows"),
                "last_date":  meta.get("signature", {}).get("last_date"),
                "metrics":    meta.get("metrics", {}),
                "size_kb":    round(size / 1024, 1),
            })
    out.sort(key=lambda r: r.get("trained_at") or "", reverse=True)
    return out


def delete(symbol: str, model_name: str | None = None) -> int:
    """Delete one cached model, or every model for a symbol. Returns count removed."""
    base = os.path.join(REGISTRY_DIR, _safe(symbol).upper())
    if not os.path.isdir(base):
        return 0

    if model_name:
        target = os.path.join(base, _safe(model_name))
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            if not os.listdir(base):
                shutil.rmtree(base, ignore_errors=True)
            return 1
        return 0

    count = len([x for x in os.listdir(base) if os.path.isdir(os.path.join(base, x))])
    shutil.rmtree(base, ignore_errors=True)
    return count


def purge_expired() -> int:
    """Remove artifacts older than CACHE_MAX_AGE_DAYS. Returns count removed."""
    max_age = getattr(config, "CACHE_MAX_AGE_DAYS", 30)
    removed = 0
    for entry in list_cached():
        try:
            age = (datetime.now() - datetime.fromisoformat(entry["trained_at"])).days
        except Exception:
            age = 10 ** 6
        if age > max_age:
            removed += delete(entry["symbol"], entry["model"])
    return removed



