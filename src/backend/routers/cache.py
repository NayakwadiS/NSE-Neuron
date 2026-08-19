"""
routers/cache.py
━━━━━━━━━━━━━━━━
Management endpoints for the per-symbol model weight cache.

GET    /api/models/cached                    — list every cached artifact
GET    /api/models/cached/{symbol}           — list cached artifacts for one symbol
DELETE /api/models/cached/{symbol}           — drop every cached model for a symbol
DELETE /api/models/cached/{symbol}/{model}   — drop one cached model
POST   /api/models/cached/purge              — remove expired artifacts
"""
from fastapi import APIRouter, HTTPException

from utils import model_registry
import config

router = APIRouter()


@router.get("/models/cached")
def list_cached():
    entries = model_registry.list_cached()
    return {
        "enabled":       config.ENABLE_MODEL_CACHE,
        "max_stale_days": config.CACHE_MAX_STALE_DAYS,
        "max_age_days":   config.CACHE_MAX_AGE_DAYS,
        "count":         len(entries),
        "total_size_kb": round(sum(e["size_kb"] for e in entries), 1),
        "models":        entries,
    }


@router.get("/models/cached/{symbol}")
def list_cached_for_symbol(symbol: str):
    sym = symbol.upper().strip()
    entries = [e for e in model_registry.list_cached() if e["symbol"] == sym]
    return {"symbol": sym, "count": len(entries), "models": entries}


@router.delete("/models/cached/{symbol}")
def delete_symbol(symbol: str):
    removed = model_registry.delete(symbol)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No cached models for {symbol}")
    return {"deleted": removed, "symbol": symbol.upper()}


@router.delete("/models/cached/{symbol}/{model_name}")
def delete_one(symbol: str, model_name: str):
    removed = model_registry.delete(symbol, model_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"{symbol}/{model_name} not cached")
    return {"deleted": removed, "symbol": symbol.upper(), "model": model_name}


@router.post("/models/cached/purge")
def purge():
    removed = model_registry.purge_expired()
    return {"purged": removed, "max_age_days": config.CACHE_MAX_AGE_DAYS}

