"""
routers/data.py
━━━━━━━━━━━━━━━
GET /api/symbols?q=       — search NSE symbols
GET /api/historical/{sym} — historical OHLC for charting
"""
from fastapi import APIRouter, HTTPException, Query
from src.backend.services.nse_service import search_symbols, fetch_data
import config

router = APIRouter()


@router.get("/symbols")
def symbols(q: str = Query(default="", min_length=1)):
    try:
        return {"symbols": search_symbols(q)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/historical/{symbol}")
def historical(symbol: str, days: int = 120):
    sym = symbol.upper().strip()
    try:
        fetch_data(sym)          # populates config.HISTORIC_DATA
        hist_df = config.HISTORIC_DATA
        if hist_df is None:
            raise HTTPException(status_code=404, detail="No data")

        cols  = [c for c in ["date", "open", "high", "low", "close"] if c in hist_df.columns]
        tail  = hist_df[cols].tail(days).dropna()
        records = []
        for _, row in tail.iterrows():
            rec = {k: float(row[k]) if k != "date" else str(row[k]) for k in cols}
            records.append(rec)

        return {"symbol": sym, "data": records}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

