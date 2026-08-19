"""
routers/data.py
━━━━━━━━━━━━━━━
GET /api/symbols?q=       — search NSE symbols
GET /api/historical/{sym} — historical OHLC for charting
"""
from fastapi import APIRouter, HTTPException, Query
from src.backend.services.nse_service import search_symbols, fetch_data

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
        # Snapshot the dataframe returned directly from fetch_data instead of
        # re-reading config.HISTORIC_DATA — the latter is a shared global that
        # a concurrent request/job could have already overwritten by now.
        _, _, hist_df = fetch_data(sym)
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

