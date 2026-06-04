"""
routers/analysis.py
━━━━━━━━━━━━━━━━━━━
POST /api/regime/{symbol}  — regime + pattern analysis
"""
from fastapi import APIRouter, HTTPException
from src.backend.services.nse_service import run_regime_analysis

router = APIRouter()


@router.post("/regime/{symbol}")
def regime_analysis(symbol: str):
    sym = symbol.upper().strip()
    try:
        return run_regime_analysis(sym)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

