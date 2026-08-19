"""
NSE-Neuron FastAPI Backend
Entry point — run with: uvicorn src.backend.main:app --reload --port 8000
"""
import sys
import os

# ── Add project root to sys.path so all existing modules are importable ──────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.routers import forecast, analysis, data, cache


app = FastAPI(
    title="NSE-Neuron API",
    description="Deep Learning Stock Forecasting API for NSE India",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(data.router,     prefix="/api", tags=["Data"])
app.include_router(cache.router,    prefix="/api", tags=["Model Cache"])


@app.get("/")
def root():
    return {"message": "NSE-Neuron API is running 🚀", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}

