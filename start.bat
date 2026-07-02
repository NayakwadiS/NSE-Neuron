@echo off
title NSE-Neuron Launcher

echo ============================================================
echo   NSE-Neuron — Starting Backend + Frontend
echo ============================================================
echo.

:: ── Check Python venv ──────────────────────────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [Python] Virtual env activated
) else (
    echo [Python] No .venv found — using system Python
)

:: ── Install backend extra deps if needed ───────────────────────────────────
echo [Backend] Checking FastAPI dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [Backend] Installing FastAPI + uvicorn...
    pip install -r src\backend\requirements.txt
)

:: ── Install frontend deps if needed ────────────────────────────────────────
echo [Frontend] Checking Node dependencies...
if not exist "src\frontend\node_modules" (
    echo [Frontend] Running npm install...
    cd src\frontend
    call npm install
    cd ..\..
)

echo.
echo [Backend]  Starting FastAPI on http://localhost:8000
echo [Frontend] Starting React  on http://localhost:5173
echo.
echo Press Ctrl+C in each window to stop the servers.
echo ============================================================
echo.

:: ── Start backend in a new window ──────────────────────────────────────────
start "NSE-Neuron Backend" cmd /k "cd /d %CD% && uvicorn src.backend.main:app --reload --port 8000 --host 0.0.0.0"

:: ── Start frontend in a new window ─────────────────────────────────────────
start "NSE-Neuron Frontend" cmd /k "cd /d %CD%\src\frontend && npm run dev"

echo Both servers launched in separate windows.
echo Open http://localhost:5173 in your browser.
pause

