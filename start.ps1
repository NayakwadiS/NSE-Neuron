# ============================================================
#  NSE-Neuron — One-click launcher (PowerShell)
#  Usage:  .\start.ps1
# ============================================================

$Root = $PSScriptRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NSE-Neuron  |  Backend + Frontend Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Activate venv if present ──────────────────────────────────────────────
$venvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[Python] Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "[Python] No .venv found — using system Python" -ForegroundColor DarkYellow
}

# ── Install FastAPI backend deps if missing ───────────────────────────────
$fastapiCheck = pip show fastapi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[Backend] Installing FastAPI dependencies..." -ForegroundColor Yellow
    pip install -r (Join-Path $Root "src\backend\requirements.txt")
}

# ── Install Node deps if missing ──────────────────────────────────────────
$nodeModules = Join-Path $Root "src\frontend\node_modules"
if (-not (Test-Path $nodeModules)) {
    Write-Host "[Frontend] Running npm install..." -ForegroundColor Yellow
    Push-Location (Join-Path $Root "src\frontend")
    npm install
    Pop-Location
}

Write-Host ""
Write-Host "[Backend]  http://localhost:8000  (API docs: http://localhost:8000/docs)" -ForegroundColor Green
Write-Host "[Frontend] http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Starting both servers..." -ForegroundColor Cyan
Write-Host ""

# ── Start backend ─────────────────────────────────────────────────────────
$backendArgs = "-NoExit", "-Command", "cd '$Root'; uvicorn src.backend.main:app --reload --port 8000 --host 0.0.0.0"
Start-Process powershell -ArgumentList $backendArgs -WindowStyle Normal

# Small delay so backend starts first
Start-Sleep -Seconds 2

# ── Start frontend ────────────────────────────────────────────────────────
$frontendDir  = Join-Path $Root "src\frontend"
$frontendArgs = "-NoExit", "-Command", "cd '$frontendDir'; npm run dev"
Start-Process powershell -ArgumentList $frontendArgs -WindowStyle Normal

Write-Host "✅  Both servers are starting in separate PowerShell windows." -ForegroundColor Green
Write-Host "   Open http://localhost:5173 in your browser." -ForegroundColor Cyan
Write-Host ""

