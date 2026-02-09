@echo off
REM ──────────────────────────────────────────────────────────
REM  GRAPHENE — One-Command Demo Launcher (Windows)
REM  Fund Flow Intelligence — PSBs Hackathon 2026
REM ──────────────────────────────────────────────────────────

echo.
echo ╔══════════════════════════════════════════════╗
echo ║           GRAPHENE — Demo Launcher           ║
echo ║      Fund Flow Intelligence System           ║
echo ╚══════════════════════════════════════════════╝
echo.

REM ─── CHECK PYTHON ───
echo [1/5] Checking prerequisites...
python --version >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3 is not installed.
    exit /b 1
)

REM ─── INSTALL DEPS ───
echo [2/5] Installing Python dependencies...
pip install -r requirements.txt --quiet 2>NUL || pip install -r requirements.txt

REM ─── GENERATE DATA ───
echo [3/5] Generating synthetic transaction data...
python -m data.generate_transactions
echo   Generated transactions.csv and accounts.csv

REM ─── SEED & TRAIN ───
echo [4/5] Seeding Neo4j and training ML models...
python -m data.seed_neo4j 2>NUL || echo   Neo4j seeding skipped (check connection)
python -m ml.train --epochs 30 --fast 2>NUL || echo   ML training skipped

REM ─── LAUNCH ───
echo [5/5] Launching Graphene API server...
echo.
echo ╔══════════════════════════════════════════════╗
echo ║         GRAPHENE IS RUNNING                  ║
echo ║                                              ║
echo ║  Dashboard:  http://localhost:8000            ║
echo ║  API Docs:   http://localhost:8000/docs       ║
echo ║  Health:     http://localhost:8000/api/health  ║
echo ║                                              ║
echo ║  Press Ctrl+C to stop.                       ║
echo ╚══════════════════════════════════════════════╝
echo.

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
