#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
#  GRAPHENE — One-Command Demo Launcher
#  Fund Flow Intelligence — PSBs Hackathon 2026
# ──────────────────────────────────────────────────────────
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           GRAPHENE — Demo Launcher           ║${NC}"
echo -e "${CYAN}║      Fund Flow Intelligence System           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ─── CHECK PREREQUISITES ───
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed.${NC}"
    exit 1
fi

# ─── INSTALL DEPENDENCIES ───
echo -e "${YELLOW}[2/5] Installing Python dependencies...${NC}"
pip install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt

# ─── GENERATE DATA ───
echo -e "${YELLOW}[3/5] Generating synthetic transaction data...${NC}"
python -m data.generate_transactions
echo -e "${GREEN}  ✓ Generated transactions.csv and accounts.csv${NC}"

# ─── SEED DATABASE & TRAIN ───
echo -e "${YELLOW}[4/5] Seeding Neo4j and training ML models...${NC}"

if python -c "from neo4j import GraphDatabase" 2>/dev/null; then
    echo -e "  Seeding Neo4j database..."
    python -m data.seed_neo4j || echo -e "${YELLOW}  ⚠ Neo4j seeding skipped (check connection)${NC}"

    echo -e "  Training ML models (fast mode)..."
    python -m ml.train --epochs 30 --fast || echo -e "${YELLOW}  ⚠ ML training skipped${NC}"
else
    echo -e "${YELLOW}  ⚠ Neo4j driver not available, skipping database operations${NC}"
fi

# ─── LAUNCH API ───
echo -e "${YELLOW}[5/5] Launching Graphene API server...${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🚀 GRAPHENE IS RUNNING               ║${NC}"
echo -e "${GREEN}║                                              ║${NC}"
echo -e "${GREEN}║  Dashboard:  http://localhost:8000            ║${NC}"
echo -e "${GREEN}║  API Docs:   http://localhost:8000/docs       ║${NC}"
echo -e "${GREEN}║  Health:     http://localhost:8000/api/health  ║${NC}"
echo -e "${GREEN}║                                              ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop.                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
