# Graphene — Fund Flow Intelligence

> **"See through every transaction."**

AI-powered Fund Flow Tracking and Fraud Detection system for the PSBs Hackathon 2026 (Union Bank of India, Problem Statement PS3).

Graphene maps end-to-end movement of funds across bank accounts, detects suspicious patterns using Graph Neural Networks and anomaly detection, and presents findings to fraud investigators through an interactive graph visualisation dashboard.

---

## Features

- **Graph-Based Fund Tracking** — Neo4j graph database maps every money movement between accounts
- **5 AML Pattern Detection** — Circular round-tripping, layering, structuring, dormant activation, profile mismatch
- **GraphSAGE Neural Network** — Learns structural fraud signatures from account neighbourhood context
- **Isolation Forest Anomaly Detection** — Unsupervised detection of novel fraud types
- **SHAP Explanations** — Every prediction comes with human-readable "why" explanations
- **Risk Scoring Engine** — Weighted ensemble: GNN (40%) + Anomaly (35%) + Rules (25%) → 0–100 score
- **Interactive Dashboard** — Cytoscape.js graph visualisation with real-time alerts
- **FIU Report Generation** — One-click PDF Suspicious Transaction Report for FIU-IND

---

## Quick Start

### Prerequisites
- Python 3.11+
- Neo4j Desktop or Community Edition (running on `bolt://localhost:7687`)

### Setup

```bash
# 1. Clone and enter directory
cd graphene

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file
copy .env.example .env     # Windows
# cp .env.example .env     # macOS/Linux

# 5. Start Neo4j and update .env with your credentials
```

### Run Demo

```bash
# Generate synthetic data
python data/generate_transactions.py

# Seed Neo4j database
python data/seed_neo4j.py

# Train ML models
python ml/train.py --epochs 30 --fast

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

---

## Architecture

```
Layer 1 — Data Ingestion     → Synthetic transaction generator + Neo4j seeder
Layer 2 — Graph Analytics     → Neo4j Cypher pattern detection queries
Layer 3 — ML Detection        → GraphSAGE GNN + Isolation Forest + SHAP
Layer 4 — Risk Scoring        → Weighted ensemble → 0-100 risk score
Layer 5 — Investigation UI    → React + Cytoscape.js dashboard + FIU reports
```

---

## Project Structure

```
graphene/
├── data/                    # Data generation and loading
│   ├── generate_transactions.py
│   ├── fraud_patterns.py
│   ├── schema.py
│   └── seed_neo4j.py
├── graph/                   # Neo4j graph operations
│   ├── neo4j_client.py
│   ├── graph_builder.py
│   ├── pattern_queries.py
│   └── graph_exporter.py
├── ml/                      # Machine learning models
│   ├── gnn_model.py
│   ├── anomaly_detector.py
│   ├── risk_scorer.py
│   ├── shap_explainer.py
│   ├── feature_engineering.py
│   └── train.py
├── api/                     # FastAPI backend
│   ├── main.py
│   ├── dependencies.py
│   └── routes/
├── frontend/                # Dashboard UI
│   └── index.html
├── requirements.txt
├── .env
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph Database | Neo4j 5.x |
| GNN | PyTorch + PyTorch Geometric (GraphSAGE) |
| Anomaly Detection | scikit-learn (Isolation Forest + LOF) |
| Explainability | SHAP |
| API | FastAPI + uvicorn |
| Frontend | React 18 + Cytoscape.js |
| PDF Reports | ReportLab |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/alerts` | List fraud alerts (filterable) |
| GET | `/api/alerts/{id}` | Alert details with SHAP |
| PATCH | `/api/alerts/{id}` | Update alert status |
| GET | `/api/alerts/stats` | Alert statistics |
| GET | `/api/graph/{account_id}` | Account subgraph |
| POST | `/api/graph/trace` | Trace fund path |
| GET | `/api/graph/patterns` | Detected AML patterns |
| POST | `/api/reports/generate` | Generate FIU STR PDF |
| GET | `/api/reports/{id}/download` | Download report |

---

## Risk Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0–30 | LOW (green) | Normal monitoring |
| 31–55 | MEDIUM (amber) | Enhanced due diligence |
| 56–75 | HIGH (orange) | Immediate review required |
| 76–100 | CRITICAL (red) | Freeze account, escalate to FIU |

---

## Team

Built for **PSBs Hackathon Series 2026** — Union Bank of India

---

*Graphene — "See through every transaction."*
