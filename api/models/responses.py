"""
Graphene — API Response Models

Pydantic response models for all API endpoints. These ensure consistent
JSON response structure across the entire Graphene API.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /api/health."""
    status: str = "ok"
    neo4j_connected: bool = False
    models_loaded: bool = False
    accounts_loaded: int = 0
    alerts_generated: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class AlertSummary(BaseModel):
    """Compact alert for list view."""
    alert_id: str
    account_id: str
    risk_score: float
    risk_tier: str
    triggered_patterns: list[str] = Field(default_factory=list)
    status: str = "OPEN"
    created_at: str = ""


class AlertDetail(BaseModel):
    """Full alert with SHAP explanation."""
    alert_id: str
    account_id: str
    risk_score: float
    risk_tier: str
    triggered_patterns: list[str] = Field(default_factory=list)
    recommendation: str = ""
    evidence_summary: str = ""
    status: str = "OPEN"
    analyst_note: Optional[str] = None
    created_at: str = ""
    account_metadata: dict = Field(default_factory=dict)
    shap_explanation: list[dict] = Field(default_factory=list)
    combined_explanation: str = ""
    gnn_fraud_prob: float = 0.0
    anomaly_score: float = 0.0


class AlertsListResponse(BaseModel):
    """Response for GET /api/alerts."""
    total: int = 0
    alerts: list[AlertSummary] = Field(default_factory=list)
    page_info: dict = Field(default_factory=dict)


class AlertStatsResponse(BaseModel):
    """Response for GET /api/alerts/stats."""
    total_alerts: int = 0
    by_tier: dict = Field(default_factory=dict)
    top_patterns: list[dict] = Field(default_factory=list)
    total_amount_at_risk: float = 0.0


class AlertUpdateRequest(BaseModel):
    """Request for PATCH /api/alerts/{id}."""
    status: str
    analyst_note: Optional[str] = None


class GraphResponse(BaseModel):
    """Cytoscape.js graph data response."""
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)


class TraceRequest(BaseModel):
    """Request for POST /api/graph/trace."""
    source_account: str
    dest_account: str


class TraceResponse(BaseModel):
    """Response for POST /api/graph/trace."""
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    path_length: int = 0
    total_amount: float = 0.0
    intermediate_accounts: list[str] = Field(default_factory=list)


class PatternResponse(BaseModel):
    """Response for GET /api/graph/patterns."""
    patterns: dict = Field(default_factory=dict)
    total_detections: int = 0


class ReportGenerateRequest(BaseModel):
    """Request for POST /api/reports/generate."""
    alert_id: str
    analyst_name: str = "Investigator"
    analyst_id: str = "INV001"


class ReportGenerateResponse(BaseModel):
    """Response for POST /api/reports/generate."""
    report_id: str
    download_url: str
    generated_at: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""
    code: int = 500
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
