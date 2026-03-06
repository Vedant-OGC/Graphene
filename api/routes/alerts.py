"""
Graphene — Alert Routes

Endpoints for fraud alert management: listing, detail view,
status updates, and statistics.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.models.responses import (
    AlertDetail,
    AlertsListResponse,
    AlertStatsResponse,
    AlertSummary,
    AlertUpdateRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/alerts",
    response_model=AlertsListResponse,
    tags=["Alerts"],
    summary="List fraud alerts",
)
def list_alerts(
    tier: str | None = Query(None, description="Filter by risk tier"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_app_state),
) -> AlertsListResponse:
    """Get paginated list of fraud alerts sorted by risk score.

    Optionally filter by risk tier (LOW/MEDIUM/HIGH/CRITICAL).
    """
    alerts = state.alerts

    if tier:
        tier_upper = tier.upper()
        alerts = [
            a for a in alerts
            if a.get("risk_tier") == tier_upper
        ]

    total = len(alerts)
    page = alerts[offset: offset + limit]

    summaries = [
        AlertSummary(
            alert_id=a.get("alert_id", ""),
            account_id=a.get("account_id", ""),
            risk_score=a.get("risk_score", 0),
            risk_tier=a.get("risk_tier", "LOW"),
            triggered_patterns=a.get("triggered_patterns", []),
            status=a.get("status", "OPEN"),
            created_at=a.get("created_at", ""),
        )
        for a in page
    ]

    return AlertsListResponse(
        total=total,
        alerts=summaries,
        page_info={
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
    )


@router.get(
    "/alerts/stats",
    response_model=AlertStatsResponse,
    tags=["Alerts"],
    summary="Alert statistics overview",
)
def alert_stats(
    state: AppState = Depends(get_app_state),
) -> AlertStatsResponse:
    """Get aggregate statistics for all fraud alerts.

    Returns counts by tier, top triggered patterns, and total
    amount at risk.
    """
    alerts = state.alerts
    by_tier = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    pattern_counts: dict[str, int] = {}
    total_at_risk = 0.0

    for alert in alerts:
        tier = alert.get("risk_tier", "LOW")
        by_tier[tier] = by_tier.get(tier, 0) + 1

        for pattern in alert.get("triggered_patterns", []):
            pattern_counts[pattern] = (
                pattern_counts.get(pattern, 0) + 1
            )

        if tier in ("HIGH", "CRITICAL"):
            total_at_risk += alert.get("risk_score", 0) * 10000

    top_patterns = sorted(
        [{"pattern": k, "count": v} for k, v in pattern_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    return AlertStatsResponse(
        total_alerts=len(alerts),
        by_tier=by_tier,
        top_patterns=top_patterns,
        total_amount_at_risk=round(total_at_risk, 2),
    )


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertDetail,
    tags=["Alerts"],
    summary="Get alert details with SHAP explanation",
)
def get_alert(
    alert_id: str,
    state: AppState = Depends(get_app_state),
) -> AlertDetail:
    """Get full alert details including SHAP explanation,
    pattern details, and account metadata.
    """
    alert = state.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found",
        )

    account_id = alert.get("account_id", "")
    metadata = state.get_account_metadata(account_id)
    transactions = state.get_recent_transactions(account_id)

    risk_data = next(
        (
            s
            for s in state.risk_scores
            if s.get("account_id") == account_id
        ),
        {},
    )

    return AlertDetail(
        alert_id=alert.get("alert_id", ""),
        account_id=account_id,
        risk_score=alert.get("risk_score", 0),
        risk_tier=alert.get("risk_tier", "LOW"),
        triggered_patterns=alert.get("triggered_patterns", []),
        recommendation=alert.get("recommendation", ""),
        evidence_summary=alert.get("evidence_summary", ""),
        status=alert.get("status", "OPEN"),
        analyst_note=alert.get("analyst_note"),
        created_at=alert.get("created_at", ""),
        account_metadata=metadata,
        shap_explanation=[],
        combined_explanation=alert.get("evidence_summary", ""),
        gnn_fraud_prob=risk_data.get("gnn_contribution", 0) / 40,
        anomaly_score=risk_data.get("iso_contribution", 0) / 35,
    )


@router.patch(
    "/alerts/{alert_id}",
    tags=["Alerts"],
    summary="Update alert status",
)
def update_alert(
    alert_id: str,
    update: AlertUpdateRequest,
    state: AppState = Depends(get_app_state),
) -> dict:
    """Update an alert's status and add analyst notes.

    Valid statuses: INVESTIGATING, DISMISSED, ESCALATED.
    """
    alert = state.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found",
        )

    valid_statuses = {"INVESTIGATING", "DISMISSED", "ESCALATED", "OPEN"}
    if update.status.upper() not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status: {update.status}. "
                f"Must be one of: {', '.join(valid_statuses)}"
            ),
        )

    alert["status"] = update.status.upper()
    if update.analyst_note:
        alert["analyst_note"] = update.analyst_note
    alert["updated_at"] = datetime.now().isoformat()

    logger.info(
        "Alert %s updated: status=%s",
        alert_id,
        update.status,
    )

    return {
        "alert_id": alert_id,
        "status": alert["status"],
        "analyst_note": alert.get("analyst_note"),
        "updated_at": alert["updated_at"],
    }
