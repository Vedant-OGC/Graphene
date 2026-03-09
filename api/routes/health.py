"""
Graphene — Health Check Route

Provides system health status including Neo4j connectivity
and model loading state.
"""

from fastapi import APIRouter, Depends

from api.dependencies import AppState, get_app_state
from api.models.responses import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System health check",
)
def health_check(
    state: AppState = Depends(get_app_state),
) -> HealthResponse:
    """Check system health and component status.

    Returns connectivity status for Neo4j and whether ML models
    are loaded and ready for inference.
    """
    neo4j_ok = False
    if state.neo4j_client:
        neo4j_ok = state.neo4j_client.health_check()

    return HealthResponse(
        status="ok" if neo4j_ok and state.models_loaded else "degraded",
        neo4j_connected=neo4j_ok,
        models_loaded=state.models_loaded,
        accounts_loaded=len(state.risk_scores),
        alerts_generated=len(state.alerts),
    )
