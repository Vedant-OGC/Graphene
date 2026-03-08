"""
Graphene — Graph Routes

Endpoints for graph visualisation: account subgraph retrieval,
fund path tracing, and detected pattern listing.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.models.responses import (
    GraphResponse,
    PatternResponse,
    TraceRequest,
    TraceResponse,
)
from graph.graph_exporter import export_subgraph_json
from graph.pattern_queries import (
    get_account_subgraph,
    trace_fund_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/graph/{account_id}",
    response_model=GraphResponse,
    tags=["Graph"],
    summary="Get account subgraph for visualisation",
)
def get_graph(
    account_id: str,
    depth: int = Query(2, ge=1, le=4),
    state: AppState = Depends(get_app_state),
) -> GraphResponse:
    """Get the neighbourhood subgraph for a specific account.

    Returns Cytoscape.js-ready graph JSON with nodes and edges
    formatted for the frontend dashboard.
    """
    if not state.neo4j_client:
        raise HTTPException(
            status_code=503,
            detail="Neo4j not connected. Cannot retrieve graph.",
        )

    try:
        raw_graph = get_account_subgraph(
            state.neo4j_client, account_id, depth
        )

        if not raw_graph.get("nodes"):
            raise HTTPException(
                status_code=404,
                detail=f"Account {account_id} not found in graph.",
            )

        formatted = export_subgraph_json(
            raw_graph["nodes"], raw_graph["edges"]
        )

        return GraphResponse(
            nodes=formatted["nodes"],
            edges=formatted["edges"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Graph retrieval failed for %s", account_id)
        raise HTTPException(
            status_code=500,
            detail=f"Graph retrieval error: {e}",
        )


@router.post(
    "/graph/trace",
    response_model=TraceResponse,
    tags=["Graph"],
    summary="Trace fund path between two accounts",
)
def trace_path(
    request: TraceRequest,
    state: AppState = Depends(get_app_state),
) -> TraceResponse:
    """Find the shortest fund flow path between two accounts.

    Returns all nodes and edges along the path with full
    transaction details.
    """
    if not state.neo4j_client:
        raise HTTPException(
            status_code=503,
            detail="Neo4j not connected.",
        )

    try:
        results = trace_fund_path(
            state.neo4j_client,
            request.source_account,
            request.dest_account,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No path found between "
                    f"{request.source_account} and "
                    f"{request.dest_account}."
                ),
            )

        path = results[0]
        nodes = path.get("nodes", [])
        edges = path.get("edges", [])

        formatted = export_subgraph_json(nodes, edges)

        intermediate = [
            n.get("id", "")
            for n in nodes[1:-1]
        ] if len(nodes) > 2 else []

        return TraceResponse(
            nodes=formatted["nodes"],
            edges=formatted["edges"],
            path_length=path.get("path_length", len(nodes) - 1),
            total_amount=path.get("total_amount", 0),
            intermediate_accounts=intermediate,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Path trace failed")
        raise HTTPException(
            status_code=500,
            detail=f"Path trace error: {e}",
        )


@router.get(
    "/graph/patterns",
    response_model=PatternResponse,
    tags=["Graph"],
    summary="Get all detected AML patterns",
)
def get_patterns(
    state: AppState = Depends(get_app_state),
) -> PatternResponse:
    """Return all detected AML pattern instances grouped by type.

    Each instance includes involved accounts, transaction IDs,
    and total amounts.
    """
    patterns = state.pattern_results
    total = sum(len(v) for v in patterns.values())

    return PatternResponse(
        patterns=patterns,
        total_detections=total,
    )
