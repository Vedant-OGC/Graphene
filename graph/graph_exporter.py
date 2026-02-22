"""
Graphene — Graph Exporter

Exports Neo4j subgraphs as Cytoscape.js-ready JSON for the frontend
dashboard. Transforms raw Neo4j query results into the specific JSON
structure expected by the graph visualisation layer.
"""

import logging
from typing import Any

from graph.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RISK_TIER_COLORS = {
    "CRITICAL": "#FF4757",
    "HIGH": "#FF6B35",
    "MEDIUM": "#FFA502",
    "LOW": "#2ED573",
    "NONE": "#636E72",
}

CONFIG = {
    "DEFAULT_NODE_SIZE": 30,
    "MIN_NODE_SIZE": 20,
    "MAX_NODE_SIZE": 80,
    "MIN_EDGE_WIDTH": 1,
    "MAX_EDGE_WIDTH": 8,
    "FRAUD_EDGE_COLOR": "#FF4757",
    "NORMAL_EDGE_COLOR": "rgba(255, 255, 255, 0.3)",
}


def _compute_risk_tier(risk_score: float) -> str:
    """Determine risk tier from score.

    Args:
        risk_score: Account risk score (0.0 to 1.0).

    Returns:
        Risk tier string.
    """
    score_100 = risk_score * 100
    if score_100 >= 76:
        return "CRITICAL"
    elif score_100 >= 56:
        return "HIGH"
    elif score_100 >= 31:
        return "MEDIUM"
    elif score_100 > 0:
        return "LOW"
    return "NONE"


def _compute_node_size(
    total_volume: float,
    max_volume: float,
) -> int:
    """Compute node size proportional to transaction volume.

    Args:
        total_volume: Total transaction volume for this account.
        max_volume: Maximum volume across all nodes in the subgraph.

    Returns:
        Node size in pixels.
    """
    if max_volume <= 0:
        return CONFIG["DEFAULT_NODE_SIZE"]

    ratio = total_volume / max_volume
    size = CONFIG["MIN_NODE_SIZE"] + ratio * (
        CONFIG["MAX_NODE_SIZE"] - CONFIG["MIN_NODE_SIZE"]
    )
    return int(size)


def _compute_edge_width(
    amount: float, max_amount: float
) -> float:
    """Compute edge width proportional to transaction amount.

    Args:
        amount: Transaction amount.
        max_amount: Maximum transaction amount in the subgraph.

    Returns:
        Edge width in pixels.
    """
    if max_amount <= 0:
        return CONFIG["MIN_EDGE_WIDTH"]

    ratio = amount / max_amount
    width = CONFIG["MIN_EDGE_WIDTH"] + ratio * (
        CONFIG["MAX_EDGE_WIDTH"] - CONFIG["MIN_EDGE_WIDTH"]
    )
    return round(width, 1)


def format_cytoscape_node(
    node_data: dict,
    max_volume: float = 0,
) -> dict:
    """Format a single node for Cytoscape.js rendering.

    Args:
        node_data: Raw node data from Neo4j query.
        max_volume: Max volume for sizing calculation.

    Returns:
        Cytoscape.js node object.
    """
    risk_score = node_data.get("risk_score", 0) or 0
    risk_tier = _compute_risk_tier(risk_score)
    total_volume = (
        node_data.get("total_sent", 0) or 0
    ) + (
        node_data.get("total_received", 0) or 0
    )

    return {
        "data": {
            "id": node_data.get("id", node_data.get("account_id", "")),
            "label": node_data.get(
                "id", node_data.get("account_id", "")
            )[:10],
            "risk_score": round(risk_score * 100, 1),
            "risk_tier": risk_tier,
            "account_type": node_data.get("account_type", "UNKNOWN"),
            "customer_name": node_data.get("customer_name", ""),
            "customer_category": node_data.get(
                "customer_category", ""
            ),
            "total_sent": node_data.get("total_sent", 0),
            "total_received": node_data.get("total_received", 0),
            "is_flagged": node_data.get("is_flagged", False),
            "kyc_status": node_data.get("kyc_status", "UNKNOWN"),
            "color": RISK_TIER_COLORS.get(risk_tier, "#636E72"),
            "size": _compute_node_size(total_volume, max_volume),
        }
    }


def format_cytoscape_edge(
    edge_data: dict,
    max_amount: float = 0,
) -> dict:
    """Format a single edge for Cytoscape.js rendering.

    Args:
        edge_data: Raw edge data from Neo4j query.
        max_amount: Max amount for width calculation.

    Returns:
        Cytoscape.js edge object.
    """
    is_fraud = edge_data.get("is_fraud", False)
    amount = edge_data.get("amount", 0) or 0

    return {
        "data": {
            "id": edge_data.get("txn_id", f"{edge_data.get('source', '')}"
                                f"__{edge_data.get('target', '')}"),
            "source": edge_data.get("source", ""),
            "target": edge_data.get("target", ""),
            "amount": amount,
            "txn_type": edge_data.get("txn_type", ""),
            "timestamp": edge_data.get("timestamp", ""),
            "is_fraud": is_fraud,
            "fraud_type": edge_data.get("fraud_type", ""),
            "color": (
                CONFIG["FRAUD_EDGE_COLOR"]
                if is_fraud
                else CONFIG["NORMAL_EDGE_COLOR"]
            ),
            "width": _compute_edge_width(amount, max_amount),
        }
    }


def export_subgraph_json(
    raw_nodes: list[dict],
    raw_edges: list[dict],
) -> dict:
    """Export a subgraph as Cytoscape.js-ready JSON.

    Takes raw node and edge data from Neo4j queries and transforms
    them into the specific JSON structure expected by the frontend.
    Computes sizes, widths, and colors based on data values.

    Args:
        raw_nodes: List of raw node dictionaries from Neo4j.
        raw_edges: List of raw edge dictionaries from Neo4j.

    Returns:
        Dictionary with 'nodes' and 'edges' arrays formatted
        for Cytoscape.js.
    """
    if not raw_nodes:
        return {"nodes": [], "edges": []}

    max_volume = max(
        (
            (n.get("total_sent", 0) or 0)
            + (n.get("total_received", 0) or 0)
        )
        for n in raw_nodes
    ) if raw_nodes else 0

    max_amount = max(
        (e.get("amount", 0) or 0) for e in raw_edges
    ) if raw_edges else 0

    nodes = [
        format_cytoscape_node(n, max_volume) for n in raw_nodes
    ]
    edges = [
        format_cytoscape_edge(e, max_amount) for e in raw_edges
    ]

    # Deduplicate nodes and build a valid ID set
    seen_ids: set[str] = set()
    unique_nodes = []
    for node in nodes:
        nid = node["data"]["id"]
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            unique_nodes.append(node)

    # Drop edges whose source or target is not in the node set —
    # Cytoscape throws a fatal error if either endpoint is missing
    valid_edges = [
        e for e in edges
        if e["data"].get("source") in seen_ids
        and e["data"].get("target") in seen_ids
    ]

    # Deduplicate edges by ID
    seen_edge_ids: set[str] = set()
    unique_edges = []
    for edge in valid_edges:
        eid = edge["data"]["id"]
        if eid not in seen_edge_ids:
            seen_edge_ids.add(eid)
            unique_edges.append(edge)

    logger.info(
        "Exported subgraph: %d nodes, %d edges (%d orphan edges dropped)",
        len(unique_nodes),
        len(unique_edges),
        len(edges) - len(unique_edges),
    )

    return {"nodes": unique_nodes, "edges": unique_edges}


def export_fund_path_json(
    path_result: dict,
) -> dict:
    """Export a fund flow path as Cytoscape.js JSON.

    Args:
        path_result: Result from trace_fund_path query.

    Returns:
        Cytoscape.js graph data for the path.
    """
    nodes = path_result.get("nodes", [])
    edges = path_result.get("edges", [])

    return export_subgraph_json(nodes, edges)


def export_pattern_graph_json(
    client: Neo4jClient,
    pattern_result: dict,
) -> dict:
    """Export a detected pattern as a focused subgraph.

    Takes a pattern detection result and fetches the relevant
    subgraph around the involved accounts for visualisation.

    Args:
        client: Active Neo4jClient instance.
        pattern_result: A single pattern detection result.

    Returns:
        Cytoscape.js graph data for the pattern.
    """
    account_ids = set()

    for key in [
        "account_ids", "sender_accounts",
        "intermediate_accounts",
    ]:
        if key in pattern_result:
            val = pattern_result[key]
            if isinstance(val, list):
                account_ids.update(val)

    for key in [
        "anchor_account", "source_account",
        "destination_account", "account_id",
    ]:
        if key in pattern_result:
            account_ids.add(pattern_result[key])

    if not account_ids:
        return {"nodes": [], "edges": []}

    query = """
    MATCH (a:Account)
    WHERE a.account_id IN $account_ids
    OPTIONAL MATCH (a)-[r:TRANSFERRED_TO]-(b:Account)
    WHERE b.account_id IN $account_ids
    RETURN collect(DISTINCT {
        id: a.account_id,
        account_type: a.account_type,
        customer_name: a.customer_name,
        risk_score: a.risk_score,
        is_flagged: a.is_flagged
    }) AS nodes,
    collect(DISTINCT {
        source: startNode(r).account_id,
        target: endNode(r).account_id,
        amount: r.amount,
        txn_type: r.txn_type,
        timestamp: toString(r.timestamp),
        txn_id: r.txn_id,
        is_fraud: r.is_fraud
    }) AS edges
    """

    results = client.execute_query(
        query, {"account_ids": list(account_ids)}
    )

    if not results:
        return {"nodes": [], "edges": []}

    record = results[0]
    return export_subgraph_json(
        record.get("nodes", []),
        record.get("edges", []),
    )
