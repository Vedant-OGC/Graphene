"""
Graphene — AML Pattern Cypher Queries

Core graph detection capability for Graphene. Implements 7 Cypher query
functions that detect specific AML typologies in the Neo4j transaction graph.

Each function returns flagged accounts/transactions with full context for
the risk scoring engine and frontend dashboard.
"""

import logging
import time
from typing import Any

from graph.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "MAX_RESULTS": 1000,
    "DEFAULT_DEPTH": 3,
    "MAX_NODES": 200,
}


def _timed_query(
    client: Neo4jClient,
    query: str,
    params: dict[str, Any] | None = None,
    label: str = "query",
) -> list[dict]:
    """Execute a query and log its execution time.

    Args:
        client: Active Neo4jClient instance.
        query: Cypher query string.
        params: Optional query parameters.
        label: Label for logging.

    Returns:
        List of result records.
    """
    start = time.perf_counter()
    results = client.execute_query(query, params)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s: %d results in %.1fms", label, len(results), elapsed
    )
    return results


def detect_circular_transactions(
    client: Neo4jClient,
    max_hop: int = 5,
    time_window_hours: int = 72,
) -> list[dict]:
    """Detect circular (round-tripping) transaction patterns.

    Finds cycles where money flows A→B→C→...→A through TRANSFERRED_TO
    relationships, all within the specified time window. This is a
    classic money laundering typology used to obscure the origin of
    illicit funds.

    Args:
        client: Active Neo4jClient instance.
        max_hop: Maximum cycle length to search for.
        time_window_hours: Time window in hours for the cycle.

    Returns:
        List of detected circular patterns with account IDs, amounts,
        and timestamps. Returns empty list if none found.
    """
    query = """
    MATCH path = (a:Account)-[:TRANSFERRED_TO*2..%d]->(a)
    WHERE ALL(
        r IN relationships(path)
        WHERE r.timestamp > datetime() - duration('PT%dH')
    )
    WITH a,
         path,
         [n IN nodes(path) | n.account_id] AS account_ids,
         [r IN relationships(path) | r.txn_id] AS txn_ids,
         [r IN relationships(path) | r.amount] AS amounts,
         [r IN relationships(path) |
             toString(r.timestamp)] AS timestamps,
         length(path) AS cycle_length
    RETURN DISTINCT
        a.account_id AS anchor_account,
        account_ids,
        txn_ids,
        amounts,
        reduce(s = 0.0, a IN amounts | s + a) AS total_amount,
        cycle_length,
        timestamps
    ORDER BY total_amount DESC
    LIMIT %d
    """ % (max_hop, time_window_hours, CONFIG["MAX_RESULTS"])

    return _timed_query(
        client, query, label="detect_circular_transactions"
    )


def detect_layering(
    client: Neo4jClient,
    min_source_amount: float = 1_000_000,
    time_window_hours: int = 48,
) -> list[dict]:
    """Detect layering (rapid multi-hop fund dispersion) patterns.

    Identifies cases where one large source amount fans out to multiple
    intermediate accounts and then reconverges into a single destination.
    This is characteristic of layering — the second stage of money
    laundering.

    Args:
        client: Active Neo4jClient instance.
        min_source_amount: Minimum source transaction in INR.
        time_window_hours: Time window for the entire flow.

    Returns:
        List of layering pattern instances with source, intermediaries,
        and destination details. Returns empty list if none found.
    """
    query = """
    MATCH (source:Account)-[r1:TRANSFERRED_TO]->(mid:Account)
          -[r2:TRANSFERRED_TO]->(dest:Account)
    WHERE r1.amount > $min_amount
      AND r1.timestamp > datetime() - duration('PT%dH')
      AND r2.timestamp > r1.timestamp
      AND r2.timestamp < r1.timestamp + duration('PT%dH')
      AND source <> dest
      AND source <> mid
      AND mid <> dest
    WITH source,
         dest,
         collect(DISTINCT mid.account_id) AS intermediaries,
         collect(DISTINCT r1.txn_id) AS fan_out_txns,
         collect(DISTINCT r2.txn_id) AS converge_txns,
         sum(r1.amount) AS total_fan_out,
         sum(r2.amount) AS total_converge
    WHERE size(intermediaries) >= 2
    RETURN source.account_id AS source_account,
           intermediaries AS intermediate_accounts,
           dest.account_id AS destination_account,
           size(intermediaries) AS intermediary_count,
           total_fan_out,
           total_converge,
           fan_out_txns,
           converge_txns
    ORDER BY total_fan_out DESC
    LIMIT %d
    """ % (
        time_window_hours,
        time_window_hours,
        CONFIG["MAX_RESULTS"],
    )

    return _timed_query(
        client,
        query,
        {"min_amount": min_source_amount},
        label="detect_layering",
    )


def detect_structuring(
    client: Neo4jClient,
    threshold: float = 49_999,
    time_window_hours: int = 24,
    min_count: int = 5,
) -> list[dict]:
    """Detect structuring (smurfing) patterns below RBI threshold.

    Identifies multiple small transactions to the same destination,
    all below the ₹50,000 RBI reporting threshold, clustered within
    a short time window. This is a FATF-listed money laundering
    typology.

    Args:
        client: Active Neo4jClient instance.
        threshold: Amount threshold (default: just under ₹50,000).
        time_window_hours: Time window for clustering.
        min_count: Minimum number of transactions to flag.

    Returns:
        List of structuring pattern instances. Returns empty list
        if none found.
    """
    query = """
    MATCH (sender:Account)-[r:TRANSFERRED_TO]->(dest:Account)
    WHERE r.amount > 0 AND r.amount <= $threshold
      AND r.timestamp > datetime() - duration('PT%dH')
    WITH dest,
         collect(DISTINCT sender.account_id) AS senders,
         collect(r.txn_id) AS txn_ids,
         collect(r.amount) AS amounts,
         collect(toString(r.timestamp)) AS timestamps,
         count(r) AS txn_count,
         sum(r.amount) AS total_amount
    WHERE txn_count >= $min_count
    RETURN dest.account_id AS destination_account,
           senders AS sender_accounts,
           txn_count AS transaction_count,
           total_amount,
           txn_ids,
           amounts,
           timestamps
    ORDER BY txn_count DESC
    LIMIT %d
    """ % (time_window_hours, CONFIG["MAX_RESULTS"])

    return _timed_query(
        client,
        query,
        {"threshold": threshold, "min_count": min_count},
        label="detect_structuring",
    )


def detect_dormant_activation(
    client: Neo4jClient,
    dormant_days: int = 60,
    spike_multiplier: float = 3.0,
) -> list[dict]:
    """Detect dormant account activation patterns.

    Finds accounts with no transaction activity for the specified
    dormant period that suddenly receive large amounts and immediately
    forward most of it onward. This pattern is common in mule account
    networks.

    Args:
        client: Active Neo4jClient instance.
        dormant_days: Minimum days of inactivity before activation.
        spike_multiplier: Amount must exceed average by this factor.

    Returns:
        List of dormant activation patterns. Returns empty list
        if none found.
    """
    query = """
    MATCH (a:Account)
    WHERE a.account_type = 'DORMANT'
       OR a.account_age_days > $dormant_days

    OPTIONAL MATCH (a)<-[r_in:TRANSFERRED_TO]-(sender:Account)
    WHERE r_in.timestamp > datetime() - duration('P30D')
    WITH a,
         collect({
           sender: sender.account_id,
           amount: r_in.amount,
           timestamp: toString(r_in.timestamp),
           txn_id: r_in.txn_id
         }) AS incoming

    OPTIONAL MATCH (a)-[r_out:TRANSFERRED_TO]->(dest:Account)
    WHERE r_out.timestamp > datetime() - duration('P30D')
    WITH a, incoming,
         collect({
           dest: dest.account_id,
           amount: r_out.amount,
           timestamp: toString(r_out.timestamp),
           txn_id: r_out.txn_id
         }) AS outgoing

    WHERE size(incoming) > 0 AND size(outgoing) > 0

    WITH a,
         incoming,
         outgoing,
         reduce(s = 0.0, i IN incoming | s + i.amount)
             AS total_incoming,
         reduce(s = 0.0, o IN outgoing | s + o.amount)
             AS total_outgoing

    WHERE total_outgoing > total_incoming * 0.7

    RETURN a.account_id AS account_id,
           a.account_type AS account_type,
           a.account_age_days AS account_age_days,
           total_incoming AS activation_amount,
           total_outgoing AS forwarded_amount,
           toFloat(total_outgoing) / total_incoming
               AS forward_ratio,
           incoming,
           outgoing
    ORDER BY activation_amount DESC
    LIMIT $limit
    """

    return _timed_query(
        client,
        query,
        {
            "dormant_days": dormant_days,
            "limit": CONFIG["MAX_RESULTS"],
        },
        label="detect_dormant_activation",
    )


def detect_profile_mismatch(
    client: Neo4jClient,
    categories: list[str] | None = None,
    min_amount: float = 500_000,
) -> list[dict]:
    """Detect profile mismatch patterns.

    Identifies accounts in low-income categories (STUDENT, RETIRED)
    that receive high-value credits incompatible with their declared
    profile. This suggests the account may be used as a front for
    illicit fund flows.

    Args:
        client: Active Neo4jClient instance.
        categories: Customer categories to check.
        min_amount: Minimum transaction amount to flag.

    Returns:
        List of profile mismatch patterns. Returns empty list
        if none found.
    """
    if categories is None:
        categories = ["STUDENT", "RETIRED"]

    query = """
    MATCH (sender:Account)-[r:TRANSFERRED_TO]->(target:Account)
    WHERE target.customer_category IN $categories
      AND r.amount >= $min_amount
    WITH target,
         collect({
           sender: sender.account_id,
           amount: r.amount,
           timestamp: toString(r.timestamp),
           txn_id: r.txn_id
         }) AS suspicious_credits,
         count(r) AS credit_count,
         sum(r.amount) AS total_suspicious_amount
    RETURN target.account_id AS account_id,
           target.customer_category AS category,
           target.customer_name AS customer_name,
           credit_count,
           total_suspicious_amount,
           suspicious_credits
    ORDER BY total_suspicious_amount DESC
    LIMIT $limit
    """

    return _timed_query(
        client,
        query,
        {
            "categories": categories,
            "min_amount": min_amount,
            "limit": CONFIG["MAX_RESULTS"],
        },
        label="detect_profile_mismatch",
    )


def get_account_subgraph(
    client: Neo4jClient,
    account_id: str,
    depth: int = 3,
) -> dict:
    """Get full neighbourhood subgraph for graph visualisation.

    Returns all accounts and transactions within the specified hop
    depth from the given account, formatted for Cytoscape.js rendering.

    Args:
        client: Active Neo4jClient instance.
        account_id: Central account ID.
        depth: Number of hops to traverse (1-4).

    Returns:
        Dictionary with 'nodes' and 'edges' arrays for Cytoscape.js.
        Returns empty graph if account not found.
    """
    depth = max(1, min(depth, 4))

    query = """
    MATCH (center:Account {account_id: $account_id})
    CALL apoc.path.subgraphAll(center, {
        relationshipFilter: 'TRANSFERRED_TO>|<TRANSFERRED_TO',
        maxLevel: $depth,
        limit: $max_nodes
    }) YIELD nodes, relationships
    UNWIND nodes AS n
    WITH collect(DISTINCT n) AS all_nodes, relationships
    UNWIND all_nodes AS node
    WITH collect({
        id: node.account_id,
        account_type: node.account_type,
        customer_name: node.customer_name,
        customer_category: node.customer_category,
        risk_score: node.risk_score,
        is_flagged: node.is_flagged,
        kyc_status: node.kyc_status
    }) AS nodes, relationships
    UNWIND relationships AS rel
    WITH nodes, collect({
        source: startNode(rel).account_id,
        target: endNode(rel).account_id,
        amount: rel.amount,
        txn_type: rel.txn_type,
        timestamp: toString(rel.timestamp),
        txn_id: rel.txn_id,
        is_fraud: rel.is_fraud
    }) AS edges
    RETURN nodes, edges
    """

    fallback_query = """
    MATCH path = (center:Account {account_id: $account_id})
                 -[:TRANSFERRED_TO*1..%d]-(neighbor:Account)
    WITH collect(DISTINCT center) + collect(DISTINCT neighbor)
         AS all_nodes,
         [r IN collect(DISTINCT last(relationships(path)))
          | r] AS all_rels
    UNWIND all_nodes AS node
    WITH collect(DISTINCT {
        id: node.account_id,
        account_type: node.account_type,
        customer_name: node.customer_name,
        customer_category: node.customer_category,
        risk_score: node.risk_score,
        is_flagged: node.is_flagged,
        kyc_status: node.kyc_status
    })[..%d] AS nodes, all_rels
    UNWIND all_rels AS rel
    RETURN nodes,
           collect(DISTINCT {
               source: startNode(rel).account_id,
               target: endNode(rel).account_id,
               amount: rel.amount,
               txn_type: rel.txn_type,
               timestamp: toString(rel.timestamp),
               txn_id: rel.txn_id,
               is_fraud: rel.is_fraud
           }) AS edges
    """ % (depth, CONFIG["MAX_NODES"])

    try:
        results = _timed_query(
            client,
            query,
            {
                "account_id": account_id,
                "depth": depth,
                "max_nodes": CONFIG["MAX_NODES"],
            },
            label="get_account_subgraph",
        )
    except Exception:
        logger.info(
            "APOC not available, using fallback query."
        )
        results = _timed_query(
            client,
            fallback_query,
            {"account_id": account_id},
            label="get_account_subgraph_fallback",
        )

    if not results:
        return {"nodes": [], "edges": []}

    record = results[0]
    return {
        "nodes": record.get("nodes", []),
        "edges": record.get("edges", []),
    }


def trace_fund_path(
    client: Neo4jClient,
    source_account: str,
    dest_account: str,
) -> list[dict]:
    """Find shortest fund path between two accounts.

    Traces the shortest path through TRANSFERRED_TO relationships
    between the source and destination accounts, returning all nodes
    and edges along the path with full transaction details.

    Args:
        client: Active Neo4jClient instance.
        source_account: Source account ID.
        dest_account: Destination account ID.

    Returns:
        List of path records with nodes, edges, and transaction
        details. Returns empty list if no path exists.
    """
    query = """
    MATCH path = shortestPath(
        (source:Account {account_id: $source})
        -[:TRANSFERRED_TO*..10]->
        (dest:Account {account_id: $dest})
    )
    WITH path,
         [n IN nodes(path) | {
           id: n.account_id,
           account_type: n.account_type,
           customer_name: n.customer_name,
           risk_score: n.risk_score,
           is_flagged: n.is_flagged
         }] AS path_nodes,
         [r IN relationships(path) | {
           source: startNode(r).account_id,
           target: endNode(r).account_id,
           amount: r.amount,
           txn_type: r.txn_type,
           timestamp: toString(r.timestamp),
           txn_id: r.txn_id,
           is_fraud: r.is_fraud
         }] AS path_edges,
         length(path) AS path_length,
         reduce(
             s = 0.0,
             r IN relationships(path) | s + r.amount
         ) AS total_amount
    RETURN path_nodes AS nodes,
           path_edges AS edges,
           path_length,
           total_amount
    LIMIT 5
    """

    return _timed_query(
        client,
        query,
        {"source": source_account, "dest": dest_account},
        label="trace_fund_path",
    )


def run_all_pattern_queries(
    client: Neo4jClient,
) -> dict[str, list[dict]]:
    """Execute all 5 AML pattern detection queries.

    Convenience function that runs all pattern detectors and returns
    results grouped by pattern type. Used during startup to cache
    results and for periodic rescanning.

    Args:
        client: Active Neo4jClient instance.

    Returns:
        Dictionary mapping pattern type names to their results.
    """
    logger.info("Running all pattern detection queries...")
    start = time.perf_counter()

    results = {
        "CIRCULAR_ROUND_TRIP": detect_circular_transactions(client),
        "LAYERING": detect_layering(client),
        "STRUCTURING": detect_structuring(client),
        "DORMANT_ACTIVATION": detect_dormant_activation(client),
        "PROFILE_MISMATCH": detect_profile_mismatch(client),
    }

    elapsed = (time.perf_counter() - start) * 1000
    total_detections = sum(len(v) for v in results.values())

    logger.info(
        "All patterns scanned: %d total detections in %.1fms",
        total_detections,
        elapsed,
    )
    for pattern, hits in results.items():
        if hits:
            logger.info("  %s: %d hits", pattern, len(hits))

    return results
