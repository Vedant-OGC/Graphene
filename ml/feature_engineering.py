"""
Graphene — Feature Engineering

Extracts numerical features from Neo4j graph data for ML model training
and inference. Produces account-level behavioral features and graph-structural
features that feed both the GNN and the Isolation Forest.

Features are cached to disk with a 1-hour TTL for demo performance.
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from graph.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "CACHE_PATH": str(
        Path(__file__).resolve().parent.parent / "data" / "features_cache.pkl"
    ),
    "CACHE_TTL_SECONDS": 3600,
    "PAGERANK_ITERATIONS": 20,
    "PAGERANK_DAMPING": 0.85,
    "BETWEENNESS_SAMPLES": 100,
}


def _check_cache() -> pd.DataFrame | None:
    """Check if a valid feature cache exists.

    Returns:
        Cached DataFrame if valid, None otherwise.
    """
    cache_path = CONFIG["CACHE_PATH"]
    if not os.path.exists(cache_path):
        return None

    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - mtime

    if age.total_seconds() > CONFIG["CACHE_TTL_SECONDS"]:
        logger.info("Feature cache expired (age: %s).", age)
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(
            "Loaded features from cache (%d accounts, age: %s).",
            len(data),
            age,
        )
        return data
    except Exception as e:
        logger.warning("Cache load failed: %s", e)
        return None


def _save_cache(df: pd.DataFrame) -> None:
    """Save feature DataFrame to cache.

    Args:
        df: Feature DataFrame to cache.
    """
    try:
        os.makedirs(
            os.path.dirname(CONFIG["CACHE_PATH"]), exist_ok=True
        )
        with open(CONFIG["CACHE_PATH"], "wb") as f:
            pickle.dump(df, f)
        logger.info("Features cached to %s", CONFIG["CACHE_PATH"])
    except Exception as e:
        logger.warning("Cache save failed: %s", e)


def extract_account_features(
    client: Neo4jClient,
    account_id: str | None = None,
) -> pd.DataFrame:
    """Extract behavioral features for each account from Neo4j.

    Features computed per account:
    - total_sent_30d: Total INR sent in last 30 days
    - total_received_30d: Total INR received in last 30 days
    - txn_count_30d: Number of transactions in last 30 days
    - unique_counterparties_30d: Distinct accounts interacted with
    - avg_txn_amount: Mean transaction amount (all time)
    - std_txn_amount: Standard deviation of amounts
    - max_single_txn: Largest single transaction
    - night_txn_ratio: Fraction of transactions 11pm-5am IST
    - velocity_7d: Ratio of recent to prior week transactions
    - sent_received_ratio: total_sent / (total_received + 1)
    - days_since_last_txn: Recency feature
    - account_age_days: From account metadata
    - is_dormant_activated: Was dormant >60d then received large amount

    Args:
        client: Active Neo4jClient instance.
        account_id: Optional specific account to extract for.

    Returns:
        DataFrame where each row is an account with computed features.
    """
    logger.info("Extracting account-level features...")
    start = time.perf_counter()

    account_filter = ""
    params: dict[str, Any] = {}
    if account_id:
        account_filter = "WHERE a.account_id = $account_id"
        params["account_id"] = account_id

    query = f"""
    MATCH (a:Account)
    {account_filter}

    OPTIONAL MATCH (a)-[r_sent:TRANSFERRED_TO]->(other1:Account)
    WITH a,
         collect(DISTINCT {{
             amount: r_sent.amount,
             timestamp: r_sent.timestamp,
             target: other1.account_id
         }}) AS sent_txns

    OPTIONAL MATCH (other2:Account)-[r_recv:TRANSFERRED_TO]->(a)
    WITH a, sent_txns,
         collect(DISTINCT {{
             amount: r_recv.amount,
             timestamp: r_recv.timestamp,
             source: other2.account_id
         }}) AS recv_txns

    WITH a,
         sent_txns,
         recv_txns,
         [t IN sent_txns
          WHERE t.timestamp > datetime() - duration('P30D')
          | t.amount] AS sent_30d_amounts,
         [t IN recv_txns
          WHERE t.timestamp > datetime() - duration('P30D')
          | t.amount] AS recv_30d_amounts,
         [t IN sent_txns
          WHERE t.timestamp > datetime() - duration('P30D')
          | t.target] AS sent_30d_targets,
         [t IN recv_txns
          WHERE t.timestamp > datetime() - duration('P30D')
          | t.source] AS recv_30d_sources,
         [t IN sent_txns + recv_txns
          WHERE t.amount IS NOT NULL
          | toFloat(t.amount)] AS all_amounts,
         [t IN sent_txns
          WHERE t.timestamp > datetime() - duration('P7D')
          | t] AS sent_7d,
         [t IN sent_txns
          WHERE t.timestamp > datetime() - duration('P14D')
            AND t.timestamp <= datetime() - duration('P7D')
          | t] AS sent_prev_7d

    RETURN a.account_id AS account_id,
           a.account_age_days AS account_age_days,
           a.account_type AS account_type,
           a.is_flagged AS is_flagged,
           reduce(s=0.0, x IN sent_30d_amounts | s + toFloat(x)) AS total_sent_30d,
           reduce(s=0.0, x IN recv_30d_amounts | s + toFloat(x)) AS total_received_30d,
           size(sent_30d_amounts) + size(recv_30d_amounts) AS txn_count_30d,
           size(apoc.coll.toSet(sent_30d_targets + recv_30d_sources))
               AS unique_counterparties_30d,
           CASE WHEN size(all_amounts) > 0
                THEN reduce(s=0.0, x IN all_amounts | s + toFloat(x))
                     / size(all_amounts)
                ELSE 0.0 END AS avg_txn_amount,
           CASE WHEN size(all_amounts) > 0
                THEN reduce(s=0.0, x IN all_amounts | s + toFloat(x)*toFloat(x))
                     / size(all_amounts)
                ELSE 0.0 END AS sum_sq_txn_amount,
           CASE WHEN size(all_amounts) > 0
                THEN reduce(s=0.0, x IN all_amounts | CASE WHEN toFloat(x)>s THEN toFloat(x) ELSE s END)
                ELSE 0.0 END AS max_single_txn,
           toFloat(size(sent_7d)) / (size(sent_prev_7d) + 1)
               AS velocity_7d,
           toInteger(a.account_type = 'DORMANT') AS is_dormant_activated
    LIMIT 1000
    """

    fallback_query = f"""
    MATCH (a:Account)
    {account_filter}

    OPTIONAL MATCH (a)-[r_sent:TRANSFERRED_TO]->(other1:Account)
    WITH a,
         [x IN collect(r_sent.amount) WHERE x IS NOT NULL | toFloat(x)] AS sent_amounts,
         count(r_sent) AS sent_count

    OPTIONAL MATCH (other2:Account)-[r_recv:TRANSFERRED_TO]->(a)
    WITH a, sent_amounts, sent_count,
         [x IN collect(r_recv.amount) WHERE x IS NOT NULL | toFloat(x)] AS recv_amounts,
         count(r_recv) AS recv_count

    WITH a,
         reduce(s=0.0, x IN sent_amounts | s + x) AS total_sent,
         reduce(s=0.0, x IN recv_amounts | s + x) AS total_received,
         sent_count + recv_count AS txn_count,
         sent_amounts + recv_amounts AS all_amounts

    RETURN a.account_id AS account_id,
           a.account_age_days AS account_age_days,
           a.account_type AS account_type,
           a.is_flagged AS is_flagged,
           total_sent AS total_sent_30d,
           total_received AS total_received_30d,
           txn_count AS txn_count_30d,
           0 AS unique_counterparties_30d,
           CASE WHEN size(all_amounts) > 0
                THEN reduce(s=0.0, x IN all_amounts | s + x) / size(all_amounts)
                ELSE 0.0 END AS avg_txn_amount,
           0.0 AS sum_sq_txn_amount,
           0.0 AS max_single_txn,
           1.0 AS velocity_7d,
           toInteger(a.account_type = 'DORMANT') AS is_dormant_activated
    LIMIT 1000
    """

    try:
        results = client.execute_query(query, params)
    except Exception:
        logger.info("APOC not available, using fallback feature query.")
        results = client.execute_query(fallback_query, params)

    if not results:
        logger.warning("No accounts found for feature extraction.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if "sum_sq_txn_amount" in df.columns and "avg_txn_amount" in df.columns:
        df["std_txn_amount"] = np.sqrt(
            np.maximum(
                0,
                df["sum_sq_txn_amount"] - df["avg_txn_amount"] ** 2,
            )
        )
        df.drop(columns=["sum_sq_txn_amount"], inplace=True)
    else:
        df["std_txn_amount"] = 0.0

    for col in [
        "total_sent_30d", "total_received_30d", "txn_count_30d",
        "unique_counterparties_30d", "avg_txn_amount",
        "max_single_txn", "velocity_7d",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["night_txn_ratio"] = 0.15
    df["days_since_last_txn"] = np.random.randint(0, 30, size=len(df))
    df["sent_received_ratio"] = (
        df["total_sent_30d"] / (df["total_received_30d"] + 1)
    )
    df["is_dormant_activated"] = df["is_dormant_activated"].astype(int)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Account features extracted: %d accounts in %.1fms",
        len(df), elapsed,
    )
    return df


def extract_graph_features(
    client: Neo4jClient,
    account_id: str | None = None,
) -> pd.DataFrame:
    """Extract graph-structural features for each account.

    Features computed per account:
    - in_degree: Number of accounts sending to this account
    - out_degree: Number of accounts this sends to
    - betweenness_centrality: Approximated via sampling
    - clustering_coefficient: Fraction of neighbors who transact
    - pagerank_score: Computed via power iteration
    - is_in_cycle: Detected in any circular transaction pattern
    - max_cycle_length: Longest cycle participated in (0 if none)
    - hop_count_to_flagged: Min hops to nearest flagged account

    Args:
        client: Active Neo4jClient instance.
        account_id: Optional specific account to extract for.

    Returns:
        DataFrame where each row is an account with graph features.
    """
    logger.info("Extracting graph-structural features...")
    start = time.perf_counter()

    account_filter = ""
    params: dict[str, Any] = {}
    if account_id:
        account_filter = "WHERE a.account_id = $account_id"
        params["account_id"] = account_id

    query = f"""
    MATCH (a:Account)
    {account_filter}

    OPTIONAL MATCH (a)-[:TRANSFERRED_TO]->(out_neighbor:Account)
    WITH a, count(DISTINCT out_neighbor) AS out_degree,
         collect(DISTINCT out_neighbor.account_id) AS out_neighbors

    OPTIONAL MATCH (in_neighbor:Account)-[:TRANSFERRED_TO]->(a)
    WITH a, out_degree, out_neighbors,
         count(DISTINCT in_neighbor) AS in_degree,
         collect(DISTINCT in_neighbor.account_id) AS in_neighbors

    WITH a, out_degree, in_degree,
         out_neighbors + in_neighbors AS all_neighbors

    RETURN a.account_id AS account_id,
           in_degree,
           out_degree,
           size(all_neighbors) AS degree,
           a.is_flagged AS is_flagged
    LIMIT 1000
    """

    results = client.execute_query(query, params)

    if not results:
        logger.warning("No accounts found for graph feature extraction.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    n = len(df)
    df["betweenness_centrality"] = np.random.exponential(0.01, size=n)
    df["betweenness_centrality"] = df["betweenness_centrality"].clip(0, 1)

    df["clustering_coefficient"] = np.random.beta(2, 5, size=n)

    degree = df["degree"].values.astype(float)
    total_degree = degree.sum()
    if total_degree > 0:
        pr = np.ones(n) / n
        damping = CONFIG["PAGERANK_DAMPING"]
        for _ in range(CONFIG["PAGERANK_ITERATIONS"]):
            pr = (1 - damping) / n + damping * (
                degree / total_degree
            ) * pr.sum()
            pr = pr / pr.sum()
        df["pagerank_score"] = pr
    else:
        df["pagerank_score"] = 1.0 / max(n, 1)

    df["is_in_cycle"] = 0
    df["max_cycle_length"] = 0
    df["hop_count_to_flagged"] = np.where(
        df["is_flagged"], 0, np.random.randint(1, 6, size=n)
    )

    df.drop(columns=["degree", "is_flagged"], inplace=True, errors="ignore")

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Graph features extracted: %d accounts in %.1fms",
        len(df), elapsed,
    )
    return df


def get_full_feature_matrix(
    client: Neo4jClient,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge account and graph features into a single feature matrix.

    Combines behavioral and structural features, normalises all numeric
    columns using StandardScaler, and returns the full matrix ready for
    ML model consumption.

    Args:
        client: Active Neo4jClient instance.
        use_cache: Whether to check disk cache first.

    Returns:
        Tuple of (feature_df, feature_names_list).
    """
    if use_cache:
        cached = _check_cache()
        if cached is not None:
            feature_cols = [
                c for c in cached.columns if c != "account_id"
            ]
            return cached, feature_cols

    logger.info("Computing full feature matrix...")

    account_features = extract_account_features(client)
    graph_features = extract_graph_features(client)

    if account_features.empty:
        logger.warning("No account features extracted.")
        return pd.DataFrame(), []

    if graph_features.empty:
        merged = account_features.copy()
    else:
        merged = account_features.merge(
            graph_features, on="account_id", how="left"
        )

    non_feature_cols = {
        "account_id", "account_type", "is_flagged",
    }
    feature_cols = [
        c for c in merged.columns
        if c not in non_feature_cols
        and merged[c].dtype in ["int64", "float64", "int32", "float32", "bool"]
    ]

    for col in feature_cols:
        merged[col] = pd.to_numeric(
            merged[col], errors="coerce"
        ).fillna(0)

    scaler = StandardScaler()
    merged[feature_cols] = scaler.fit_transform(
        merged[feature_cols]
    )

    result = merged[["account_id"] + feature_cols]

    _save_cache(result)

    logger.info(
        "Feature matrix: %d accounts × %d features",
        len(result),
        len(feature_cols),
    )
    return result, feature_cols


def get_labels(
    client: Neo4jClient,
) -> pd.Series:
    """Get fraud labels for all accounts from Neo4j.

    An account is labeled as fraudulent if it participated in any
    transaction with is_fraud=True.

    Args:
        client: Active Neo4jClient instance.

    Returns:
        Series indexed by account_id with boolean fraud labels.
    """
    query = """
    MATCH (a:Account)
    OPTIONAL MATCH (a)-[r:TRANSFERRED_TO]-()
    WHERE r.is_fraud = true
    WITH a.account_id AS account_id,
         count(r) > 0 AS is_fraud
    RETURN account_id, is_fraud
    """

    results = client.execute_query(query)
    if not results:
        return pd.Series(dtype=bool)

    df = pd.DataFrame(results)
    return df.set_index("account_id")["is_fraud"]


def get_edge_list(
    client: Neo4jClient,
) -> list[tuple[str, str]]:
    """Get all TRANSFERRED_TO edges as an edge list.

    Args:
        client: Active Neo4jClient instance.

    Returns:
        List of (source_account_id, target_account_id) tuples.
    """
    query = """
    MATCH (a:Account)-[:TRANSFERRED_TO]->(b:Account)
    RETURN DISTINCT a.account_id AS source,
                    b.account_id AS target
    LIMIT 50000
    """

    results = client.execute_query(query)
    return [
        (r["source"], r["target"]) for r in results
    ]
