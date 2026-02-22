"""
Graphene — Graph Builder

Converts raw transaction DataFrames into Neo4j node/relationship
creation queries. Used by the seeder and for dynamic graph updates.
"""

import logging
from typing import Any

import pandas as pd

from graph.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_account_nodes(
    client: Neo4jClient,
    accounts_df: pd.DataFrame,
    batch_size: int = 500,
) -> int:
    """Create Account nodes in Neo4j from a DataFrame.

    Args:
        client: Active Neo4jClient instance.
        accounts_df: DataFrame with account metadata.
        batch_size: Number of accounts per batch insert.

    Returns:
        Total number of accounts created.
    """
    logger.info(
        "Building %d account nodes...", len(accounts_df)
    )
    accounts = accounts_df.to_dict("records")
    total = 0

    for i in range(0, len(accounts), batch_size):
        batch = accounts[i: i + batch_size]
        client.execute_write(
            """
            UNWIND $accounts AS acc
            MERGE (a:Account {account_id: acc.account_id})
            SET a.account_type = acc.account_type,
                a.customer_name = acc.customer_name,
                a.customer_category = acc.customer_category,
                a.branch_code = acc.branch_code,
                a.kyc_status = acc.kyc_status,
                a.account_age_days = acc.account_age_days,
                a.risk_score = acc.risk_score,
                a.is_flagged = acc.is_flagged
            """,
            {"accounts": batch},
        )
        total += len(batch)

    logger.info("Created %d account nodes.", total)
    return total


def build_transaction_graph(
    client: Neo4jClient,
    txn_df: pd.DataFrame,
    batch_size: int = 500,
) -> dict[str, int]:
    """Create Transaction nodes and all relationships in Neo4j.

    Creates:
    - Transaction nodes with full attributes
    - SENT relationships (Account → Transaction)
    - RECEIVED_BY relationships (Transaction → Account)
    - TRANSFERRED_TO convenience relationships (Account → Account)

    Args:
        client: Active Neo4jClient instance.
        txn_df: DataFrame with transaction data.
        batch_size: Number of transactions per batch.

    Returns:
        Dictionary with creation counts.
    """
    logger.info(
        "Building transaction graph for %d transactions...",
        len(txn_df),
    )
    txn_df = txn_df.copy()
    txn_df["timestamp"] = pd.to_datetime(
        txn_df["timestamp"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    txn_df["fraud_type"] = txn_df["fraud_type"].fillna("")

    transactions = txn_df.to_dict("records")
    stats: dict[str, int] = {
        "transactions": 0,
        "sent_rels": 0,
        "received_rels": 0,
        "transfer_rels": 0,
    }

    for i in range(0, len(transactions), batch_size):
        batch = transactions[i: i + batch_size]
        n = len(batch)

        client.execute_write(
            """
            UNWIND $txns AS txn
            CREATE (t:Transaction {
                txn_id: txn.txn_id,
                timestamp: datetime(txn.timestamp),
                amount: txn.amount,
                txn_type: txn.txn_type,
                channel: txn.channel,
                status: txn.status,
                description: txn.description,
                is_fraud: txn.is_fraud,
                fraud_type: txn.fraud_type
            })
            """,
            {"txns": batch},
        )
        stats["transactions"] += n

        client.execute_write(
            """
            UNWIND $txns AS txn
            MATCH (s:Account {account_id: txn.sender_account})
            MATCH (t:Transaction {txn_id: txn.txn_id})
            CREATE (s)-[:SENT {
                amount: txn.amount,
                timestamp: datetime(txn.timestamp),
                txn_id: txn.txn_id
            }]->(t)
            """,
            {"txns": batch},
        )
        stats["sent_rels"] += n

        client.execute_write(
            """
            UNWIND $txns AS txn
            MATCH (t:Transaction {txn_id: txn.txn_id})
            MATCH (r:Account {
                account_id: txn.receiver_account
            })
            CREATE (t)-[:RECEIVED_BY {
                amount: txn.amount,
                timestamp: datetime(txn.timestamp)
            }]->(r)
            """,
            {"txns": batch},
        )
        stats["received_rels"] += n

        client.execute_write(
            """
            UNWIND $txns AS txn
            MATCH (s:Account {account_id: txn.sender_account})
            MATCH (r:Account {
                account_id: txn.receiver_account
            })
            CREATE (s)-[:TRANSFERRED_TO {
                txn_id: txn.txn_id,
                amount: txn.amount,
                timestamp: datetime(txn.timestamp),
                txn_type: txn.txn_type,
                is_fraud: txn.is_fraud
            }]->(r)
            """,
            {"txns": batch},
        )
        stats["transfer_rels"] += n

    logger.info(
        "Graph built: %d txns, %d SENT, %d RECEIVED_BY, "
        "%d TRANSFERRED_TO",
        stats["transactions"],
        stats["sent_rels"],
        stats["received_rels"],
        stats["transfer_rels"],
    )
    return stats


def update_account_risk(
    client: Neo4jClient,
    account_id: str,
    risk_score: float,
    is_flagged: bool,
) -> None:
    """Update an account's risk score and flagged status in Neo4j.

    Args:
        client: Active Neo4jClient instance.
        account_id: The account to update.
        risk_score: New risk score (0.0 to 1.0).
        is_flagged: Whether the account is flagged.
    """
    client.execute_write(
        """
        MATCH (a:Account {account_id: $account_id})
        SET a.risk_score = $risk_score,
            a.is_flagged = $is_flagged
        """,
        {
            "account_id": account_id,
            "risk_score": risk_score,
            "is_flagged": is_flagged,
        },
    )
    logger.debug(
        "Updated risk for %s: score=%.3f, flagged=%s",
        account_id,
        risk_score,
        is_flagged,
    )


def batch_update_risk_scores(
    client: Neo4jClient,
    risk_data: list[dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """Batch update risk scores for multiple accounts.

    Args:
        client: Active Neo4jClient instance.
        risk_data: List of dicts with account_id, risk_score, is_flagged.
        batch_size: Number of updates per batch.

    Returns:
        Number of accounts updated.
    """
    total = 0
    for i in range(0, len(risk_data), batch_size):
        batch = risk_data[i: i + batch_size]
        client.execute_write(
            """
            UNWIND $updates AS upd
            MATCH (a:Account {account_id: upd.account_id})
            SET a.risk_score = upd.risk_score,
                a.is_flagged = upd.is_flagged
            """,
            {"updates": batch},
        )
        total += len(batch)

    logger.info("Updated risk scores for %d accounts.", total)
    return total
