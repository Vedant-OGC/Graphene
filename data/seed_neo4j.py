"""
Graphene — Neo4j Database Seeder

Loads generated transaction and account data into Neo4j with the full
graph schema: Account nodes, Transaction nodes, and SENT, RECEIVED_BY,
TRANSFERRED_TO relationships.

Usage:
    python data/seed_neo4j.py
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "BATCH_SIZE": 500,
    "DATA_DIR": str(Path(__file__).resolve().parent),
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "graphene2026"),
}


def clear_database(driver) -> None:
    """Clear all existing data from Neo4j for demo repeatability.

    Args:
        driver: Neo4j driver instance.
    """
    logger.info("Clearing existing Neo4j data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared.")


def create_indexes(driver) -> None:
    """Create indexes and constraints for optimal query performance.

    Args:
        driver: Neo4j driver instance.
    """
    logger.info("Creating indexes and constraints...")
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT account_id_unique "
            "IF NOT EXISTS FOR (a:Account) "
            "REQUIRE a.account_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT txn_id_unique "
            "IF NOT EXISTS FOR (t:Transaction) "
            "REQUIRE t.txn_id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX txn_timestamp_index "
            "IF NOT EXISTS FOR (t:Transaction) "
            "ON (t.timestamp)"
        )
        session.run(
            "CREATE INDEX account_flagged_index "
            "IF NOT EXISTS FOR (a:Account) "
            "ON (a.is_flagged)"
        )
    logger.info("Indexes and constraints created.")


def load_accounts(driver, accounts_df: pd.DataFrame) -> int:
    """Load account nodes into Neo4j in batches.

    Args:
        driver: Neo4j driver instance.
        accounts_df: DataFrame with account data.

    Returns:
        Number of accounts loaded.
    """
    logger.info("Loading %d accounts...", len(accounts_df))
    accounts = accounts_df.to_dict("records")
    total_loaded = 0

    for i in range(0, len(accounts), CONFIG["BATCH_SIZE"]):
        batch = accounts[i: i + CONFIG["BATCH_SIZE"]]
        with driver.session() as session:
            session.run(
                """
                UNWIND $accounts AS acc
                CREATE (a:Account {
                    account_id: acc.account_id,
                    account_type: acc.account_type,
                    customer_name: acc.customer_name,
                    customer_category: acc.customer_category,
                    branch_code: acc.branch_code,
                    kyc_status: acc.kyc_status,
                    account_age_days: acc.account_age_days,
                    risk_score: acc.risk_score,
                    is_flagged: acc.is_flagged
                })
                """,
                accounts=batch,
            )
        total_loaded += len(batch)
        if total_loaded % 1000 == 0 or total_loaded == len(accounts):
            logger.info(
                "  Accounts loaded: %d / %d",
                total_loaded, len(accounts),
            )

    return total_loaded


def load_transactions(
    driver, txn_df: pd.DataFrame
) -> tuple[int, int]:
    """Load transactions and create all relationships in Neo4j.

    Creates:
    - Transaction nodes
    - (:Account)-[:SENT]->(:Transaction) relationships
    - (:Transaction)-[:RECEIVED_BY]->(:Account) relationships
    - (:Account)-[:TRANSFERRED_TO]->(:Account) convenience relationships

    Args:
        driver: Neo4j driver instance.
        txn_df: DataFrame with transaction data.

    Returns:
        Tuple of (transactions loaded, relationships created).
    """
    logger.info("Loading %d transactions...", len(txn_df))

    txn_df = txn_df.copy()
    txn_df["timestamp"] = pd.to_datetime(
        txn_df["timestamp"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    txn_df["fraud_type"] = txn_df["fraud_type"].fillna("")

    transactions = txn_df.to_dict("records")
    total_loaded = 0
    total_rels = 0

    for i in range(0, len(transactions), CONFIG["BATCH_SIZE"]):
        batch = transactions[i: i + CONFIG["BATCH_SIZE"]]

        with driver.session() as session:
            session.run(
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
                txns=batch,
            )

            session.run(
                """
                UNWIND $txns AS txn
                MATCH (sender:Account {account_id: txn.sender_account})
                MATCH (t:Transaction {txn_id: txn.txn_id})
                CREATE (sender)-[:SENT {
                    amount: txn.amount,
                    timestamp: datetime(txn.timestamp),
                    txn_id: txn.txn_id
                }]->(t)
                """,
                txns=batch,
            )

            session.run(
                """
                UNWIND $txns AS txn
                MATCH (t:Transaction {txn_id: txn.txn_id})
                MATCH (receiver:Account {
                    account_id: txn.receiver_account
                })
                CREATE (t)-[:RECEIVED_BY {
                    amount: txn.amount,
                    timestamp: datetime(txn.timestamp)
                }]->(receiver)
                """,
                txns=batch,
            )

            session.run(
                """
                UNWIND $txns AS txn
                MATCH (sender:Account {account_id: txn.sender_account})
                MATCH (receiver:Account {
                    account_id: txn.receiver_account
                })
                CREATE (sender)-[:TRANSFERRED_TO {
                    txn_id: txn.txn_id,
                    amount: txn.amount,
                    timestamp: datetime(txn.timestamp),
                    txn_type: txn.txn_type,
                    is_fraud: txn.is_fraud
                }]->(receiver)
                """,
                txns=batch,
            )

        total_loaded += len(batch)
        total_rels += len(batch) * 3

        if total_loaded % 1000 == 0 or total_loaded == len(transactions):
            logger.info(
                "  Progress: %d / %d transactions loaded",
                total_loaded,
                len(transactions),
            )

    return total_loaded, total_rels


def seed_database(
    csv_path: str | None = None,
    neo4j_uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Main seeder function — loads all data into Neo4j.

    Clears existing data, creates indexes, loads accounts and
    transactions with all relationships.

    Args:
        csv_path: Path to data directory with CSV files.
        neo4j_uri: Neo4j connection URI.
        user: Neo4j username.
        password: Neo4j password.
    """
    data_dir = csv_path or CONFIG["DATA_DIR"]
    uri = neo4j_uri or CONFIG["NEO4J_URI"]
    usr = user or CONFIG["NEO4J_USER"]
    pwd = password or CONFIG["NEO4J_PASSWORD"]

    txn_path = os.path.join(data_dir, "transactions.csv")
    acc_path = os.path.join(data_dir, "accounts.csv")

    if not os.path.exists(txn_path) or not os.path.exists(acc_path):
        logger.error(
            "CSV files not found in %s. "
            "Run generate_transactions.py first.",
            data_dir,
        )
        sys.exit(1)

    txn_df = pd.read_csv(txn_path)
    accounts_df = pd.read_csv(acc_path)

    logger.info(
        "Connecting to Neo4j at %s ...", uri
    )

    try:
        driver = GraphDatabase.driver(uri, auth=(usr, pwd))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully.")
    except ServiceUnavailable as e:
        logger.error(
            "Cannot connect to Neo4j at %s — is it running? %s",
            uri, e,
        )
        sys.exit(1)

    try:
        clear_database(driver)
        create_indexes(driver)
        n_accounts = load_accounts(driver, accounts_df)
        n_txns, n_rels = load_transactions(driver, txn_df)

        print("\n" + "=" * 60)
        print("  GRAPHENE — Neo4j Seeding Complete")
        print("=" * 60)
        print(f"  Seeded {n_accounts:,} accounts, "
              f"{n_txns:,} transactions, "
              f"{n_rels:,} relationships")
        print("=" * 60 + "\n")
    finally:
        driver.close()
        logger.info("Neo4j driver closed.")


if __name__ == "__main__":
    seed_database()
