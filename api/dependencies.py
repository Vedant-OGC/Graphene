"""
Graphene — Shared Dependencies

Dependency injection for shared resources: Neo4j client, ML models,
cached pattern results, and risk scores. Loaded at startup and
injected into route handlers via FastAPI's dependency system.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from graph.neo4j_client import Neo4jClient, GrapheneDBError

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "PATTERN_CACHE": str(
        Path(__file__).resolve().parent.parent
        / "ml" / "models" / "pattern_cache.json"
    ),
    "RISK_SCORES_CACHE": str(
        Path(__file__).resolve().parent.parent
        / "ml" / "models" / "risk_scores.json"
    ),
}


class AppState:
    """Shared application state loaded at startup.

    Holds references to the Neo4j client, ML models, cached
    pattern results, and computed risk scores/alerts.

    Attributes:
        neo4j_client: Active Neo4j connection.
        pattern_results: Cached AML pattern detection results.
        risk_scores: Cached risk scores per account.
        alerts: Generated alerts sorted by risk score.
        models_loaded: Whether ML models are loaded.
    """

    def __init__(self):
        self.neo4j_client: Neo4jClient | None = None
        self.pattern_results: dict[str, list] = {}
        self.risk_scores: list[dict] = []
        self.alerts: list[dict] = []
        self.models_loaded: bool = False
        self._reports: dict[str, dict] = {}

    def initialize(self) -> None:
        """Load all resources needed for the API.

        Connects to Neo4j and loads cached ML results from disk.
        """
        logger.info("Initialising application state...")

        try:
            self.neo4j_client = Neo4jClient()
            if self.neo4j_client.health_check():
                logger.info("Neo4j connection: OK")
            else:
                logger.warning("Neo4j health check failed.")
        except GrapheneDBError as e:
            logger.error("Neo4j connection failed: %s", e)
            self.neo4j_client = None

        if os.path.exists(CONFIG["PATTERN_CACHE"]):
            with open(CONFIG["PATTERN_CACHE"], "r") as f:
                self.pattern_results = json.load(f)
            logger.info(
                "Loaded pattern cache: %d pattern types",
                len(self.pattern_results),
            )

        if os.path.exists(CONFIG["RISK_SCORES_CACHE"]):
            with open(CONFIG["RISK_SCORES_CACHE"], "r") as f:
                data = json.load(f)
                self.risk_scores = data.get("scores", [])
                self.alerts = data.get("alerts", [])
            logger.info(
                "Loaded %d risk scores, %d alerts",
                len(self.risk_scores),
                len(self.alerts),
            )

        self.models_loaded = True
        logger.info(
            "Graphene API ready — %d accounts loaded, "
            "%d alerts generated",
            len(self.risk_scores),
            len(self.alerts),
        )

    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        if self.neo4j_client:
            self.neo4j_client.close()
            logger.info("Neo4j connection closed.")

    def get_alert_by_id(
        self, alert_id: str
    ) -> dict | None:
        """Find an alert by its ID.

        Args:
            alert_id: The alert UUID to find.

        Returns:
            Alert dict or None if not found.
        """
        for alert in self.alerts:
            if alert.get("alert_id") == alert_id:
                return alert
        return None

    def get_account_metadata(
        self, account_id: str
    ) -> dict:
        """Fetch account metadata from Neo4j.

        Args:
            account_id: The account ID.

        Returns:
            Account metadata dictionary.
        """
        if not self.neo4j_client:
            return {"account_id": account_id}

        try:
            results = self.neo4j_client.execute_query(
                """
                MATCH (a:Account {account_id: $id})
                RETURN a.account_id AS account_id,
                       a.account_type AS account_type,
                       a.customer_name AS customer_name,
                       a.customer_category AS customer_category,
                       a.branch_code AS branch_code,
                       a.kyc_status AS kyc_status,
                       a.account_age_days AS account_age_days,
                       a.risk_score AS risk_score,
                       a.is_flagged AS is_flagged
                """,
                {"id": account_id},
            )
            if results:
                return results[0]
        except Exception as e:
            logger.warning(
                "Failed to get metadata for %s: %s",
                account_id, e,
            )

        return {"account_id": account_id}

    def get_recent_transactions(
        self, account_id: str, limit: int = 10
    ) -> list[dict]:
        """Fetch recent transactions for an account.

        Args:
            account_id: The account ID.
            limit: Maximum transactions to return.

        Returns:
            List of recent transaction dicts.
        """
        if not self.neo4j_client:
            return []

        try:
            return self.neo4j_client.execute_query(
                """
                MATCH (a:Account {account_id: $id})
                      -[r:TRANSFERRED_TO]-(other:Account)
                RETURN r.txn_id AS txn_id,
                       r.amount AS amount,
                       r.txn_type AS txn_type,
                       toString(r.timestamp) AS timestamp,
                       other.account_id AS counterparty,
                       r.is_fraud AS is_fraud
                ORDER BY r.timestamp DESC
                LIMIT $limit
                """,
                {"id": account_id, "limit": limit},
            )
        except Exception as e:
            logger.warning(
                "Failed to get transactions for %s: %s",
                account_id, e,
            )
            return []

    def store_report(
        self, report_id: str, report_data: dict
    ) -> None:
        """Store a generated report in memory.

        Args:
            report_id: Report UUID.
            report_data: Report metadata and file path.
        """
        self._reports[report_id] = report_data

    def get_report(self, report_id: str) -> dict | None:
        """Retrieve a stored report.

        Args:
            report_id: Report UUID.

        Returns:
            Report data dict or None.
        """
        return self._reports.get(report_id)


app_state = AppState()


def get_app_state() -> AppState:
    """Dependency injection function for route handlers.

    Returns:
        The shared AppState singleton.
    """
    return app_state
