"""
Graphene — Neo4j Client

Production-quality Neo4j client with connection pooling, context manager
support, and custom error handling. Singleton pattern loaded from .env.

Usage:
    from graph.neo4j_client import get_client
    client = get_client()
    results = client.execute_query("MATCH (a:Account) RETURN a LIMIT 10")
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import (
    AuthError,
    Neo4jError,
    ServiceUnavailable,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "graphene2026"),
    "MAX_CONNECTION_POOL_SIZE": 10,
    "CONNECTION_TIMEOUT": 30,
}


class GrapheneDBError(Exception):
    """Custom exception for Graphene database operations.

    Wraps raw Neo4j driver exceptions with clear context messages.
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class Neo4jClient:
    """Neo4j client with connection pooling and context manager support.

    Provides a clean interface for executing read and write queries
    against the Neo4j graph database. Handles connection lifecycle
    and wraps all exceptions in GrapheneDBError.

    Attributes:
        uri: Neo4j connection URI.
        user: Neo4j username.
        password: Neo4j password.
        driver: Neo4j driver instance.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialise Neo4j driver with connection pool.

        Args:
            uri: Neo4j bolt URI. Defaults to env NEO4J_URI.
            user: Neo4j username. Defaults to env NEO4J_USER.
            password: Neo4j password. Defaults to env NEO4J_PASSWORD.

        Raises:
            GrapheneDBError: If connection cannot be established.
        """
        self.uri = uri or CONFIG["NEO4J_URI"]
        self.user = user or CONFIG["NEO4J_USER"]
        self.password = password or CONFIG["NEO4J_PASSWORD"]
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=CONFIG["MAX_CONNECTION_POOL_SIZE"],
                connection_timeout=CONFIG["CONNECTION_TIMEOUT"],
            )
            logger.info("Neo4j driver initialised for %s", self.uri)
        except AuthError as e:
            raise GrapheneDBError(
                f"Authentication failed for Neo4j at {self.uri}. "
                f"Check NEO4J_USER and NEO4J_PASSWORD in .env",
                original_error=e,
            ) from e
        except ServiceUnavailable as e:
            raise GrapheneDBError(
                f"Cannot connect to Neo4j at {self.uri}. "
                f"Ensure Neo4j is running.",
                original_error=e,
            ) from e

    def __enter__(self) -> "Neo4jClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing driver."""
        self.close()

    def close(self) -> None:
        """Properly close the Neo4j driver and release connections."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")
            self.driver = None

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict]:
        """Execute a read query and return results as list of dicts.

        Args:
            query: Cypher query string.
            params: Optional query parameters.

        Returns:
            List of result records as dictionaries.

        Raises:
            GrapheneDBError: If query execution fails.
        """
        if not self.driver:
            raise GrapheneDBError("Neo4j driver is not initialised.")

        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                records = [dict(record) for record in result]
                logger.debug(
                    "Query returned %d records", len(records)
                )
                return records
        except Neo4jError as e:
            logger.exception("Query execution failed: %s", query[:100])
            raise GrapheneDBError(
                f"Query execution failed: {e.message}",
                original_error=e,
            ) from e

    def execute_write(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict:
        """Execute a write query and return summary.

        Args:
            query: Cypher write query string.
            params: Optional query parameters.

        Returns:
            Dictionary with query counters (nodes_created, etc).

        Raises:
            GrapheneDBError: If write operation fails.
        """
        if not self.driver:
            raise GrapheneDBError("Neo4j driver is not initialised.")

        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                summary = result.consume()
                counters = summary.counters
                return {
                    "nodes_created": counters.nodes_created,
                    "nodes_deleted": counters.nodes_deleted,
                    "relationships_created": (
                        counters.relationships_created
                    ),
                    "relationships_deleted": (
                        counters.relationships_deleted
                    ),
                    "properties_set": counters.properties_set,
                }
        except Neo4jError as e:
            logger.exception(
                "Write operation failed: %s", query[:100]
            )
            raise GrapheneDBError(
                f"Write operation failed: {e.message}",
                original_error=e,
            ) from e

    def health_check(self) -> bool:
        """Verify connectivity to Neo4j.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j health check: OK")
            return True
        except ServiceUnavailable:
            logger.error("Neo4j health check: FAILED")
            return False


_client_instance: Neo4jClient | None = None


def get_client() -> Neo4jClient:
    """Get the module-level Neo4j client singleton.

    Creates a new client on first call, reuses on subsequent calls.

    Returns:
        The shared Neo4jClient instance.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = Neo4jClient()
    return _client_instance


def close_client() -> None:
    """Close the module-level Neo4j client singleton."""
    global _client_instance
    if _client_instance is not None:
        _client_instance.close()
        _client_instance = None
