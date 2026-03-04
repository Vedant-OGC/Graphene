"""
Graphene — Training Pipeline

CLI-runnable script that orchestrates the full ML training pipeline:
data loading → feature extraction → GNN training → anomaly detection →
pattern queries → risk scoring.

Usage:
    python ml/train.py --epochs 100          # full training
    python ml/train.py --epochs 30 --fast    # demo mode (cached features)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jClient, GrapheneDBError
from graph.pattern_queries import run_all_pattern_queries
from ml.anomaly_detector import GrapheneAnomalyDetector
from ml.feature_engineering import (
    get_edge_list,
    get_full_feature_matrix,
    get_labels,
)
from ml.gnn_model import train_gnn, load_and_predict
from ml.risk_scorer import RiskScorer

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "MODELS_DIR": str(
        Path(__file__).resolve().parent / "models"
    ),
    "PATTERN_CACHE": str(
        Path(__file__).resolve().parent / "models" / "pattern_cache.json"
    ),
    "RISK_SCORES_CACHE": str(
        Path(__file__).resolve().parent / "models" / "risk_scores.json"
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Graphene ML Training Pipeline",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of GNN training epochs (default: 100)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use cached features if available (demo mode)",
    )
    return parser.parse_args()


def run_training(epochs: int = 100, fast: bool = False) -> dict:
    """Execute the full training pipeline.

    Steps:
    1. Connect to Neo4j
    2. Extract features (or load from cache)
    3. Train GNN model
    4. Train Anomaly Detector
    5. Run all pattern queries
    6. Compute risk scores for all accounts
    7. Save all results

    Args:
        epochs: Number of GNN training epochs.
        fast: Whether to use cached features.

    Returns:
        Dictionary with training summary metrics.
    """
    pipeline_start = time.perf_counter()
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)

    logger.info("=" * 60)
    logger.info("  GRAPHENE — ML Training Pipeline")
    logger.info("  Epochs: %d | Fast mode: %s", epochs, fast)
    logger.info("=" * 60)

    logger.info("[1/6] Connecting to Neo4j...")
    client = None
    for attempt in range(1, 4):
        try:
            client = Neo4jClient()
            if client.health_check():
                logger.info("  Neo4j connection OK.")
                break
            logger.warning(
                "  Health check failed (attempt %d/3). Retrying...",
                attempt,
            )
            client.close()
            client = None
        except GrapheneDBError as e:
            logger.warning(
                "  Connection error (attempt %d/3): %s", attempt, e
            )
            client = None
        time.sleep(3)

    if client is None:
        logger.error(
            "Cannot connect to Neo4j after 3 attempts. "
            "Is the DBMS running in Neo4j Desktop?"
        )
        sys.exit(1)

    try:
        logger.info("[2/6] Extracting features...")
        feature_df, feature_names = get_full_feature_matrix(
            client, use_cache=fast
        )
        if feature_df.empty:
            logger.error("No features extracted. Is the database seeded?")
            sys.exit(1)

        labels = get_labels(client)
        edge_list = get_edge_list(client)
        account_ids = feature_df["account_id"].tolist()

        logger.info(
            "  Features: %d accounts × %d features",
            len(feature_df),
            len(feature_names),
        )
        logger.info(
            "  Labels: %d fraud / %d total (%.1f%%)",
            labels.sum(),
            len(labels),
            100.0 * labels.sum() / max(len(labels), 1),
        )
        logger.info("  Edges: %d", len(edge_list))

        logger.info("[3/6] Training GNN model...")
        gnn_model = train_gnn(
            feature_df, labels, edge_list, epochs=epochs
        )

        gnn_results = load_and_predict(feature_df, edge_list)
        gnn_auc = 0.0
        try:
            from sklearn.metrics import (
                roc_auc_score,
                precision_score,
                recall_score,
            )
            aligned_labels = labels.reindex(
                gnn_results["account_id"]
            ).fillna(False).astype(int)
            gnn_auc = roc_auc_score(
                aligned_labels, gnn_results["gnn_fraud_prob"]
            )
            gnn_preds = (
                gnn_results["gnn_fraud_prob"] > 0.5
            ).astype(int)
            gnn_precision = precision_score(
                aligned_labels, gnn_preds, zero_division=0
            )
            gnn_recall = recall_score(
                aligned_labels, gnn_preds, zero_division=0
            )
        except Exception as e:
            logger.warning("Could not compute GNN metrics: %s", e)
            gnn_auc = 0.0
            gnn_precision = 0.0
            gnn_recall = 0.0

        logger.info("[4/6] Training Anomaly Detector...")
        detector = GrapheneAnomalyDetector()
        detector.fit(feature_df)
        anomaly_results = detector.predict(feature_df)

        n_anomalies = anomaly_results["ensemble_is_anomaly"].sum()

        logger.info("[5/6] Running pattern detection queries...")
        pattern_results = run_all_pattern_queries(client)

        serializable_patterns = {}
        for k, v in pattern_results.items():
            serializable_patterns[k] = [
                _make_serializable(item) for item in v
            ]

        with open(CONFIG["PATTERN_CACHE"], "w") as f:
            json.dump(serializable_patterns, f, indent=2, default=str)
        logger.info(
            "  Pattern cache saved to %s",
            CONFIG["PATTERN_CACHE"],
        )

        logger.info("[6/6] Computing risk scores...")
        scorer = RiskScorer()
        risk_df = scorer.compute_scores(
            gnn_results, anomaly_results,
            pattern_results, account_ids,
        )

        alerts = scorer.generate_all_alerts(risk_df, min_tier="MEDIUM")

        risk_output = {
            "scores": risk_df.to_dict("records"),
            "alerts": alerts,
        }
        with open(CONFIG["RISK_SCORES_CACHE"], "w") as f:
            json.dump(risk_output, f, indent=2, default=str)
        logger.info(
            "  Risk scores saved to %s",
            CONFIG["RISK_SCORES_CACHE"],
        )

        elapsed = time.perf_counter() - pipeline_start

        summary = {
            "gnn_auc_roc": round(gnn_auc, 4),
            "gnn_precision": round(gnn_precision, 4),
            "gnn_recall": round(gnn_recall, 4),
            "anomalies_flagged": int(n_anomalies),
            "patterns": {
                k: len(v) for k, v in pattern_results.items()
            },
            "alerts_critical": len(
                [a for a in alerts if a["risk_tier"] == "CRITICAL"]
            ),
            "alerts_high": len(
                [a for a in alerts if a["risk_tier"] == "HIGH"]
            ),
            "alerts_medium": len(
                [a for a in alerts if a["risk_tier"] == "MEDIUM"]
            ),
            "total_alerts": len(alerts),
            "training_time_seconds": round(elapsed, 1),
        }

        _print_summary(summary)
        return summary

    finally:
        if client is not None:
            client.close()


def _make_serializable(obj):
    """Convert Neo4j types to JSON-serializable Python types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _print_summary(summary: dict) -> None:
    """Print a formatted training summary.

    Args:
        summary: Training results dictionary.
    """
    print("\n" + "=" * 60)
    print("  GRAPHENE — Training Complete")
    print("=" * 60)
    print(
        f"  GNN AUC-ROC: {summary['gnn_auc_roc']:.3f} | "
        f"Precision: {summary['gnn_precision']:.2f} | "
        f"Recall: {summary['gnn_recall']:.2f}"
    )
    print(
        f"  Anomaly Detector: "
        f"{summary['anomalies_flagged']} accounts flagged"
    )

    patterns = summary.get("patterns", {})
    parts = []
    for name, count in patterns.items():
        short = name.lower().replace("_", " ")
        parts.append(f"{count} {short}")
    print(f"  Patterns detected: {', '.join(parts)}")

    print(
        f"  Total CRITICAL alerts: {summary['alerts_critical']} | "
        f"HIGH: {summary['alerts_high']} | "
        f"MEDIUM: {summary['alerts_medium']}"
    )
    print(
        f"  Training time: "
        f"{summary['training_time_seconds']:.1f}s"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_training(epochs=args.epochs, fast=args.fast)
