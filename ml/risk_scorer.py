"""
Graphene — Risk Scorer

Combines GNN predictions, Isolation Forest scores, and Cypher pattern
matches into a single 0-100 risk score per account. This is the score
displayed on the dashboard.

Risk Score = 0.40 × GNN + 0.35 × Anomaly + 0.25 × Rules
Risk Tiers: LOW (0-30), MEDIUM (31-55), HIGH (56-75), CRITICAL (76-100)
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "W_GNN": 0.40,
    "W_ISO": 0.35,
    "W_RULES": 0.25,
    "RULE_SCORES": {
        "CIRCULAR_ROUND_TRIP": 40,
        "LAYERING": 35,
        "STRUCTURING": 30,
        "DORMANT_ACTIVATION": 25,
        "PROFILE_MISMATCH": 20,
    },
    "TIERS": {
        "CRITICAL": (76, 100),
        "HIGH": (56, 75),
        "MEDIUM": (31, 55),
        "LOW": (0, 30),
    },
    "RECOMMENDATIONS": {
        "CRITICAL": "Freeze account immediately and file STR with FIU-IND.",
        "HIGH": "Immediate review required. Escalate to senior investigator.",
        "MEDIUM": "Enhanced due diligence. Monitor closely for 30 days.",
        "LOW": "Normal monitoring. No immediate action required.",
    },
}


def _compute_risk_tier(score: float) -> str:
    """Determine risk tier from a 0-100 score.

    Args:
        score: Risk score (0-100).

    Returns:
        Risk tier string (LOW/MEDIUM/HIGH/CRITICAL).
    """
    if score >= 76:
        return "CRITICAL"
    elif score >= 56:
        return "HIGH"
    elif score >= 31:
        return "MEDIUM"
    return "LOW"


def _compute_rule_score(
    account_id: str,
    pattern_results: dict[str, list],
) -> tuple[float, list[str]]:
    """Compute rule-based score from pattern detection results.

    Args:
        account_id: The account to score.
        pattern_results: Dict mapping pattern type to detection results.

    Returns:
        Tuple of (rule_score 0-100, list of triggered pattern names).
    """
    score = 0.0
    triggered: list[str] = []

    for pattern_type, detections in pattern_results.items():
        if not detections:
            continue

        for detection in detections:
            involved_accounts: set[str] = set()

            for key in [
                "account_ids", "sender_accounts",
                "intermediate_accounts",
            ]:
                val = detection.get(key)
                if isinstance(val, list):
                    involved_accounts.update(str(a) for a in val)

            for key in [
                "anchor_account", "source_account",
                "destination_account", "account_id",
            ]:
                val = detection.get(key)
                if val:
                    involved_accounts.add(str(val))

            if account_id in involved_accounts:
                rule_points = CONFIG["RULE_SCORES"].get(
                    pattern_type, 10
                )
                score += rule_points
                if pattern_type not in triggered:
                    triggered.append(pattern_type)

    return min(score, 100.0), triggered


class RiskScorer:
    """Combines ML scores and rule matches into a unified risk score.

    The risk scorer is the final output layer that merges:
    - GNN fraud probabilities (supervised, structural)
    - Anomaly detection scores (unsupervised, behavioral)
    - Deterministic rule matches (FATF AML typology rules)

    Into a single 0-100 score per account with a risk tier classification.
    """

    def compute_scores(
        self,
        gnn_results: pd.DataFrame,
        anomaly_results: pd.DataFrame,
        pattern_results: dict[str, list],
        account_ids: list[str],
    ) -> pd.DataFrame:
        """Compute risk scores for all accounts.

        Args:
            gnn_results: DataFrame with account_id, gnn_fraud_prob.
            anomaly_results: DataFrame with account_id,
                             ensemble_anomaly_score.
            pattern_results: Dict from run_all_pattern_queries.
            account_ids: List of all account IDs to score.

        Returns:
            DataFrame with per-account risk breakdown.
        """
        logger.info(
            "Computing risk scores for %d accounts...",
            len(account_ids),
        )

        gnn_map: dict[str, float] = {}
        if not gnn_results.empty:
            gnn_map = dict(
                zip(
                    gnn_results["account_id"],
                    gnn_results["gnn_fraud_prob"],
                )
            )

        anomaly_map: dict[str, float] = {}
        if not anomaly_results.empty:
            anomaly_map = dict(
                zip(
                    anomaly_results["account_id"],
                    anomaly_results["ensemble_anomaly_score"],
                )
            )

        results = []
        for acc_id in account_ids:
            gnn_prob = gnn_map.get(acc_id, 0.0)
            anomaly_score = anomaly_map.get(acc_id, 0.0)
            rule_score, triggered = _compute_rule_score(
                acc_id, pattern_results
            )

            gnn_contribution = CONFIG["W_GNN"] * gnn_prob * 100
            iso_contribution = CONFIG["W_ISO"] * anomaly_score * 100
            rule_contribution = CONFIG["W_RULES"] * rule_score

            risk_score = np.clip(
                gnn_contribution + iso_contribution + rule_contribution,
                0.0,
                100.0,
            )

            risk_tier = _compute_risk_tier(risk_score)

            results.append({
                "account_id": acc_id,
                "risk_score": round(float(risk_score), 2),
                "risk_tier": risk_tier,
                "gnn_contribution": round(float(gnn_contribution), 2),
                "iso_contribution": round(float(iso_contribution), 2),
                "rule_contribution": round(
                    float(rule_contribution), 2
                ),
                "triggered_patterns": triggered,
                "recommendation": CONFIG["RECOMMENDATIONS"].get(
                    risk_tier, ""
                ),
                "timestamp": datetime.now().isoformat(),
            })

        result_df = pd.DataFrame(results)

        tier_counts = result_df["risk_tier"].value_counts().to_dict()
        logger.info(
            "Risk scores computed — CRITICAL: %d, HIGH: %d, "
            "MEDIUM: %d, LOW: %d",
            tier_counts.get("CRITICAL", 0),
            tier_counts.get("HIGH", 0),
            tier_counts.get("MEDIUM", 0),
            tier_counts.get("LOW", 0),
        )

        return result_df

    def generate_alert(
        self,
        account_id: str,
        risk_df_row: dict[str, Any],
    ) -> dict:
        """Generate a structured alert from a risk score row.

        Args:
            account_id: The flagged account ID.
            risk_df_row: A single row from compute_scores output.

        Returns:
            Structured alert dictionary.
        """
        risk_score = risk_df_row.get("risk_score", 0)
        risk_tier = risk_df_row.get("risk_tier", "LOW")
        triggered = risk_df_row.get("triggered_patterns", [])
        recommendation = risk_df_row.get("recommendation", "")

        evidence_parts = []
        if risk_df_row.get("gnn_contribution", 0) > 10:
            evidence_parts.append(
                f"GNN model indicates {risk_df_row['gnn_contribution']:.0f}% "
                f"structural risk from account neighbourhood."
            )
        if risk_df_row.get("iso_contribution", 0) > 10:
            evidence_parts.append(
                f"Anomaly detection score contributes "
                f"{risk_df_row['iso_contribution']:.0f}% — "
                f"behavioral patterns deviate from baseline."
            )
        if triggered:
            patterns_str = ", ".join(triggered)
            evidence_parts.append(
                f"AML patterns detected: {patterns_str}."
            )

        evidence_summary = " ".join(evidence_parts) or (
            "Account flagged based on aggregate risk signals."
        )

        return {
            "alert_id": str(uuid4()),
            "account_id": account_id,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "triggered_patterns": triggered,
            "recommendation": recommendation,
            "evidence_summary": evidence_summary,
            "created_at": datetime.now().isoformat(),
            "status": "OPEN",
        }

    def generate_all_alerts(
        self, risk_df: pd.DataFrame, min_tier: str = "MEDIUM"
    ) -> list[dict]:
        """Generate alerts for all accounts above the given tier.

        Args:
            risk_df: DataFrame from compute_scores.
            min_tier: Minimum tier to generate alerts for.

        Returns:
            List of alert dictionaries.
        """
        tier_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_level = tier_order.get(min_tier, 2)

        alerts = []
        for _, row in risk_df.iterrows():
            tier_level = tier_order.get(row["risk_tier"], 0)
            if tier_level >= min_level:
                alert = self.generate_alert(
                    row["account_id"], row.to_dict()
                )
                alerts.append(alert)

        alerts.sort(
            key=lambda x: x["risk_score"], reverse=True
        )

        logger.info("Generated %d alerts.", len(alerts))
        return alerts
