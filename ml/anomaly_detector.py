"""
Graphene — Anomaly Detector

Ensemble anomaly detector combining Isolation Forest and Local Outlier Factor.
This is the UNSUPERVISED detector — finds anomalies without needing fraud labels,
critical for detecting novel fraud types not in training data.

Ensemble score: 0.6 × normalized(IsoForest) + 0.4 × normalized(LOF)
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "MODEL_PATH": str(
        Path(__file__).resolve().parent / "models" / "anomaly_detector.pkl"
    ),
    "ISO_N_ESTIMATORS": 300,
    "ISO_CONTAMINATION": 0.08,
    "LOF_N_NEIGHBORS": 20,
    "LOF_CONTAMINATION": 0.08,
    "ENSEMBLE_ISO_WEIGHT": 0.6,
    "ENSEMBLE_LOF_WEIGHT": 0.4,
    "ANOMALY_THRESHOLD": 0.65,
    "SEED": 42,
}


class GrapheneAnomalyDetector:
    """Ensemble anomaly detector for transaction fraud detection.

    Combines Isolation Forest and Local Outlier Factor scores
    into a single anomaly score per account. The ensemble approach
    captures both global outliers (Isolation Forest) and local
    density anomalies (LOF).

    Attributes:
        iso_forest: Isolation Forest model.
        lof: Local Outlier Factor model.
        scaler: StandardScaler for feature normalization.
        is_fitted: Whether the models have been trained.
    """

    def __init__(
        self,
        contamination: float = 0.08,
    ):
        """Initialise the anomaly detector ensemble.

        Args:
            contamination: Expected fraction of anomalies.
        """
        self.iso_forest = IsolationForest(
            n_estimators=CONFIG["ISO_N_ESTIMATORS"],
            contamination=contamination,
            max_samples="auto",
            random_state=CONFIG["SEED"],
            n_jobs=-1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=CONFIG["LOF_N_NEIGHBORS"],
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._feature_names: list[str] = []

    def fit(self, feature_df: pd.DataFrame) -> None:
        """Fit both anomaly detection models on the feature data.

        Args:
            feature_df: DataFrame with account features.
                        Must include 'account_id' column.

        Raises:
            ValueError: If feature_df is empty.
        """
        if feature_df.empty:
            raise ValueError("Cannot fit on empty DataFrame.")

        feature_cols = [
            c for c in feature_df.columns if c != "account_id"
        ]
        self._feature_names = feature_cols

        X = feature_df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X)

        logger.info(
            "Fitting Isolation Forest (%d estimators)...",
            CONFIG["ISO_N_ESTIMATORS"],
        )
        self.iso_forest.fit(X_scaled)

        logger.info(
            "Fitting LOF (n_neighbors=%d)...",
            CONFIG["LOF_N_NEIGHBORS"],
        )
        self.lof.fit(X_scaled)

        self.is_fitted = True

        os.makedirs(
            os.path.dirname(CONFIG["MODEL_PATH"]), exist_ok=True
        )
        joblib.dump(self, CONFIG["MODEL_PATH"])

        logger.info(
            "Anomaly detector fitted on %d samples, %d features. "
            "Saved to %s",
            X.shape[0],
            X.shape[1],
            CONFIG["MODEL_PATH"],
        )

    def predict(
        self, feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate anomaly scores and predictions.

        Args:
            feature_df: DataFrame with same features used for fitting.
                        Must include 'account_id' column.

        Returns:
            DataFrame with columns:
                - account_id
                - iso_score (raw, lower = more anomalous)
                - lof_score (raw, lower = more anomalous)
                - iso_is_anomaly (bool)
                - lof_is_anomaly (bool)
                - ensemble_anomaly_score (0-1, higher = suspicious)
                - ensemble_is_anomaly (bool)

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "AnomalyDetector not fitted. Call fit() first."
            )

        feature_cols = [
            c for c in feature_df.columns if c != "account_id"
        ]
        X = feature_df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        iso_scores = self.iso_forest.score_samples(X_scaled)
        iso_predictions = self.iso_forest.predict(X_scaled)

        lof_scores = self.lof.score_samples(X_scaled)
        lof_predictions = self.lof.predict(X_scaled)

        iso_norm = self._normalize_scores(iso_scores)
        lof_norm = self._normalize_scores(lof_scores)

        ensemble_score = (
            CONFIG["ENSEMBLE_ISO_WEIGHT"] * iso_norm
            + CONFIG["ENSEMBLE_LOF_WEIGHT"] * lof_norm
        )

        result = pd.DataFrame({
            "account_id": feature_df["account_id"].values,
            "iso_score": iso_scores,
            "lof_score": lof_scores,
            "iso_is_anomaly": iso_predictions == -1,
            "lof_is_anomaly": lof_predictions == -1,
            "ensemble_anomaly_score": ensemble_score,
            "ensemble_is_anomaly": (
                ensemble_score > CONFIG["ANOMALY_THRESHOLD"]
            ),
        })

        n_anomalies = result["ensemble_is_anomaly"].sum()
        logger.info(
            "Anomaly detection: %d/%d accounts flagged (%.1f%%)",
            n_anomalies,
            len(result),
            100.0 * n_anomalies / max(len(result), 1),
        )

        return result

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize raw anomaly scores to [0, 1] range.

        Lower raw scores → higher normalized scores (more anomalous).

        Args:
            scores: Raw anomaly scores from a detector.

        Returns:
            Normalized scores where higher = more suspicious.
        """
        min_s = scores.min()
        max_s = scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        normalized = (max_s - scores) / (max_s - min_s)
        return np.clip(normalized, 0.0, 1.0)

    @classmethod
    def load(
        cls,
        path: str | None = None,
    ) -> "GrapheneAnomalyDetector":
        """Load a fitted anomaly detector from disk.

        Args:
            path: Path to the saved model. Defaults to CONFIG path.

        Returns:
            Loaded GrapheneAnomalyDetector instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        model_path = path or CONFIG["MODEL_PATH"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Anomaly detector not found at {model_path}. "
                f"Run train.py first."
            )

        detector = joblib.load(model_path)
        logger.info("Anomaly detector loaded from %s", model_path)
        return detector


def explain_anomaly(
    detector: GrapheneAnomalyDetector,
    feature_df: pd.DataFrame,
    account_id: str,
) -> dict:
    """Generate SHAP explanation for why an account was flagged.

    Uses SHAP TreeExplainer on the Isolation Forest component to
    identify the top contributing features.

    Args:
        detector: Fitted GrapheneAnomalyDetector instance.
        feature_df: Full feature DataFrame with account_id column.
        account_id: The specific account to explain.

    Returns:
        Dictionary with explanation details:
        - account_id, top_features, anomaly_score, explanation_text
    """
    account_row = feature_df[
        feature_df["account_id"] == account_id
    ]
    if account_row.empty:
        return {
            "account_id": account_id,
            "top_features": [],
            "anomaly_score": 0.0,
            "explanation_text": "Account not found in feature data.",
        }

    feature_cols = [
        c for c in feature_df.columns if c != "account_id"
    ]
    X = account_row[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = detector.scaler.transform(X)

    iso_score = detector.iso_forest.score_samples(X_scaled)[0]
    anomaly_score = float(
        GrapheneAnomalyDetector._normalize_scores(
            np.array([iso_score])
        )[0]
    )

    try:
        import shap
        explainer = shap.TreeExplainer(detector.iso_forest)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = shap_values[0]

        feature_importance = list(zip(feature_cols, sv))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for fname, sval in feature_importance[:5]:
            actual_val = float(X[0][feature_cols.index(fname)])
            direction = (
                "increases" if sval > 0 else "decreases"
            )
            top_features.append({
                "feature_name": fname,
                "shap_value": round(float(sval), 4),
                "actual_value": round(actual_val, 2),
                "direction": f"{direction} risk",
            })
    except ImportError:
        logger.warning("SHAP not available for anomaly explanation.")
        top_features = [
            {
                "feature_name": fname,
                "shap_value": 0.0,
                "actual_value": float(X[0][i]),
                "direction": "unknown",
            }
            for i, fname in enumerate(feature_cols[:5])
        ]

    explanation_parts = []
    for feat in top_features[:3]:
        explanation_parts.append(
            f"{feat['feature_name']} ({feat['direction']}, "
            f"SHAP: {feat['shap_value']:.3f})"
        )

    explanation_text = (
        f"Account {account_id} flagged with anomaly score "
        f"{anomaly_score:.2f}. Key factors: "
        + "; ".join(explanation_parts)
        + "."
    )

    return {
        "account_id": account_id,
        "top_features": top_features,
        "anomaly_score": anomaly_score,
        "explanation_text": explanation_text,
    }
