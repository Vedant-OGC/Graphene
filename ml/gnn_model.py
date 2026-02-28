"""
Graphene — GraphSAGE GNN Model

Implements a 3-layer GraphSAGE-based Graph Neural Network for account-level
fraud classification. GraphSAGE handles inductive learning on unseen nodes,
essential for detecting new fraud accounts at inference time.

Architecture: SAGEConv(in→128) → SAGEConv(128→64) → SAGEConv(64→32) → Linear(32→2)
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "MODEL_PATH": str(
        Path(__file__).resolve().parent / "models" / "gnn_best.pt"
    ),
    "HIDDEN_DIM_1": 128,
    "HIDDEN_DIM_2": 64,
    "HIDDEN_DIM_3": 32,
    "DROPOUT_1": 0.3,
    "DROPOUT_2": 0.2,
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 5e-4,
    "PATIENCE": 15,
    "DEFAULT_EPOCHS": 100,
    "SEED": 42,
    "VAL_RATIO": 0.2,
    "PRINT_EVERY": 10,
}

torch.manual_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(
        "GPU detected: %s (%.1f GB VRAM) — training on CUDA.",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
    )
else:
    logger.info("No GPU found — training on CPU.")


try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning(
        "PyTorch Geometric not installed. "
        "Using fallback linear model."
    )


class GrapheneGNN(nn.Module):
    """3-layer GraphSAGE network for fraud node classification.

    Architecture:
        Layer 1: SAGEConv(in→128) + BatchNorm + ReLU + Dropout(0.3)
        Layer 2: SAGEConv(128→64) + BatchNorm + ReLU + Dropout(0.2)
        Layer 3: SAGEConv(64→32) + ReLU
        Output: Linear(32→2) → log_softmax

    Attributes:
        conv1, conv2, conv3: SAGEConv layers.
        bn1, bn2: BatchNorm layers.
        classifier: Final linear layer.
    """

    def __init__(self, in_channels: int):
        """Initialise the GraphSAGE model.

        Args:
            in_channels: Number of input features per node.
        """
        super().__init__()

        if HAS_PYG:
            self.conv1 = SAGEConv(
                in_channels, CONFIG["HIDDEN_DIM_1"], aggr="mean"
            )
            self.conv2 = SAGEConv(
                CONFIG["HIDDEN_DIM_1"], CONFIG["HIDDEN_DIM_2"], aggr="mean"
            )
            self.conv3 = SAGEConv(
                CONFIG["HIDDEN_DIM_2"], CONFIG["HIDDEN_DIM_3"], aggr="mean"
            )
        else:
            self.conv1 = nn.Linear(in_channels, CONFIG["HIDDEN_DIM_1"])
            self.conv2 = nn.Linear(
                CONFIG["HIDDEN_DIM_1"], CONFIG["HIDDEN_DIM_2"]
            )
            self.conv3 = nn.Linear(
                CONFIG["HIDDEN_DIM_2"], CONFIG["HIDDEN_DIM_3"]
            )

        self.bn1 = nn.BatchNorm1d(CONFIG["HIDDEN_DIM_1"])
        self.bn2 = nn.BatchNorm1d(CONFIG["HIDDEN_DIM_2"])
        self.classifier = nn.Linear(CONFIG["HIDDEN_DIM_3"], 2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the GNN.

        Args:
            x: Node feature tensor [num_nodes, in_channels].
            edge_index: Edge index tensor [2, num_edges].

        Returns:
            Log-softmax predictions [num_nodes, 2].
        """
        if HAS_PYG:
            h = self.conv1(x, edge_index)
        else:
            h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=CONFIG["DROPOUT_1"], training=self.training)

        if HAS_PYG:
            h = self.conv2(h, edge_index)
        else:
            h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=CONFIG["DROPOUT_2"], training=self.training)

        if HAS_PYG:
            h = self.conv3(h, edge_index)
        else:
            h = self.conv3(h)
        h = F.relu(h)

        out = self.classifier(h)
        return F.log_softmax(out, dim=1)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Get fraud probability for each node.

        Args:
            x: Node feature tensor.
            edge_index: Edge index tensor.

        Returns:
            Numpy array of fraud probabilities [0, 1] per node.
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, edge_index)
            probs = torch.exp(log_probs)
            return probs[:, 1].cpu().numpy()

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Get 32-dimensional node embeddings before classification.

        Args:
            x: Node feature tensor.
            edge_index: Edge index tensor.

        Returns:
            Numpy array of shape [num_nodes, 32].
        """
        self.eval()
        with torch.no_grad():
            if HAS_PYG:
                h = self.conv1(x, edge_index)
            else:
                h = self.conv1(x)
            h = self.bn1(h)
            h = F.relu(h)

            if HAS_PYG:
                h = self.conv2(h, edge_index)
            else:
                h = self.conv2(h)
            h = self.bn2(h)
            h = F.relu(h)

            if HAS_PYG:
                h = self.conv3(h, edge_index)
            else:
                h = self.conv3(h)
            h = F.relu(h)

            return h.cpu().numpy()


def _build_pyg_data(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    edge_list: list[tuple],
) -> Any:
    """Convert DataFrame and edge list to PyTorch Geometric Data.

    Args:
        feature_df: Feature matrix with account_id column.
        labels: Series indexed by account_id with boolean fraud labels.
        edge_list: List of (source_id, target_id) tuples.

    Returns:
        PyTorch Geometric Data object.
    """
    account_ids = feature_df["account_id"].tolist()
    id_to_idx = {aid: i for i, aid in enumerate(account_ids)}

    feature_cols = [
        c for c in feature_df.columns if c != "account_id"
    ]
    x = torch.tensor(
        feature_df[feature_cols].values, dtype=torch.float
    )

    y = torch.zeros(len(account_ids), dtype=torch.long)
    for aid, idx in id_to_idx.items():
        if aid in labels.index:
            y[idx] = int(labels[aid])

    edges_src = []
    edges_tgt = []
    for src, tgt in edge_list:
        if src in id_to_idx and tgt in id_to_idx:
            edges_src.append(id_to_idx[src])
            edges_tgt.append(id_to_idx[tgt])

    if not edges_src:
        edges_src = list(range(len(account_ids)))
        edges_tgt = list(range(len(account_ids)))

    edge_index = torch.tensor(
        [edges_src, edges_tgt], dtype=torch.long
    )

    if HAS_PYG:
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(account_ids))

    if HAS_PYG:
        data = Data(x=x, edge_index=edge_index, y=y)
    else:
        data = type("Data", (), {
            "x": x, "edge_index": edge_index, "y": y,
            "num_nodes": len(account_ids),
        })()

    return data


def train_gnn(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    edge_list: list[tuple],
    epochs: int = 100,
) -> GrapheneGNN:
    """Train the GraphSAGE GNN on the transaction graph.

    Handles class imbalance with weighted loss function.
    Uses early stopping based on validation AUC-ROC.

    Args:
        feature_df: Feature matrix with account_id column.
        labels: Series indexed by account_id with fraud labels.
        edge_list: List of (source_id, target_id) tuples.
        epochs: Number of training epochs.

    Returns:
        Trained GrapheneGNN model.
    """
    logger.info("Preparing training data...")

    data = _build_pyg_data(feature_df, labels, edge_list)
    n_features = data.x.shape[1]
    n_nodes = data.x.shape[0]

    indices = list(range(n_nodes))
    labels_list = data.y.numpy().tolist()

    train_idx, val_idx = train_test_split(
        indices,
        test_size=CONFIG["VAL_RATIO"],
        stratify=labels_list,
        random_state=CONFIG["SEED"],
    )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(DEVICE)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(DEVICE)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    # Move graph data to GPU
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    y = data.y.to(DEVICE)

    n_fraud = y.sum().item()
    n_normal = n_nodes - n_fraud
    if n_fraud > 0:
        weight = torch.tensor(
            [1.0, n_normal / n_fraud], dtype=torch.float
        ).to(DEVICE)
    else:
        weight = torch.tensor([1.0, 1.0], dtype=torch.float).to(DEVICE)

    model = GrapheneGNN(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    criterion = nn.NLLLoss(weight=weight)

    best_auc = 0.0
    patience_counter = 0
    best_state = None

    logger.info(
        "Training GNN on %s: %d nodes, %d features, %d edges, "
        "fraud ratio: %.1f%%",
        str(DEVICE).upper(),
        n_nodes,
        n_features,
        data.edge_index.shape[1],
        100.0 * n_fraud / max(n_nodes, 1),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % CONFIG["PRINT_EVERY"] == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index)
                val_probs = torch.exp(val_out[val_mask])[:, 1].cpu().numpy()
                val_labels = y[val_mask].cpu().numpy()

                try:
                    val_auc = roc_auc_score(val_labels, val_probs)
                except ValueError:
                    val_auc = 0.5

                logger.info(
                    "Epoch %3d/%d — Loss: %.4f | Val AUC: %.4f",
                    epoch, epochs, loss.item(), val_auc,
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    best_state = {
                        k: v.clone()
                        for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= CONFIG["PATIENCE"]:
                    logger.info(
                        "Early stopping at epoch %d "
                        "(best AUC: %.4f)",
                        epoch, best_auc,
                    )
                    break

    if best_state:
        model.load_state_dict(best_state)

    # Move back to CPU for saving (portable checkpoint)
    model_cpu = model.cpu()

    os.makedirs(
        os.path.dirname(CONFIG["MODEL_PATH"]), exist_ok=True
    )
    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "in_channels": n_features,
            "best_auc": best_auc,
        },
        CONFIG["MODEL_PATH"],
    )
    logger.info(
        "GNN saved to %s (best AUC: %.4f)",
        CONFIG["MODEL_PATH"], best_auc,
    )

    return model_cpu


def load_and_predict(
    feature_df: pd.DataFrame,
    edge_list: list[tuple],
) -> pd.DataFrame:
    """Load trained GNN and generate predictions.

    Args:
        feature_df: Feature matrix with account_id column.
        edge_list: List of (source_id, target_id) tuples.

    Returns:
        DataFrame with account_id, gnn_fraud_prob, gnn_embedding_2d.
    """
    logger.info("Loading GNN model for prediction on %s...", str(DEVICE).upper())

    checkpoint = torch.load(
        CONFIG["MODEL_PATH"], map_location=DEVICE, weights_only=False
    )
    in_channels = checkpoint["in_channels"]

    model = GrapheneGNN(in_channels).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_placeholder = pd.Series(
        False,
        index=feature_df["account_id"],
    )
    data = _build_pyg_data(
        feature_df, labels_placeholder, edge_list
    )

    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)

    fraud_probs = model.predict_proba(x, edge_index)
    embeddings = model.get_node_embeddings(x, edge_index)

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, random_state=CONFIG["SEED"]
        )
        embeddings_2d = reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning(
            "UMAP not available. Using PCA for 2D projection."
        )
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=CONFIG["SEED"])
        embeddings_2d = pca.fit_transform(embeddings)

    result_df = pd.DataFrame({
        "account_id": feature_df["account_id"].values,
        "gnn_fraud_prob": fraud_probs,
        "gnn_embed_x": embeddings_2d[:, 0],
        "gnn_embed_y": embeddings_2d[:, 1],
    })

    logger.info(
        "GNN predictions: %d accounts, "
        "mean fraud prob: %.4f, "
        "max fraud prob: %.4f",
        len(result_df),
        result_df["gnn_fraud_prob"].mean(),
        result_df["gnn_fraud_prob"].max(),
    )

    return result_df
