"""Model inference for live trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from trading.config import DataConfig


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration (must match training)."""
    n_fundamental_features: int = 19
    n_price_features: int = 9
    n_embedding_dim: int = 768
    fundamental_latent: int = 32
    price_latent: int = 16
    news_latent: int = 32
    fundamental_dropout: float = 0.2
    price_dropout: float = 0.2
    news_dropout: float = 0.3
    news_alpha: float = 0.8
    # Training params (not used for inference)
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    n_epochs: int = 20
    pairs_per_day: int = 5000
    hard_fraction: float = 0.0


class MultiBranchRanker(nn.Module):
    """Multi-branch model with influence-controlled news embedding.

    Must match the architecture used in training (2.0 model.ipynb).
    """

    def __init__(
        self,
        n_fundamental_features: int = 19,
        n_price_features: int = 9,
        n_embedding_dim: int = 768,
        fundamental_latent: int = 32,
        price_latent: int = 16,
        news_latent: int = 32,
        news_alpha: float = 0.8,
    ):
        super().__init__()
        self.news_alpha = news_alpha

        # Fundamentals encoder
        self.fund_encoder = nn.Sequential(
            nn.Linear(n_fundamental_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, fundamental_latent),
            nn.ReLU(),
        )

        # Price encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(n_price_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, price_latent),
            nn.ReLU(),
        )

        # News encoder
        self.news_encoder = nn.Sequential(
            nn.Linear(n_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, news_latent),
            nn.ReLU(),
        )

        # Output head
        fused_dim = fundamental_latent + price_latent + news_latent
        self.output_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        price: torch.Tensor,
        fund: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning score."""
        h_f = self.fund_encoder(fund)
        h_p = self.price_encoder(price)
        h_n = self.news_encoder(emb)

        # Apply news influence cap
        h_n_scaled = self.news_alpha * h_n

        # Fuse and output
        h = torch.cat([h_f, h_p, h_n_scaled], dim=-1)
        score = self.output_head(h)
        return score.squeeze(-1)


class ModelInference:
    """Handles model loading and inference."""

    def __init__(self, model_path: Path | str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        # Load checkpoint - handle ModelConfig from __main__ namespace
        import sys
        # Register ModelConfig in __main__ so unpickling works
        if "ModelConfig" not in dir(sys.modules.get("__main__", {})):
            import __main__
            __main__.ModelConfig = ModelConfig
        self.checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Extract config and column names
        self.config = self.checkpoint.get("config")
        self.price_cols = self.checkpoint.get("price_cols", [])
        self.fund_cols = self.checkpoint.get("fund_cols", [])
        self.emb_cols = self.checkpoint.get("emb_cols", [])

        # Build model
        self.model = self._build_model()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"  Price features: {len(self.price_cols)}")
        logger.info(f"  Fund features: {len(self.fund_cols)}")
        logger.info(f"  Embedding dims: {len(self.emb_cols)}")

    def _build_model(self) -> MultiBranchRanker:
        """Build model with config from checkpoint."""
        if self.config is not None:
            model = MultiBranchRanker(
                n_fundamental_features=self.config.n_fundamental_features,
                n_price_features=self.config.n_price_features,
                n_embedding_dim=self.config.n_embedding_dim,
                fundamental_latent=self.config.fundamental_latent,
                price_latent=self.config.price_latent,
                news_latent=self.config.news_latent,
                news_alpha=self.config.news_alpha,
            )
        else:
            # Default config
            model = MultiBranchRanker(
                n_fundamental_features=len(self.fund_cols) or 19,
                n_price_features=len(self.price_cols) or 9,
                n_embedding_dim=len(self.emb_cols) or 768,
            )

        return model.to(self.device)

    @torch.no_grad()
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score all rows in dataframe, return scores."""
        # Extract features
        price_features = df[self.price_cols].values.astype(np.float32)
        fund_features = df[self.fund_cols].values.astype(np.float32)

        # Handle embeddings (may not all be present)
        if all(c in df.columns for c in self.emb_cols):
            emb_features = df[self.emb_cols].values.astype(np.float32)
        else:
            # Fill missing with zeros
            emb_features = np.zeros((len(df), len(self.emb_cols)), dtype=np.float32)
            for i, col in enumerate(self.emb_cols):
                if col in df.columns:
                    emb_features[:, i] = df[col].values

        # Convert to tensors
        price_t = torch.tensor(price_features, device=self.device)
        fund_t = torch.tensor(fund_features, device=self.device)
        emb_t = torch.tensor(emb_features, device=self.device)

        # Handle NaN by filling with 0
        price_t = torch.nan_to_num(price_t, 0.0)
        fund_t = torch.nan_to_num(fund_t, 0.0)
        emb_t = torch.nan_to_num(emb_t, 0.0)

        # Inference
        scores = self.model(price_t, fund_t, emb_t)
        return scores.cpu().numpy()

    def rank_stocks(
        self,
        df: pd.DataFrame,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Score and rank stocks. ascending=True for bottom-K (shorts)."""
        df = df.copy()
        df["score"] = self.score(df)
        df["rank"] = df["score"].rank(ascending=ascending)
        return df.sort_values("rank")

    def get_bottom_k(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Get bottom-K stocks (for shorting)."""
        ranked = self.rank_stocks(df, ascending=True)
        return ranked.head(k)

    def get_top_k(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Get top-K stocks (for going long)."""
        ranked = self.rank_stocks(df, ascending=False)
        return ranked.head(k)
