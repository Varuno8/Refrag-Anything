"""Projection layers for mapping encoder outputs into decoder space."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PhiConfig:
    """Configuration for the projection network."""

    input_dim: int
    output_dim: int
    hidden_dim: int
    activation: str = "gelu"


class PhiProjector:
    """Lightweight two-layer MLP used to project chunk embeddings."""

    def __init__(
        self,
        config: PhiConfig,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.w1 = self._init_matrix(config.hidden_dim, config.input_dim)
        self.b1 = np.zeros(config.hidden_dim, dtype=np.float32)
        self.w2 = self._init_matrix(config.output_dim, config.hidden_dim)
        self.b2 = np.zeros(config.output_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a single embedding into decoder space."""

        hidden = self._activate(self.w1 @ embedding + self.b1)
        out = self.w2 @ hidden + self.b2
        return out.astype(np.float32, copy=False)

    def batch_project(self, embeddings: Iterable[np.ndarray]) -> np.ndarray:
        """Project a batch of embeddings."""

        stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in embeddings])
        hidden = self._activate(stacked @ self.w1.T + self.b1)
        out = hidden @ self.w2.T + self.b2
        return out.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    def _init_matrix(self, rows: int, cols: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (rows + cols))
        return self.rng.uniform(-limit, limit, size=(rows, cols)).astype(np.float32)

    def _activate(self, tensor: np.ndarray) -> np.ndarray:
        if self.config.activation == "relu":
            return np.maximum(tensor, 0.0, dtype=np.float32)
        if self.config.activation == "tanh":
            return np.tanh(tensor, dtype=np.float32)
        # default to GELU approximation
        return 0.5 * tensor * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (tensor + 0.044715 * np.power(tensor, 3)))
        )


__all__ = ["PhiConfig", "PhiProjector"]
