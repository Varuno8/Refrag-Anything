"""Context planning utilities for the REFRAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from raganything.refrag.encoders import ChunkEncoder
from raganything.refrag.projection import PhiProjector
from raganything.refrag.cache import RefragCache

LOGGER = logging.getLogger(__name__)


@dataclass
class ContextPlan:
    """Data container for a prepared REFRAG context."""

    question: str
    question_tokens: List[int]
    chunk_nodes: List[Any]
    chunk_embeddings: np.ndarray
    projected: np.ndarray
    chunk_texts: List[str]
    chunk_scores: List[float] = field(default_factory=list)
    expand_mask: List[bool] = field(default_factory=list)

    def mark_expanded(self, index: int) -> None:
        if 0 <= index < len(self.expand_mask):
            self.expand_mask[index] = True

    def expanded_texts(self) -> List[str]:
        return [text for text, flag in zip(self.chunk_texts, self.expand_mask) if flag]

    def expanded_token_spans(self, tokenizer: Any) -> List[List[int]]:
        spans: List[List[int]] = []
        for text, flag in zip(self.chunk_texts, self.expand_mask):
            if not flag:
                continue
            spans.append(_tokenize(tokenizer, text))
        return spans


class ContextPlanner:
    """Build :class:`ContextPlan` objects from retrieval results."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        encoder: ChunkEncoder,
        projector: PhiProjector,
        cache: RefragCache,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.projector = projector
        self.cache = cache

    # ------------------------------------------------------------------
    def build_plan(self, query: str, retrieval_data: Dict[str, Any]) -> ContextPlan:
        nodes = list(self._extract_nodes(retrieval_data))
        embeddings: List[np.ndarray] = []
        projected: List[np.ndarray] = []
        texts: List[str] = []
        scores: List[float] = []

        for node in nodes:
            node_id = self._node_id(node)
            if node_id is not None:
                embedding = self.cache.get_or_set(node_id, lambda: self.encoder.encode(node))
                projection_key = f"phi::{node_id}"
                projected_vec = self.cache.get_or_set(
                    projection_key, lambda: self.projector.project(embedding)
                )
            else:
                embedding = self.encoder.encode(node)
                projected_vec = self.projector.project(embedding)

            embeddings.append(embedding)
            projected.append(projected_vec)
            texts.append(self._extract_text(node))
            scores.append(self._extract_score(node))

        question_tokens = _tokenize(self.tokenizer, query)
        expand_mask = [False] * len(nodes)

        return ContextPlan(
            question=query,
            question_tokens=question_tokens,
            chunk_nodes=nodes,
            chunk_embeddings=np.asarray(embeddings, dtype=np.float32)
            if embeddings
            else np.zeros((0, self.encoder.encoder_dim), dtype=np.float32),
            projected=np.asarray(projected, dtype=np.float32)
            if projected
            else np.zeros((0, self.projector.config.output_dim), dtype=np.float32),
            chunk_texts=texts,
            chunk_scores=scores,
            expand_mask=expand_mask,
        )

    # ------------------------------------------------------------------
    def _extract_nodes(self, retrieval_data: Dict[str, Any]) -> Iterable[Any]:
        if not retrieval_data:
            return []
        if isinstance(retrieval_data, list):
            return retrieval_data
        if "chunks" in retrieval_data and isinstance(retrieval_data["chunks"], Sequence):
            return retrieval_data["chunks"]
        if "retrieved_chunks" in retrieval_data and isinstance(
            retrieval_data["retrieved_chunks"], Sequence
        ):
            return retrieval_data["retrieved_chunks"]
        if "results" in retrieval_data and isinstance(
            retrieval_data["results"], Sequence
        ):
            return retrieval_data["results"]
        return []

    def _extract_text(self, node: Any) -> str:
        if isinstance(node, dict):
            for key in ("content", "text", "body", "chunk_text", "raw_text"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        else:
            for key in ("content", "text", "body", "chunk_text", "raw_text"):
                value = getattr(node, key, None)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def _extract_score(self, node: Any) -> float:
        if isinstance(node, dict):
            for key in ("score", "similarity", "weight"):
                value = node.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        else:
            for key in ("score", "similarity", "weight"):
                value = getattr(node, key, None)
                if isinstance(value, (int, float)):
                    return float(value)
        return float("nan")

    def _node_id(self, node: Any) -> str | None:
        if isinstance(node, dict):
            for key in ("id", "node_id", "chunk_id", "hash"):
                value = node.get(key)
                if isinstance(value, str):
                    return value
        else:
            for key in ("id", "node_id", "chunk_id"):
                value = getattr(node, key, None)
                if isinstance(value, str):
                    return value
        return None


def _tokenize(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text))
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Tokenizer.encode failed; falling back to naive tokenization")
    if callable(tokenizer):
        tokens = tokenizer(text)
        if isinstance(tokens, dict) and "input_ids" in tokens:
            return list(tokens["input_ids"])
        if isinstance(tokens, (list, tuple)):
            return [int(tok) for tok in tokens]
    # naive whitespace tokeniser fallback
    return [hash(token) % 100000 for token in text.split()]


__all__ = ["ContextPlan", "ContextPlanner"]
