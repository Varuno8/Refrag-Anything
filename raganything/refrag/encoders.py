"""Multimodal chunk encoders used by the REFRAG runtime."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class EncoderResources:
    """Container for optional encoder callables."""

    text_encoder: Optional[Callable[[str], np.ndarray]] = None
    image_encoder: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None
    table_encoder: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None
    equation_encoder: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None


class ChunkEncoder:
    """Encode heterogeneous retrieval chunks into dense vectors.

    The implementation favours production safety: external model dependencies are
    optional and every modality has a deterministic fallback hashing encoder.  The
    resulting vectors are returned as ``np.float32`` arrays and share a common
    dimensionality ``encoder_dim`` so they can be consumed by the projector.
    """

    def __init__(
        self,
        encoder_dim: int,
        resources: Optional[EncoderResources] = None,
        *,
        normalize: bool = True,
        embedding_func: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.encoder_dim = encoder_dim
        self.normalize = normalize
        self.resources = resources or EncoderResources()
        self.embedding_func = embedding_func

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def encode(self, node: Dict[str, Any]) -> np.ndarray:
        """Encode a chunk node into a dense embedding."""

        modality = self._detect_modality(node)
        if modality == "image":
            vector = self._encode_image(node)
        elif modality == "table":
            vector = self._encode_table(node)
        elif modality == "equation":
            vector = self._encode_equation(node)
        else:
            vector = self._encode_text(node)

        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # modality helpers
    # ------------------------------------------------------------------
    def _encode_text(self, node: Dict[str, Any]) -> np.ndarray:
        text = self._extract_text(node)
        if not text:
            return self._hash_embedding("EMPTY_TEXT")

        if self.resources.text_encoder is not None:
            try:
                encoding = self.resources.text_encoder(text)
                return self._coerce_vector(encoding)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("Text encoder failed; falling back to hashing encoder")

        if self.embedding_func is not None:
            try:
                embedding = self.embedding_func(text)
                return self._coerce_vector(embedding)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "Custom embedding_func failed; falling back to hashing encoder"
                )

        return self._hash_embedding(text)

    def _encode_image(self, node: Dict[str, Any]) -> np.ndarray:
        if self.resources.image_encoder is not None:
            try:
                return self._coerce_vector(self.resources.image_encoder(node))
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("Image encoder failed; falling back to hashing encoder")

        source = self._extract_image_source(node)
        return self._hash_embedding(source or "UNKNOWN_IMAGE")

    def _encode_table(self, node: Dict[str, Any]) -> np.ndarray:
        if self.resources.table_encoder is not None:
            try:
                return self._coerce_vector(self.resources.table_encoder(node))
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("Table encoder failed; falling back to hashing encoder")

        text = self._extract_table_text(node)
        return self._hash_embedding(text or "EMPTY_TABLE")

    def _encode_equation(self, node: Dict[str, Any]) -> np.ndarray:
        if self.resources.equation_encoder is not None:
            try:
                return self._coerce_vector(self.resources.equation_encoder(node))
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "Equation encoder failed; falling back to hashing encoder"
                )

        latex = self._extract_equation_text(node)
        return self._hash_embedding(latex or "EMPTY_EQUATION")

    # ------------------------------------------------------------------
    # extraction helpers
    # ------------------------------------------------------------------
    def _detect_modality(self, node: Dict[str, Any]) -> str:
        modality = self._safe_get(node, "type") or self._safe_get(node, "content_type")
        if modality:
            return str(modality).lower()
        if self._safe_get(node, "table_data") or self._safe_get(node, "table_body"):
            return "table"
        if self._safe_get(node, "latex"):
            return "equation"
        if self._safe_get(node, "img_path") or self._safe_get(node, "image_path"):
            return "image"
        return "text"

    def _extract_text(self, node: Dict[str, Any]) -> str:
        for key in ("content", "text", "body", "chunk_text", "summary", "raw_text"):
            value = self._safe_get(node, key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _extract_image_source(self, node: Dict[str, Any]) -> str:
        for key in ("img_path", "image_path", "image_id", "image_hash"):
            value = self._safe_get(node, key)
            if isinstance(value, str) and value.strip():
                return value
        captions = self._safe_get(node, "captions")
        if isinstance(captions, list) and captions:
            return " ".join(str(c) for c in captions)
        return ""

    def _extract_table_text(self, node: Dict[str, Any]) -> str:
        parts = []
        for key in ("table_data", "table_body", "table_caption", "table_headers"):
            value = self._safe_get(node, key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(v) for v in value)
        text = "\n".join(parts)
        if text:
            return text
        return self._extract_text(node)

    def _extract_equation_text(self, node: Dict[str, Any]) -> str:
        latex = self._safe_get(node, "latex")
        if isinstance(latex, str) and latex.strip():
            return latex
        return self._extract_text(node)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def _hash_embedding(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
        base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        base = base / 255.0
        if base.size >= self.encoder_dim:
            return base[: self.encoder_dim]
        reps = int(np.ceil(self.encoder_dim / base.size))
        tiled = np.tile(base, reps)
        return tiled[: self.encoder_dim]

    def _coerce_vector(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            vector = value
        elif isinstance(value, (list, tuple)):
            vector = np.asarray(value, dtype=np.float32)
        elif hasattr(value, "detach") and hasattr(value, "cpu"):
            vector = value.detach().cpu().numpy()
        else:
            raise TypeError(
                f"Unsupported encoder output type: {type(value)!r}; expected array-like"
            )

        if vector.ndim == 2:
            if vector.shape[0] == 1:
                vector = vector[0]
            elif vector.shape[1] == 1:
                vector = vector[:, 0]
            else:
                raise ValueError(
                    "Expected a 1D vector from encoder, received 2D matrix"
                )
        if vector.ndim != 1:
            raise ValueError("Encoder output must be a 1D vector")
        if vector.size != self.encoder_dim:
            if vector.size == 0:
                return self._hash_embedding("EMPTY_VECTOR")
            vector = self._resize_vector(vector, self.encoder_dim)
        return vector.astype(np.float32, copy=False)

    def _resize_vector(self, vector: np.ndarray, target: int) -> np.ndarray:
        if vector.size == target:
            return vector
        reps = int(np.ceil(target / vector.size))
        tiled = np.tile(vector, reps)
        return tiled[:target]

    def _safe_get(self, node: Dict[str, Any], key: str) -> Any:
        if isinstance(node, dict):
            return node.get(key)
        return getattr(node, key, None)


__all__ = ["ChunkEncoder", "EncoderResources"]
