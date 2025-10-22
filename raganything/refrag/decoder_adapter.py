"""Adapter utilities for assembling decoder inputs with REFRAG vectors."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class DecoderInputs:
    """Container for question tokens, soft chunk vectors, and expansions."""

    question_tokens: List[int]
    projected_chunks: np.ndarray
    expanded_token_spans: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def assemble_decoder_inputs(
    question_tokens: Iterable[int],
    projected_chunks: np.ndarray,
    expanded_token_spans: Optional[Iterable[Iterable[int]]] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecoderInputs:
    """Create a :class:`DecoderInputs` object from individual parts."""

    expanded = [list(span) for span in expanded_token_spans or []]
    question = list(question_tokens)
    return DecoderInputs(
        question_tokens=question,
        projected_chunks=np.asarray(projected_chunks, dtype=np.float32),
        expanded_token_spans=expanded,
        metadata=metadata or {},
    )


class DecoderAdapter:
    """Bridge between REFRAG soft vectors and the downstream decoder."""

    def __init__(self, *, supports_soft_tokens: bool = False) -> None:
        self._supports_soft_tokens = supports_soft_tokens

    # ------------------------------------------------------------------
    @property
    def supports_soft_tokens(self) -> bool:
        return self._supports_soft_tokens

    # ------------------------------------------------------------------
    def build_fallback_prompt(self, plan: "ContextPlan") -> str:
        """Construct a textual prompt equivalent for downstream models."""

        lines = ["You are using REFRAG compressed context chunks."]
        lines.append(f"Question: {plan.question}")
        lines.append("Context plan summary:")
        for idx, node in enumerate(plan.chunk_nodes):
            chunk_info = [f"Chunk {idx + 1}"]
            modality = getattr(node, "type", None) or node.get("type") if isinstance(node, dict) else None
            if modality:
                chunk_info.append(f"type={modality}")
            score = plan.chunk_scores[idx] if idx < len(plan.chunk_scores) else None
            if score is not None:
                chunk_info.append(f"score={score:.4f}")
            status = "expanded" if plan.expand_mask[idx] else "compressed"
            chunk_info.append(status)
            lines.append(" - " + ", ".join(chunk_info))
            if plan.expand_mask[idx]:
                excerpt = plan.chunk_texts[idx].strip()
                if excerpt:
                    lines.append(f"   content: {excerpt}")
        return "\n".join(lines)

    async def generate_with_fallback(
        self,
        inputs: DecoderInputs,
        *,
        llm_callable: Optional[Callable[[str], Any]] = None,
        fallback_prompt: str,
    ) -> str:
        """Generate text by falling back to a traditional prompt."""

        if llm_callable is None:
            raise ValueError("llm_callable must be provided for fallback generation")

        result = llm_callable(fallback_prompt)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise TypeError(
                "llm_callable must return a string when used in fallback mode"
            )
        return result

    async def generate(
        self,
        inputs: DecoderInputs,
        *,
        plan: "ContextPlan",
        llm_callable: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """Generate using soft tokens when supported, otherwise fallback."""

        if not self.supports_soft_tokens:
            fallback_prompt = self.build_fallback_prompt(plan)
            return await self.generate_with_fallback(
                inputs, llm_callable=llm_callable, fallback_prompt=fallback_prompt
            )

        # Placeholder for future native soft-token integration.
        LOGGER.warning(
            "Soft-token generation is not implemented; falling back to textual prompt"
        )
        fallback_prompt = self.build_fallback_prompt(plan)
        return await self.generate_with_fallback(
            inputs, llm_callable=llm_callable, fallback_prompt=fallback_prompt
        )


__all__ = ["DecoderAdapter", "DecoderInputs", "assemble_decoder_inputs"]
