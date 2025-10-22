"""Runtime orchestration for REFRAG decoding inside RAGAnything."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from raganything.refrag.cache import CacheConfig, RefragCache
from raganything.refrag.context.context_planner_refrag import ContextPlan, ContextPlanner
from raganything.refrag.decoder_adapter import (
    DecoderAdapter,
    DecoderInputs,
    assemble_decoder_inputs,
)
from raganything.refrag.encoders import ChunkEncoder
from raganything.refrag.projection import PhiConfig, PhiProjector
from raganything.refrag.selective_expand import (
    SelectiveExpansionHeuristic,
    SelectiveExpansionPolicy,
)

LOGGER = logging.getLogger(__name__)


class RefragRuntime:
    """High-level facade coordinating all REFRAG components."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        encoder_dim: int,
        decoder_dim: int,
        projector_hidden: int,
        expansion_policy: SelectiveExpansionPolicy,
        cache_config: CacheConfig,
        embedding_func: Optional[Callable[[str], Any]] = None,
        decoder_adapter: Optional[DecoderAdapter] = None,
    ) -> None:
        self.cache = RefragCache(cache_config)
        self.encoder = ChunkEncoder(encoder_dim, embedding_func=embedding_func)
        phi_config = PhiConfig(
            input_dim=encoder_dim, output_dim=decoder_dim, hidden_dim=projector_hidden
        )
        self.projector = PhiProjector(phi_config)
        self.context_planner = ContextPlanner(
            tokenizer=tokenizer, encoder=self.encoder, projector=self.projector, cache=self.cache
        )
        self.selective_expander = SelectiveExpansionHeuristic(expansion_policy)
        self.decoder_adapter = decoder_adapter or DecoderAdapter()

    # ------------------------------------------------------------------
    async def build_plan(self, query: str, retrieval_data: dict[str, Any]) -> ContextPlan:
        return await self.context_planner.build_plan(query, retrieval_data)

    async def generate(
        self,
        plan: ContextPlan,
        *,
        query: str,
        llm_callable: Optional[Callable[[str], Any]] = None,
        fallback_callable: Optional[Callable[[], Awaitable[str]]] = None,
    ) -> str:
        to_expand = self.selective_expander.initial_selection(plan, query)
        for idx in to_expand:
            plan.mark_expanded(idx)

        decoder_inputs = self._assemble_inputs(plan)

        if llm_callable is None:
            if fallback_callable is None:
                raise ValueError("Either llm_callable or fallback_callable must be provided")
            LOGGER.debug("No llm_callable supplied; falling back to LightRAG pipeline")
            return await fallback_callable()

        return await self.decoder_adapter.generate(
            decoder_inputs,
            plan=plan,
            llm_callable=llm_callable,
        )

    # ------------------------------------------------------------------
    def _assemble_inputs(self, plan: ContextPlan) -> DecoderInputs:
        token_spans = plan.expanded_token_spans(self.context_planner.tokenizer)
        return assemble_decoder_inputs(
            plan.question_tokens,
            plan.projected,
            token_spans,
            metadata={
                "expanded_indices": [idx for idx, flag in enumerate(plan.expand_mask) if flag],
                "total_chunks": len(plan.chunk_nodes),
            },
        )


__all__ = ["RefragRuntime"]
