"""Selective expansion heuristics for REFRAG compressed chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from raganything.refrag.context.context_planner_refrag import ContextPlan

KEYWORDS_QUOTE = {"quote", "verbatim", "exact", "citation", "footnote", "legal"}
NUMERIC_PATTERN = re.compile(r"\d")


@dataclass
class SelectiveExpansionPolicy:
    """Configuration for selective expansion."""

    expand_numeric_tables: bool = True
    expand_legal_citations: bool = True
    entropy_tau: float = 2.0
    mode: str = "heuristic"


class SelectiveExpansionHeuristic:
    """Rule-based selective expansion policy."""

    def __init__(self, policy: SelectiveExpansionPolicy) -> None:
        self.policy = policy

    def initial_selection(self, plan: ContextPlan, query: str) -> List[int]:
        selections: List[int] = []
        lower_query = query.lower()
        requires_quote = any(keyword in lower_query for keyword in KEYWORDS_QUOTE)

        for idx, (node, text) in enumerate(zip(plan.chunk_nodes, plan.chunk_texts)):
            modality = _get_modality(node)
            if self.policy.expand_numeric_tables and modality in {"table", "equation"}:
                selections.append(idx)
                continue

            if self.policy.expand_legal_citations and _looks_like_legal(text):
                selections.append(idx)
                continue

            if requires_quote and _contains_query_phrase(lower_query, text.lower()):
                selections.append(idx)
                continue

            if _is_numeric_dense(text):
                selections.append(idx)

        return sorted(set(selections))

    def update_from_entropy(self, plan: ContextPlan, entropy_values: Iterable[float]) -> List[int]:
        if self.policy.mode != "heuristic":
            return []
        threshold = self.policy.entropy_tau
        expansions: List[int] = []
        for idx, entropy in enumerate(entropy_values):
            if entropy >= threshold:
                expansions.append(idx)
        return expansions


def _contains_query_phrase(query: str, text: str) -> bool:
    words = [w for w in re.split(r"\W+", query) if len(w) > 3]
    if not words:
        return False
    return any(word in text for word in words)


def _looks_like_legal(text: str) -> bool:
    text = text.lower()
    return any(marker in text for marker in ("§", "article", "act", "regulation", "u.s.c"))


def _is_numeric_dense(text: str) -> bool:
    if not text:
        return False
    digits = len(NUMERIC_PATTERN.findall(text))
    ratio = digits / max(len(text), 1)
    return ratio > 0.1


def _get_modality(node: object) -> str:
    if isinstance(node, dict):
        modality = node.get("type") or node.get("content_type")
    else:
        modality = getattr(node, "type", None) or getattr(node, "content_type", None)
    return str(modality).lower() if modality else "text"


__all__ = ["SelectiveExpansionPolicy", "SelectiveExpansionHeuristic"]
