"""Adaptive query refinement and multi-query decomposition engine."""

from __future__ import annotations

import logging
from typing import Any
from langchain_core.documents import Document

try:
    from app.gateway import llm_gateway
    from app.schemas.retrieval_schemas import CRAGDecision, QueryRefinement
except ModuleNotFoundError:
    from gateway import llm_gateway
    from schemas.retrieval_schemas import CRAGDecision, QueryRefinement

logger = logging.getLogger("QueryRefiner")


class QueryRefiner:
    """Refines failed or partial search queries based on CRAG failure diagnosis."""

    @staticmethod
    def refine(
        original_query: str,
        current_query: str,
        crag_decision: CRAGDecision,
        conversation_context: str = "",
        attempt: int = 1,
        previous_strategies: list[str] | None = None,
        session_id: str = "default",
    ) -> QueryRefinement:
        """Generate an improved search query and strategy targeting the missing information."""
        prev_strats = ", ".join(previous_strategies or [])

        prompt = (
            "You are an expert retrieval optimization engine for an enterprise RAG system.\n"
            "The previous retrieval attempt failed or was incomplete. Your goal is to produce a refined, "
            "highly targeted search query that will successfully locate the missing document evidence.\n\n"
            f"Original User Query: {original_query}\n"
            f"Previous Search Query: {current_query}\n"
            f"Attempt Number: {attempt}\n"
            f"Previous Strategies Attempted: {prev_strats or 'None'}\n"
            f"CRAG Evaluation Reason: {crag_decision.reason}\n"
            f"Missing Information: {', '.join(crag_decision.missing_information) if crag_decision.missing_information else 'Specific factual grounding'}\n"
            f"Conversation Context:\n{conversation_context}\n\n"
            "Instructions:\n"
            "1. Analyze why previous retrieval failed.\n"
            "2. Select the most appropriate refinement strategy:\n"
            "   - 'query_rewrite': Rephrase with clearer structure and syntax.\n"
            "   - 'terminology_expansion': Include domain synonyms, technical terms, and acronyms.\n"
            "   - 'entity_expansion': Explicitly name key entities, modules, components, or files.\n"
            "   - 'metadata_focused': Target metadata attributes, sections, or chapter keywords.\n"
            "   - 'multi_query' or 'decomposition': Break a multi-part query into 2-3 focused sub-queries.\n"
            "   - 'narrower_query': Focus on the specific missing detail rather than the broad topic.\n"
            "   - 'broader_query': Relax overly specific constraints if previous query found nothing.\n"
            "   - 'temporal_clarification': Add version, sequence, or time markers.\n"
            "3. Ensure the 'refined_query' is meaningfully different from the previous query and specifically targets the missing facts."
        )

        try:
            refinement = llm_gateway.structured_invoke(
                prompt=prompt,
                schema=QueryRefinement,
                session_id=session_id,
            )
            # Guarantee refined query is not empty and not identical to current query
            if not refinement.refined_query.strip() or refinement.refined_query.strip() == current_query.strip():
                refinement.refined_query = f"{original_query} {' '.join(crag_decision.missing_information[:2])}".strip()
            return refinement
        except Exception as err:
            logger.warning(f"Structured query refinement failed: {err}. Applying fallback heuristic refinement.")
            missing = " ".join(crag_decision.missing_information) if crag_decision.missing_information else "details"
            fallback_query = f"{original_query} {missing}".strip()
            return QueryRefinement(
                original_query=original_query,
                refined_query=fallback_query,
                strategy="terminology_expansion",
                reason="Appended missing information keywords to original query.",
                missing_information=crag_decision.missing_information,
                search_focus=[original_query, missing],
                sub_queries=[],
            )


global_query_refiner = QueryRefiner()
