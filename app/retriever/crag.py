"""Corrective RAG (CRAG) evidence evaluation engine."""

from __future__ import annotations

import logging
from typing import Any
from langchain_core.documents import Document

try:
    from app.gateway import llm_gateway
    from app.schemas.retrieval_schemas import CRAGDecision
except ModuleNotFoundError:
    from gateway import llm_gateway
    from schemas.retrieval_schemas import CRAGDecision

logger = logging.getLogger("CRAGEvaluator")


class CRAGEvaluator:
    """Evaluates retrieved document evidence quality, completeness, and contradictions."""

    @staticmethod
    def evaluate(
        query: str,
        documents: list[Document],
        conversation_context: str = "",
        session_id: str = "default",
    ) -> CRAGDecision:
        """Evaluate retrieved documents against the query and return a structured CRAGDecision."""
        if not documents:
            return CRAGDecision(
                decision="bad",
                relevance_score=0.0,
                completeness_score=0.0,
                contradiction=False,
                missing_information=["No document chunks were found for this query."],
                reason="No candidate documents were retrieved.",
                should_refine=True,
            )

        evidence_text = "\n\n".join(
            f"[Doc {idx+1} | Source: {doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
            for idx, doc in enumerate(documents[:8])
        )[:10_000]

        prompt = (
            "You are a strict, objective Corrective RAG (CRAG) evidence evaluation engine.\n"
            "Analyze whether the retrieved document evidence contains sufficient, accurate, and relevant "
            "facts to directly answer the user query.\n\n"
            f"User Query: {query}\n"
            f"Conversation Context:\n{conversation_context}\n\n"
            f"Retrieved Document Evidence:\n{evidence_text}\n\n"
            "Evaluation Criteria:\n"
            "- 'good': Evidence contains the direct, specific facts required to answer the query accurately.\n"
            "- 'partial': Evidence is on-topic and partially answers the query, but leaves key questions or details unanswered.\n"
            "- 'bad': Evidence is irrelevant, tangential, contradictory, or lacks the necessary facts.\n\n"
            "Be critical and precise. If the evidence is bad or partial, specify the exact missing_information."
        )

        try:
            decision = llm_gateway.structured_invoke(
                prompt=prompt,
                schema=CRAGDecision,
                session_id=session_id,
            )
            # Ensure should_refine flag matches decision
            if decision.decision in {"partial", "bad"}:
                decision.should_refine = True
            else:
                decision.should_refine = False
            return decision
        except Exception as err:
            logger.warning(f"CRAG evaluation structured call failed: {err}. Applying heuristic fallback.")
            # Heuristic fallback based on document length and term overlap
            query_terms = set(query.lower().split())
            evidence_terms = set(evidence_text.lower().split())
            overlap = len(query_terms.intersection(evidence_terms)) / max(len(query_terms), 1)

            if overlap > 0.6:
                return CRAGDecision(
                    decision="good",
                    relevance_score=0.8,
                    completeness_score=0.8,
                    contradiction=False,
                    missing_information=[],
                    reason="Heuristic evaluation: high term overlap and document presence.",
                    should_refine=False,
                )
            elif overlap > 0.3:
                return CRAGDecision(
                    decision="partial",
                    relevance_score=0.5,
                    completeness_score=0.5,
                    contradiction=False,
                    missing_information=["Additional specific context required."],
                    reason="Heuristic evaluation: moderate term overlap.",
                    should_refine=True,
                )
            else:
                return CRAGDecision(
                    decision="bad",
                    relevance_score=0.2,
                    completeness_score=0.1,
                    contradiction=False,
                    missing_information=["Specific document facts matching query terms."],
                    reason="Heuristic evaluation: low term overlap.",
                    should_refine=True,
                )


global_crag_evaluator = CRAGEvaluator()
