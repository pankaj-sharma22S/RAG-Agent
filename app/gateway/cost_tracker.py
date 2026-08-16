"""Centralized token and cost tracking for the LLM Gateway."""

from __future__ import annotations

import os
import time
from datetime import datetime
from threading import Lock
from typing import Any
from dotenv import load_dotenv

load_dotenv()

try:
    from app.schemas.gateway_schemas import CostRecord, UsageMetadata
except ModuleNotFoundError:
    from schemas.gateway_schemas import CostRecord, UsageMetadata

# Pricing table per 1M tokens (USD)
FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "ollama": {"input": 0.0, "output": 0.0},
    "gemini": {"input": 0.075, "output": 0.30},  # gemini-2.5-flash estimate
}


class CostTracker:
    """Thread-safe collector for request-level and aggregate token/cost usage."""

    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._lock = Lock()

    def estimate_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost in USD based on provider/model rates."""
        prov = provider.lower()
        pricing = FALLBACK_PRICING.get(prov, {"input": 0.10, "output": 0.40})
        cost = (input_tokens / 1_000_000.0) * pricing["input"] + (output_tokens / 1_000_000.0) * pricing["output"]
        return round(cost, 6)

    def record_usage(
        self,
        request_id: str,
        session_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False,
    ) -> CostRecord:
        """Record usage metrics for an individual completion."""
        cost = 0.0 if cache_hit else self.estimate_cost(provider, input_tokens, output_tokens)
        record = CostRecord(
            request_id=request_id,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
            fallback_used=fallback_used,
        )

        with self._lock:
            # Keep max 5,000 recent records in memory
            if len(self._records) >= 5000:
                self._records = self._records[-3000:]
            self._records.append(record)

        return record

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Aggregate metrics for a specific conversation session."""
        with self._lock:
            session_records = [r for r in self._records if r.session_id == session_id]

        return self._aggregate_records(session_records)

    def get_global_stats(self) -> dict[str, Any]:
        """Aggregate global metrics across all requests."""
        with self._lock:
            all_records = list(self._records)

        return self._aggregate_records(all_records)

    def _aggregate_records(self, records: list[CostRecord]) -> dict[str, Any]:
        total_requests = len(records)
        if total_requests == 0:
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "cache_hits": 0,
                "fallback_count": 0,
                "by_provider": {},
            }

        total_in = sum(r.input_tokens for r in records)
        total_out = sum(r.output_tokens for r in records)
        total_cost = sum(r.estimated_cost_usd for r in records)
        total_latency = sum(r.latency_ms for r in records)
        cache_hits = sum(1 for r in records if r.cache_hit)
        fallback_count = sum(1 for r in records if r.fallback_used)

        by_provider: dict[str, dict[str, Any]] = {}
        for r in records:
            p = r.provider
            if p not in by_provider:
                by_provider[p] = {"requests": 0, "tokens": 0, "cost_usd": 0.0}
            by_provider[p]["requests"] += 1
            by_provider[p]["tokens"] += r.total_tokens
            by_provider[p]["cost_usd"] = round(by_provider[p]["cost_usd"] + r.estimated_cost_usd, 6)

        return {
            "total_requests": total_requests,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(total_latency / total_requests, 2),
            "cache_hits": cache_hits,
            "fallback_count": fallback_count,
            "by_provider": by_provider,
        }


global_cost_tracker = CostTracker()
