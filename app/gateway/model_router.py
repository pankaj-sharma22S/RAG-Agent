"""Smart model routing and provider resolution with health tracking, dynamic Ollama/Gemini discovery, and OpenRouter support."""

from __future__ import annotations

import logging
import os
import re
import time
from threading import Lock
from typing import Any, Literal
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ModelRouter")

try:
    from app.schemas.gateway_schemas import GatewayRoutingDecision, ModelRouteDecision
except ModuleNotFoundError:
    from schemas.gateway_schemas import GatewayRoutingDecision, ModelRouteDecision


class ModelRouter:
    """Selects and sequences models across Ollama (llama3.2:latest), Gemini, and OpenRouter with circuit breaker."""

    def __init__(self) -> None:
        self._unhealthy_cooldown: dict[str, float] = {}
        self._discovered_ollama_model: str | None = None
        self._lock = Lock()
        self.reload_config()

    def reload_config(self) -> None:
        """Refresh configuration from environment and probe local Ollama."""
        load_dotenv()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "ollama/llama3.2:latest")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct")

        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_KEY")
        )
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Discover installed Ollama model if available
        self._discover_ollama_model()

    def _discover_ollama_model(self) -> None:
        """Detect locally installed Ollama models to prevent 'model not found' errors."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
                if models:
                    clean_target = self.ollama_model.replace("ollama/", "")
                    if clean_target in models:
                        self._discovered_ollama_model = f"ollama/{clean_target}"
                    elif "llama3.2:latest" in models:
                        self._discovered_ollama_model = "ollama/llama3.2:latest"
                    elif "llama3.2" in models:
                        self._discovered_ollama_model = "ollama/llama3.2"
                    else:
                        self._discovered_ollama_model = f"ollama/{models[0]}"
        except Exception:
            self._discovered_ollama_model = None

    def mark_provider_unhealthy(self, provider: str, cooldown_seconds: float = 60.0) -> None:
        """Temporarily mark a provider as unhealthy so subsequent calls avoid timeout latency."""
        with self._lock:
            self._unhealthy_cooldown[provider] = time.time() + cooldown_seconds

    def is_provider_healthy(self, provider: str) -> bool:
        """Check if provider is currently healthy or out of cooldown."""
        with self._lock:
            cooldown_until = self._unhealthy_cooldown.get(provider, 0.0)
            return time.time() >= cooldown_until

    def get_provider_chain(
        self,
        complexity: Literal["simple", "medium", "complex"] = "medium",
        force_provider: Literal["ollama", "gemini", "openrouter"] | None = None,
        runtime_gemini_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ordered list of provider candidates across Ollama, Gemini, and OpenRouter."""
        self.reload_config()

        effective_gemini_key = runtime_gemini_key or self.gemini_api_key
        effective_ollama_model = self._discovered_ollama_model or self.ollama_model

        ollama_entry = {
            "provider": "ollama",
            "model": effective_ollama_model,
            "api_base": self.ollama_base_url,
            "api_key": None,
        }
        gemini_entry = {
            "provider": "gemini",
            "model": self.gemini_model,
            "api_key": effective_gemini_key,
        }
        openrouter_entry = {
            "provider": "openrouter",
            "model": self.openrouter_model,
            "api_key": self.openrouter_api_key,
        }

        if force_provider:
            if force_provider == "ollama":
                return [ollama_entry, gemini_entry, openrouter_entry]
            elif force_provider == "gemini":
                return [gemini_entry, openrouter_entry, ollama_entry]
            elif force_provider == "openrouter":
                return [openrouter_entry, gemini_entry, ollama_entry]

        ollama_healthy = self.is_provider_healthy("ollama")
        gemini_healthy = self.is_provider_healthy("gemini")

        primary = os.getenv("PRIMARY_PROVIDER", os.getenv("DEFAULT_LLM_PROVIDER", "gemini")).lower()
        if primary == "gemini" and self.gemini_api_key and gemini_healthy:
            return [gemini_entry, ollama_entry, openrouter_entry]

        # 1. Primary: Local Ollama (if configured as primary and healthy)
        if self._discovered_ollama_model and ollama_healthy:
            return [ollama_entry, gemini_entry, openrouter_entry]

        # 2. Fallback: Gemini (if key present and healthy)
        if self.gemini_api_key and gemini_healthy:
            return [gemini_entry, openrouter_entry, ollama_entry]

        return [ollama_entry, gemini_entry, openrouter_entry]

    def route_request(
        self,
        complexity: Literal["simple", "medium", "complex"] = "medium",
        force_provider: Literal["ollama", "gemini", "openrouter"] | None = None,
    ) -> ModelRouteDecision:
        """Return the primary routing decision."""
        chain = self.get_provider_chain(complexity, force_provider)
        primary = chain[0]
        return ModelRouteDecision(
            complexity=complexity,
            selected_provider=primary["provider"],
            selected_model=primary["model"],
            reason=f"Selected {primary['provider']} for {complexity} task complexity.",
        )

    def route_query(self, query: str, has_internal_context: bool = False) -> GatewayRoutingDecision:
        """Classify a request into a deterministic retrieval destination and model tier."""
        text = (query or "").strip()
        normalized = text.lower()

        mutative_pattern = re.compile(
            r"\b(delete|drop|remove|destroy|deploy|cancel|revoke|disable|modify|overwrite|truncate|reset|"
            r"change permissions|grant access|send|publish)\b"
        )
        web_pattern = re.compile(
            r"\b(latest|current|today|now|this week|news|weather|release notes|pricing|transaction fees|"
            r"stock price|real[- ]time|recent)\b"
        )
        internal_pattern = re.compile(
            r"\b(our|internal|company|uploaded|document|codebase|repository|repo|policy|schema|architecture|"
            r"vacation|database|splitter\.py|internal file)\b|\.(py|md|pdf|docx|xlsx)\b"
        )
        comparison_pattern = re.compile(r"\b(compare|comparison|versus|vs\.?|against|difference between|internal .* and)\b")
        multimodal_pattern = re.compile(r"\b(image|images|photo|diagram|chart|table|visual|screenshot|figure|pdf scan)\b")
        deep_pattern = re.compile(r"\b(security audit|vulnerability|penetration|math proof|formal proof|multi-hop|algorithmic planning)\b")

        is_mutative = bool(mutative_pattern.search(normalized))
        requires_vit = False  # Set after mutative-action classification below.
        has_internal = bool(internal_pattern.search(normalized)) or has_internal_context
        needs_web = bool(web_pattern.search(normalized))
        is_comparison = bool(comparison_pattern.search(normalized))

        if is_mutative:
            route = "clarification_required"
            intent = "mutative_action"
            clarification = "Please confirm the exact target, scope, and intended change before any mutative action is considered."
        elif is_comparison and has_internal and needs_web:
            route = "hybrid_crag"
            intent = "internal_external_comparison"
            clarification = None
        elif needs_web:
            route = "web_search"
            intent = "current_external_information"
            clarification = None
        elif has_internal:
            route = "internal_rag"
            intent = "proprietary_or_document_knowledge"
            clarification = None
        else:
            route = "direct_chat"
            intent = "general_conversation_or_knowledge"
            clarification = None

        requires_vit = not is_mutative and bool(multimodal_pattern.search(normalized))

        if deep_pattern.search(normalized):
            tier = "deep_reasoning"
        elif route in {"internal_rag", "web_search", "hybrid_crag"}:
            tier = "standard_reasoning"
        else:
            tier = "cheap_fast"

        queries = [text] if route in {"internal_rag", "web_search", "hybrid_crag"} and text else []
        return GatewayRoutingDecision(
            intent=intent,
            target_route=route,
            recommended_model_tier=tier,
            is_mutative_action=is_mutative,
            requires_multimodal_vit=requires_vit,
            rewritten_search_queries=queries,
            clarification_message=clarification,
        )

    def route_query_json(self, query: str, has_internal_context: bool = False) -> str:
        """Return only the parseable JSON routing contract."""
        return self.route_query(query, has_internal_context).model_dump_json()


global_model_router = ModelRouter()
