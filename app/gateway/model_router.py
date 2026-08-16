"""Smart model routing and provider resolution with health tracking, dynamic Ollama/Gemini discovery, and OpenRouter support."""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any, Literal
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ModelRouter")

try:
    from app.schemas.gateway_schemas import ModelRouteDecision
except ModuleNotFoundError:
    from schemas.gateway_schemas import ModelRouteDecision


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

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
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
    ) -> list[dict[str, Any]]:
        """Return ordered list of provider candidates across Ollama, Gemini, and OpenRouter."""
        self.reload_config()

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
            "api_key": self.gemini_api_key,
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


global_model_router = ModelRouter()
