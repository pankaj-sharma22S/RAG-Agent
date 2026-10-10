import os
import logging
from typing import Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger("GeminiClient")


def get_gemini_api_key(runtime_key: str | None = None) -> str:
    """Resolve Gemini API key from explicit runtime parameter or environment variables."""
    key = (
        (runtime_key and runtime_key.strip())
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_KEY", "").strip()
    )
    return key


def get_gemini_llm(
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Dynamically build and return a ChatGoogleGenerativeAI instance with proper error checking."""
    resolved_key = get_gemini_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file, "
            "or provide it dynamically via request headers ('x-gemini-api-key')."
        )
    target_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash").replace("gemini/", "")
    return ChatGoogleGenerativeAI(
        model=target_model,
        temperature=temperature,
        api_key=resolved_key,
        **kwargs,
    )


class _LazyGeminiProxy:
    """Proxy object that defers ChatGoogleGenerativeAI creation until first use or provides clear error messages."""

    def __init__(self) -> None:
        self._cached_instance: ChatGoogleGenerativeAI | None = None

    def _get_instance(self) -> ChatGoogleGenerativeAI:
        if self._cached_instance is None:
            self._cached_instance = get_gemini_llm()
        return self._cached_instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_instance().invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_instance().stream(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_instance().with_structured_output(*args, **kwargs)


# Exported singleton / proxy for existing code
llm = _LazyGeminiProxy()