"""Central LLM Gateway providing unified LiteLLM execution, multi-tier fallback, caching, and cost tracking."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Iterator, Sequence, Type, TypeVar
from dotenv import load_dotenv
import litellm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

load_dotenv()

try:
    from app.gateway.cache import global_response_cache
    from app.gateway.cost_tracker import global_cost_tracker
    from app.gateway.fallback import FallbackManager
    from app.gateway.model_router import global_model_router
    from app.schemas.gateway_schemas import (
        FallbackDecision,
        LLMRequest,
        LLMResponse,
        UsageMetadata,
        GatewayRoutingDecision,
    )
    from app.guardrails.pii import global_pii_guard
except ModuleNotFoundError:
    from gateway.cache import global_response_cache
    from gateway.cost_tracker import global_cost_tracker
    from gateway.fallback import FallbackManager
    from gateway.model_router import global_model_router
    from schemas.gateway_schemas import (
        FallbackDecision,
        LLMRequest,
        LLMResponse,
        UsageMetadata,
        GatewayRoutingDecision,
    )
    from guardrails.pii import global_pii_guard

try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f

logger = logging.getLogger("LLMGateway")


def _sanitize_model_content(content: str) -> str:
    """Apply the shared secret/PII redaction policy at the gateway boundary."""
    return global_pii_guard.scan_and_mask(content or "").sanitized_text

# LiteLLM configuration
litellm.drop_params = True
litellm.set_verbose = False

T = TypeVar("T", bound=BaseModel)


class LLMGateway:
    """Central gateway orchestrating LLM calls across Ollama (llama3.2:latest) and Gemini."""

    def __init__(self) -> None:
        self.default_timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))

    def route_query(self, query: str, has_internal_context: bool = False) -> GatewayRoutingDecision:
        """Return the deterministic semantic route for an incoming user request."""
        return global_model_router.route_query(query, has_internal_context)

    def route_query_json(self, query: str, has_internal_context: bool = False) -> str:
        """Return the semantic route as valid JSON only."""
        return self.route_query(query, has_internal_context).model_dump_json()

    def _convert_messages(self, messages: list[Any]) -> list[dict[str, str]]:
        """Convert LangChain messages or dicts to standard role/content dicts."""
        formatted: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(msg)
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, BaseMessage):
                role = getattr(msg, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                formatted.append({"role": role, "content": str(msg.content)})
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted

    @safe_traceable(name="llm_gateway_invoke", run_type="llm", tags=["gateway", "llm", "multi-tier-fallback"])
    def invoke(self, request: LLMRequest) -> LLMResponse:
        """Execute completion through the gateway with caching, routing, and fallback."""
        start_time = time.time()
        request_id = f"req-{uuid.uuid4().hex[:10]}"
        raw_messages = self._convert_messages(request.messages)

        # 1. Smart Model Routing
        candidates = global_model_router.get_provider_chain(
            complexity=request.complexity,
            force_provider=request.force_provider,
            runtime_gemini_key=request.api_key,
        )

        timeout = request.timeout if request.timeout is not None else self.default_timeout
        primary_candidate = candidates[0]
        schema_name = (
            request.response_format.__name__
            if hasattr(request.response_format, "__name__")
            else str(request.response_format)
            if request.response_format
            else None
        )

        # 2. Check Cache
        cache_key = global_response_cache.generate_cache_key(
            messages=raw_messages,
            model=primary_candidate["model"],
            temperature=request.temperature,
            session_id=request.session_id,
            structured_schema=schema_name,
        )
        if request.use_cache:
            cached_data = global_response_cache.get(cache_key)
            if cached_data:
                duration_ms = (time.time() - start_time) * 1000
                global_cost_tracker.record_usage(
                    request_id=request_id,
                    session_id=request.session_id,
                    model=primary_candidate["model"],
                    provider=primary_candidate["provider"],
                    input_tokens=cached_data.get("input_tokens", 0),
                    output_tokens=cached_data.get("output_tokens", 0),
                    latency_ms=duration_ms,
                    cache_hit=True,
                )
                return LLMResponse(
                    content=_sanitize_model_content(cached_data["content"]),
                    structured_data=cached_data.get("structured_data"),
                    model=primary_candidate["model"],
                    provider=primary_candidate["provider"],
                    cache_hit=True,
                    fallback_used=False,
                    usage=UsageMetadata(
                        input_tokens=cached_data.get("input_tokens", 0),
                        output_tokens=cached_data.get("output_tokens", 0),
                        total_tokens=cached_data.get("total_tokens", 0),
                        latency_ms=duration_ms,
                        estimated_cost_usd=0.0,
                    ),
                )

        # 3. Multi-Tier Fallback Loop
        errors: list[tuple[str, str, Exception]] = []
        fallback_decision: FallbackDecision | None = None
        last_exception: Exception | None = None

        for idx, candidate in enumerate(candidates):
            provider = candidate["provider"]
            model = candidate["model"]
            api_key = candidate.get("api_key")
            api_base = candidate.get("api_base")

            # Check credentials for providers that require them
            if provider == "gemini" and not api_key:
                errors.append((provider, model, ValueError("GEMINI_API_KEY is not configured.")))
                continue
            if provider == "openrouter" and not api_key:
                errors.append((provider, model, ValueError("OPENROUTER_API_KEY is not configured.")))
                continue

            try:
                litellm_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": raw_messages,
                    "temperature": request.temperature,
                    "timeout": timeout,
                }
                if api_key:
                    litellm_kwargs["api_key"] = api_key
                if api_base:
                    litellm_kwargs["api_base"] = api_base
                if request.max_tokens:
                    litellm_kwargs["max_tokens"] = request.max_tokens

                if "gemini" in model.lower():
                    litellm_kwargs["thinking"] = {"type": "disabled"}

                # Handle structured output formatting via response_format if provided
                if request.response_format and issubclass(request.response_format, BaseModel):
                    litellm_kwargs["response_format"] = request.response_format

                response = litellm.completion(**litellm_kwargs)
                duration_ms = (time.time() - start_time) * 1000

                content = _sanitize_model_content(response.choices[0].message.content or "")
                structured_obj = None

                if request.response_format and issubclass(request.response_format, BaseModel):
                    try:
                        structured_obj = request.response_format.model_validate_json(content)
                    except Exception:
                        # Fallback parsing
                        structured_obj = None

                # Extract usage
                usage_dict = getattr(response, "usage", None)
                in_tokens = getattr(usage_dict, "prompt_tokens", 0) if usage_dict else len(json.dumps(raw_messages)) // 4
                out_tokens = getattr(usage_dict, "completion_tokens", 0) if usage_dict else len(content) // 4
                total_toks = in_tokens + out_tokens

                # Record cost & audit
                cost_rec = global_cost_tracker.record_usage(
                    request_id=request_id,
                    session_id=request.session_id,
                    model=model,
                    provider=provider,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    latency_ms=duration_ms,
                    cache_hit=False,
                    fallback_used=(idx > 0),
                )

                # Store in cache
                if request.use_cache and content and not content.strip().startswith("I encountered an issue"):
                    global_response_cache.set(
                        cache_key,
                        {
                            "content": content,
                            "structured_data": structured_obj.model_dump() if structured_obj else None,
                            "input_tokens": in_tokens,
                            "output_tokens": out_tokens,
                            "total_tokens": total_toks,
                        },
                    )

                if idx > 0 and errors:
                    prev_prov, prev_model, prev_err = errors[0]
                    fallback_decision = FallbackManager.build_fallback_record(
                        primary_model=prev_model,
                        primary_provider=prev_prov,
                        fallback_model=model,
                        fallback_provider=provider,
                        error=prev_err,
                    )

                return LLMResponse(
                    content=content,
                    structured_data=structured_obj,
                    model=model,
                    provider=provider,
                    cache_hit=False,
                    fallback_used=(idx > 0),
                    fallback_info=fallback_decision,
                    usage=UsageMetadata(
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=total_toks,
                        latency_ms=duration_ms,
                        estimated_cost_usd=cost_rec.estimated_cost_usd,
                    ),
                )

            except Exception as error:
                last_exception = error
                err_str = str(error).lower()
                is_quota_exhausted = any(k in err_str for k in ["resource_exhausted", "daily", "quota", "per_minute", "429", "rate_limit"])
                if is_quota_exhausted:
                    logger.warning(f"Rate limit / Quota exhausted on [{provider}/{model}]. Marking unhealthy for 1h and failing over instantly...")
                    global_model_router.mark_provider_unhealthy(provider, cooldown_seconds=3600.0)
                    errors.append((provider, model, error))
                    continue

                errors.append((provider, model, error))
                logger.warning(f"Provider [{provider}/{model}] failed: {error}")
                global_model_router.mark_provider_unhealthy(provider, cooldown_seconds=60.0)
                continue

        # 4. All providers failed
        error_summary = "; ".join(f"[{p}/{m}]: {e}" for p, m, e in errors)
        raise RuntimeError(
            f"All configured LLM providers failed. Attempted candidates: {error_summary}"
        ) from last_exception

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Stream token-by-token completions with fallback resilience."""
        raw_messages = self._convert_messages(request.messages)
        candidates = global_model_router.get_provider_chain(
            complexity=request.complexity,
            force_provider=request.force_provider,
            runtime_gemini_key=request.api_key,
        )
        timeout = request.timeout if request.timeout is not None else self.default_timeout

        for candidate in candidates:
            provider = candidate["provider"]
            model = candidate["model"]
            api_key = candidate.get("api_key")
            api_base = candidate.get("api_base")

            if provider == "gemini" and not api_key:
                continue
            if provider == "openrouter" and not api_key:
                continue

            try:
                litellm_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": raw_messages,
                    "temperature": request.temperature,
                    "timeout": timeout,
                    "stream": True,
                }
                if api_key:
                    litellm_kwargs["api_key"] = api_key
                if api_base:
                    litellm_kwargs["api_base"] = api_base
                if request.max_tokens:
                    litellm_kwargs["max_tokens"] = request.max_tokens

                if "gemini" in model.lower():
                    litellm_kwargs["thinking"] = {"type": "disabled"}

                response = litellm.completion(**litellm_kwargs)
                yielded_any = False
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yielded_any = True
                        yield _sanitize_model_content(delta)
                if yielded_any:
                    return
            except Exception as error:
                err_str = str(error).lower()
                is_quota_exhausted = any(k in err_str for k in ["resource_exhausted", "daily", "quota", "per_minute", "429", "rate_limit"])
                if is_quota_exhausted:
                    logger.warning(f"Streaming quota exhausted on [{provider}/{model}]. Marking unhealthy for 1h and failing over instantly...")
                    global_model_router.mark_provider_unhealthy(provider, cooldown_seconds=3600.0)
                    continue

                logger.warning(f"Streaming provider [{provider}/{model}] failed: {error}")
                global_model_router.mark_provider_unhealthy(provider, cooldown_seconds=60.0)
                continue

        yield "I encountered an issue connecting to the AI language models. Please verify provider connectivity or API keys."

    def structured_invoke(self, prompt: str | list[Any], schema: Type[T], session_id: str = "default") -> T:
        """Helper to invoke structured output using Pydantic validation."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        req = LLMRequest(
            messages=self._convert_messages(messages),
            complexity="medium",
            response_format=schema,
            session_id=session_id,
        )
        resp = self.invoke(req)
        if resp.structured_data and isinstance(resp.structured_data, schema):
            return resp.structured_data
        
        # Fallback to direct validation from raw content
        try:
            return schema.model_validate_json(resp.content)
        except Exception:
            # Try cleaning markdown json formatting if present
            cleaned = resp.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return schema.model_validate_json(cleaned.strip())


# Global Gateway Instance
llm_gateway = LLMGateway()


class GatewayChatModel(BaseChatModel):
    """LangChain BaseChatModel adapter wrapping the Central LLMGateway."""

    gateway: Any = Field(default=None)
    temperature: float = 0.2
    complexity: str = "medium"
    session_id: str = "default"
    api_key: str | None = Field(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.gateway is None:
            self.gateway = llm_gateway

    @property
    def _llm_type(self) -> str:
        return "litellm_gateway"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        req = LLMRequest(
            messages=self.gateway._convert_messages(messages),
            complexity=self.complexity,  # type: ignore
            temperature=self.temperature,
            session_id=self.session_id,
            api_key=self.api_key or kwargs.get("api_key"),
        )
        resp = self.gateway.invoke(req)
        message = AIMessage(content=resp.content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def stream(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        req = LLMRequest(
            messages=self.gateway._convert_messages(input if isinstance(input, list) else [input]),
            complexity=self.complexity,  # type: ignore
            temperature=self.temperature,
            session_id=self.session_id,
            api_key=self.api_key or kwargs.get("api_key"),
        )
        for chunk in self.gateway.stream(req):
            yield AIMessage(content=chunk)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a callable runnable that maps input to structured Pydantic schema."""
        from langchain_core.runnables import RunnableLambda

        def _call_structured(input_val: Any) -> Any:
            return self.gateway.structured_invoke(input_val, schema, session_id=self.session_id)

        return RunnableLambda(_call_structured)


GatewayChatModel.model_rebuild()

# LangChain-compatible singleton adapter for existing chains and nodes
gateway_chat_model = GatewayChatModel()
