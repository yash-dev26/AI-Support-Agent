"""
Centralized, decoupled chat model factory supporting multiple LLM providers
(OpenAI, Grok/xAI) with unified configuration and tool binding.

Decouples the chat completions layer across the project from hardcoded API
providers so the chatbot node, RAG policy engine, and guardrails can run
seamlessly against Grok, OpenAI, or any OpenAI-compatible provider.
"""
from dataclasses import dataclass
import os
import threading
from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_GROK_MODEL = "grok-2"
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float


def get_llm_settings(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> LLMSettings:
    """Resolve LLM settings from parameters and environment variables.

    Provider resolution order:
      1. Explicit `provider` argument ("openai", "grok", "xai")
      2. LLM_PROVIDER env var
      3. If GROK_API_KEY or XAI_API_KEY is present and OPENAI_API_KEY is not, "grok"
      4. Default: "openai"
    """
    resolved_provider = provider or os.getenv("LLM_PROVIDER")
    if resolved_provider:
        resolved_provider = resolved_provider.strip().lower()
    else:
        has_grok = bool(os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        if has_grok and not has_openai:
            resolved_provider = "grok"
        else:
            resolved_provider = "openai"

    # Normalize "xai" -> "grok" internally for consistent setting representation
    if resolved_provider == "xai":
        resolved_provider = "grok"

    # Resolve temperature
    if temperature is None:
        temp_env = os.getenv("LLM_TEMPERATURE")
        resolved_temp = float(temp_env) if temp_env is not None else 0.0
    else:
        resolved_temp = float(temperature)

    # Resolve provider-specific details
    if resolved_provider == "grok":
        resolved_model = (
            model
            or os.getenv("LLM_MODEL")
            or os.getenv("GROK_MODEL")
            or os.getenv("XAI_MODEL")
            or DEFAULT_GROK_MODEL
        )
        resolved_api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("XAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("GROK_BASE_URL")
            or os.getenv("XAI_BASE_URL")
            or DEFAULT_GROK_BASE_URL
        )
    else:
        # Default: OpenAI
        resolved_provider = "openai"
        resolved_model = (
            model
            or os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_OPENAI_MODEL
        )
        resolved_api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )

    return LLMSettings(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=resolved_temp,
    )


def create_chat_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a new chat model instance according to resolved settings.

    Supports OpenAI and Grok (xAI). For Grok, initializes via langchain-xai
    or ChatOpenAI pointed at xAI's OpenAI-compatible completions endpoint.
    """
    settings = get_llm_settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    call_kwargs = dict(kwargs)
    if settings.temperature is not None:
        call_kwargs["temperature"] = settings.temperature

    if settings.provider == "grok":
        # Attempt langchain-xai first; fallback to OpenAI-compatible ChatOpenAI
        try:
            from langchain_xai import ChatXAI

            xai_kwargs = dict(call_kwargs)
            if settings.api_key:
                xai_kwargs["api_key"] = settings.api_key
            if settings.base_url:
                xai_kwargs["base_url"] = settings.base_url
            return ChatXAI(model=settings.model, **xai_kwargs)
        except ImportError:
            # Fallback to ChatOpenAI with xAI base_url
            from langchain_openai import ChatOpenAI

            openai_kwargs = dict(call_kwargs)
            if settings.api_key:
                openai_kwargs["api_key"] = settings.api_key
            openai_kwargs["base_url"] = settings.base_url or DEFAULT_GROK_BASE_URL
            return ChatOpenAI(model=settings.model, **openai_kwargs)

    # OpenAI provider
    init_kwargs = dict(call_kwargs)
    if settings.api_key:
        init_kwargs["api_key"] = settings.api_key
    if settings.base_url:
        init_kwargs["base_url"] = settings.base_url

    return init_chat_model(
        model=settings.model,
        model_provider="openai",
        **init_kwargs,
    )


# Singleton cache for default chat model and tool-bound model
_default_chat_model: BaseChatModel | None = None
_default_llm_with_tools: Any | None = None
_lock = threading.Lock()


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return the cached default chat model or create a new configured one.

    If any overrides are passed, creates and returns a new instance.
    If no overrides are passed, returns the lazily-created default instance.
    """
    global _default_chat_model
    has_overrides = any(
        v is not None
        for v in (provider, model, api_key, base_url, temperature)
    ) or bool(kwargs)

    if has_overrides:
        return create_chat_model(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )

    with _lock:
        if _default_chat_model is None:
            _default_chat_model = create_chat_model()
        return _default_chat_model


def get_llm_with_tools(tools: Sequence[Any] | None = None) -> Any:
    """Return the chat model bound with tools.

    If tools is None, binds app.graph.tools.TOOLS to the default chat model.
    """
    global _default_llm_with_tools
    if tools is not None:
        llm = get_chat_model()
        return llm.bind_tools(tools=tools)

    with _lock:
        if _default_llm_with_tools is None:
            from app.graph.tools import TOOLS
            llm = get_chat_model()
            _default_llm_with_tools = llm.bind_tools(tools=TOOLS)
        return _default_llm_with_tools


def reset_chat_model() -> None:
    """Clear cached chat model singletons. Useful for testing and config changes."""
    global _default_chat_model, _default_llm_with_tools
    with _lock:
        _default_chat_model = None
        _default_llm_with_tools = None
