"""
Unit tests for the decoupled LLM factory and Grok (xAI) / OpenAI provider support.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.core import llm
from app.core.retry import RETRYABLE_LLM_ERRORS, RETRYABLE_OPENAI_ERRORS
from app.graph.tools import TOOLS


@pytest.fixture(autouse=True)
def clean_llm_state(monkeypatch):
    """Ensure clean environment and cached model singletons before and after each test."""
    llm.reset_chat_model()
    # Clear LLM-related environment variables
    for var in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_TEMPERATURE",
        "GROK_API_KEY",
        "XAI_API_KEY",
        "GROK_MODEL",
        "XAI_MODEL",
        "GROK_BASE_URL",
        "XAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    llm.reset_chat_model()


def test_default_settings_resolve_to_openai():
    settings = llm.get_llm_settings()
    assert settings.provider == "openai"
    assert settings.model == "gpt-4.1"
    assert settings.base_url is None
    assert settings.temperature == 0.0


def test_grok_settings_via_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setenv("GROK_API_KEY", "xai-dummy-key-123")
    settings = llm.get_llm_settings()

    assert settings.provider == "grok"
    assert settings.model == "grok-2"
    assert settings.api_key == "xai-dummy-key-123"
    assert settings.base_url == "https://api.x.ai/v1"


def test_xai_provider_alias_resolves_to_grok(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "XAI")
    monkeypatch.setenv("XAI_API_KEY", "xai-key-456")
    settings = llm.get_llm_settings()

    assert settings.provider == "grok"
    assert settings.model == "grok-2"
    assert settings.api_key == "xai-key-456"
    assert settings.base_url == "https://api.x.ai/v1"


def test_auto_detect_grok_when_grok_key_present_without_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-detected-key")
    settings = llm.get_llm_settings()

    assert settings.provider == "grok"
    assert settings.api_key == "xai-detected-key"


def test_custom_model_and_base_url_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setenv("LLM_MODEL", "grok-2-latest")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom.x.ai/v1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")

    settings = llm.get_llm_settings()
    assert settings.provider == "grok"
    assert settings.model == "grok-2-latest"
    assert settings.base_url == "https://custom.x.ai/v1"
    assert settings.temperature == 0.7


def test_create_chat_model_openai():
    model = llm.create_chat_model(
        provider="openai",
        model="gpt-4.1",
        api_key="sk-test-fake",
    )
    assert model is not None
    assert hasattr(model, "invoke")


def test_create_chat_model_grok():
    model = llm.create_chat_model(
        provider="grok",
        model="grok-2",
        api_key="xai-test-fake",
    )
    assert model is not None
    assert hasattr(model, "invoke")


def test_get_llm_with_tools_binds_tools():
    bound = llm.get_llm_with_tools(tools=TOOLS)
    assert bound is not None
    assert hasattr(bound, "invoke")


def test_reset_chat_model_clears_cached_instances():
    m1 = llm.get_chat_model(api_key="sk-fake-1")
    assert llm._default_chat_model is None  # had overrides, not cached

    # Without overrides, it should cache singleton
    m2 = llm.get_chat_model()
    m3 = llm.get_chat_model()
    assert m2 is m3

    llm.reset_chat_model()
    assert llm._default_chat_model is None
    m4 = llm.get_chat_model()
    assert m4 is not m2


def test_retryable_llm_errors_matches_retryable_openai_errors():
    assert RETRYABLE_LLM_ERRORS == RETRYABLE_OPENAI_ERRORS
    assert len(RETRYABLE_LLM_ERRORS) == 3
