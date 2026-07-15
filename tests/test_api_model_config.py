from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import uni_eval.models.api as api_module


def test_resolve_api_config_reads_named_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODEL_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TEST_MODEL_API_KEY", "test-token")

    base_url, api_key = api_module._resolve_api_config(
        api_base="https://fallback.example.com/v1",
        api_key="EMPTY",
        api_base_env="TEST_MODEL_BASE_URL",
        api_key_env="TEST_MODEL_API_KEY",
    )

    assert base_url == "https://api.example.com/v1"
    assert api_key == "test-token"


def test_resolve_api_config_rejects_missing_named_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODEL_API_KEY", raising=False)

    with pytest.raises(
        ValueError, match="environment variable TEST_MODEL_API_KEY is not set"
    ):
        api_module._resolve_api_config(
            api_base="https://api.example.com/v1",
            api_key="EMPTY",
            api_key_env="TEST_MODEL_API_KEY",
        )


def test_resolve_api_config_keeps_literal_public_values(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert api_module._resolve_api_config(
        api_base="https://api.example.com/v1", api_key="test-token"
    ) == ("https://api.example.com/v1", "test-token")


def test_api_model_can_disable_environment_proxy(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(Client=FakeHTTPClient)
    )
    monkeypatch.setattr(api_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")

    api_module.APIModel(
        model_name="test-model",
        api_base="https://api.example.com/v1",
        api_key="test-token",
        use_env_proxy=False,
    )

    assert created["http_client"].kwargs == {"trust_env": False}
