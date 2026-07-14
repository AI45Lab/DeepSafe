from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "DeHarmScore-trace"


def schemas_module():
    sys.path.insert(0, str(PACKAGE_ROOT))
    return importlib.import_module("core_judge.schemas")


def test_model_resolves_key_and_base_url_from_environment(monkeypatch) -> None:
    schemas = schemas_module()
    monkeypatch.setenv("TEST_DEHARM_KEY", "test-token")
    monkeypatch.setenv("TEST_DEHARM_URL", "https://api.example.com/v1")

    config = schemas.ModelConfig.from_dict(
        {
            "name": "test-model",
            "base_url": "https://fallback.example.com/v1",
            "base_url_env": "TEST_DEHARM_URL",
            "api_key_env": "TEST_DEHARM_KEY",
        }
    )

    assert config.resolve_base_url() == "https://api.example.com/v1"
    assert config.resolve_api_key() == "test-token"


def test_missing_model_key_environment_variable_fails(monkeypatch) -> None:
    schemas = schemas_module()
    monkeypatch.delenv("TEST_DEHARM_KEY", raising=False)
    config = schemas.ModelConfig.from_dict(
        {
            "name": "test-model",
            "base_url": "https://api.example.com/v1",
            "api_key_env": "TEST_DEHARM_KEY",
        }
    )

    with pytest.raises(
        ValueError, match="environment variable TEST_DEHARM_KEY is not set"
    ):
        config.resolve_api_key()


def test_search_provider_resolves_key_from_environment(monkeypatch) -> None:
    schemas = schemas_module()
    monkeypatch.setenv("TEST_SEARCH_KEY", "test-search-token")
    config = schemas.SearchProviderConfig.from_dict(
        {"provider": "serper", "api_key_env": "TEST_SEARCH_KEY"}
    )

    assert config.resolve_api_key() == "test-search-token"


def test_public_default_config_has_no_local_model_path_or_literal_key() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "core_judge" / "config.yaml").read_text(encoding="utf-8")
    )

    assert "/" not in config["model"]["name"]
    assert config["model"]["api_key_env"] == "OPENAI_API_KEY"
    assert "api_key" not in config["model"]

