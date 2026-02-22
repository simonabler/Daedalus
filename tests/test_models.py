"""Tests for model factory credential validation and provider routing."""

from types import SimpleNamespace

import pytest


def _settings(**overrides):
    values = {
        "planner_model": "gpt-4o-mini",
        "coder_1_model": "gpt-4o-mini",
        "coder_2_model": "gpt-4o-mini",
        "documenter_model": "gpt-4o-mini",
        "tester_model": "gpt-4o-mini",
        "openai_api_key": "openai-test-key",
        "anthropic_api_key": "anthropic-test-key",
        "ollama_base_url": "http://localhost:11434",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_openai_model_uses_openai_factory(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(coder_1_model="gpt-4o"))
    monkeypatch.setattr(models, "_make_openai", lambda *args, **kwargs: "ok-openai")
    assert models.get_llm("coder_a") == "ok-openai"


def test_anthropic_model_uses_anthropic_factory(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(coder_1_model="claude-opus-4-5"))
    monkeypatch.setattr(models, "_make_anthropic", lambda *args, **kwargs: "ok-anthropic")
    assert models.get_llm("coder_a") == "ok-anthropic"


def test_ollama_model_uses_ollama_factory(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(coder_2_model="ollama:llama3.1:70b"))
    monkeypatch.setattr(models, "_make_ollama", lambda *args, **kwargs: "ok-ollama")
    assert models.get_llm("coder_b") == "ok-ollama"


def test_coder_2_independently_configurable(monkeypatch):
    from app.agents import models
    settings = _settings(coder_1_model="gpt-4o", coder_2_model="ollama:deepseek-coder-v2")
    monkeypatch.setattr(models, "get_settings", lambda: settings)
    monkeypatch.setattr(models, "_make_openai", lambda *args, **kwargs: "openai")
    monkeypatch.setattr(models, "_make_ollama", lambda *args, **kwargs: "ollama")
    assert models.get_llm("coder_a") == "openai"
    assert models.get_llm("coder_b") == "ollama"


def test_planner_missing_openai_key_has_clear_message(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(openai_api_key=""))
    with pytest.raises(ValueError) as exc:
        models.get_llm("planner")
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "planner" in msg


def test_anthropic_model_missing_key_has_clear_message(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(
        coder_1_model="claude-sonnet-4-20250514",
        anthropic_api_key="",
    ))
    with pytest.raises(ValueError) as exc:
        models.get_llm("coder_a")
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "coder_a" in msg


def test_ollama_does_not_require_any_api_key(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(
        coder_1_model="ollama:llama3.1",
        openai_api_key="",
        anthropic_api_key="",
    ))
    monkeypatch.setattr(models, "_make_ollama", lambda *args, **kwargs: "ok-ollama")
    assert models.get_llm("coder_a") == "ok-ollama"


def test_is_ollama_model_prefix():
    from app.agents.models import _is_ollama_model
    assert _is_ollama_model("ollama:llama3.1:70b") is True
    assert _is_ollama_model("gpt-4o") is False
    assert _is_ollama_model("claude-opus-4-5") is False


def test_is_anthropic_model_contains_claude():
    from app.agents.models import _is_anthropic_model
    assert _is_anthropic_model("claude-opus-4-5") is True
    assert _is_anthropic_model("gpt-4o") is False
    assert _is_anthropic_model("ollama:llama3.1") is False


def test_strip_ollama_prefix():
    from app.agents.models import _strip_ollama_prefix
    assert _strip_ollama_prefix("ollama:llama3.1:70b") == "llama3.1:70b"
    assert _strip_ollama_prefix("ollama:deepseek-coder-v2") == "deepseek-coder-v2"


def test_documenter_uses_configured_model(monkeypatch):
    from app.agents import models
    monkeypatch.setattr(models, "get_settings", lambda: _settings(documenter_model="gpt-4o-mini"))
    monkeypatch.setattr(models, "_make_openai", lambda *args, **kwargs: "ok-doc")
    assert models.get_llm("documenter") == "ok-doc"
