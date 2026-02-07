"""Tests for model factory credential validation."""

from types import SimpleNamespace

import pytest


def _settings(**overrides):
    values = {
        "planner_model": "gpt-4o-mini",
        "coder_a_model": "claude-sonnet-4-20250514",
        "coder_b_model": "gpt-5.2",
        "tester_model": "gpt-4o-mini",
        "openai_api_key": "openai-test-key",
        "anthropic_api_key": "anthropic-test-key",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_planner_missing_openai_key_has_clear_message(monkeypatch):
    from app.agents import models

    monkeypatch.setattr(models, "get_settings", lambda: _settings(openai_api_key=""))

    with pytest.raises(ValueError) as exc:
        models.get_llm("planner")

    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "planner" in msg


def test_coder_a_missing_anthropic_key_has_clear_message(monkeypatch):
    from app.agents import models

    monkeypatch.setattr(models, "get_settings", lambda: _settings(anthropic_api_key=""))

    with pytest.raises(ValueError) as exc:
        models.get_llm("coder_a")

    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "coder_a" in msg


def test_planner_trims_openai_key(monkeypatch):
    from app.agents import models

    captured = {}

    def fake_make_openai(model, api_key, temperature, max_tokens):
        captured["model"] = model
        captured["api_key"] = api_key
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return {"provider": "openai", "model": model}

    monkeypatch.setattr(models, "get_settings", lambda: _settings(openai_api_key="  key-with-spaces  "))
    monkeypatch.setattr(models, "_make_openai", fake_make_openai)

    llm = models.get_llm("planner")

    assert llm["provider"] == "openai"
    assert captured["api_key"] == "key-with-spaces"


def test_coder_b_openai_model_uses_openai_factory(monkeypatch):
    from app.agents import models

    monkeypatch.setattr(models, "get_settings", lambda: _settings(coder_b_model="gpt-5.2"))
    monkeypatch.setattr(models, "_make_openai", lambda *args, **kwargs: "ok-openai")

    llm = models.get_llm("coder_b")

    assert llm == "ok-openai"
