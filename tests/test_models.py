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


# ─────────────────────────────────────────────────────────────────────────────
# load_system_prompt — composed prompts and base-file assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSystemPromptComposed:
    """load_system_prompt() assembles coder/reviewer prompts from shared base files."""

    def test_coder_a_header_mentions_coder_b(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("coder_a")
        assert "Coder A" in p
        assert "Coder B" in p

    def test_coder_b_header_mentions_coder_a(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("coder_b")
        assert "Coder B" in p
        assert "Coder A" in p

    def test_coder_a_and_b_share_same_body(self):
        """After stripping the first two lines (header), both prompts are identical."""
        from app.agents.models import load_system_prompt
        a = "\n".join(load_system_prompt("coder_a").splitlines()[2:])
        b = "\n".join(load_system_prompt("coder_b").splitlines()[2:])
        assert a == b, "coder_a and coder_b body diverged — edit coder_base.txt, not individual files"

    def test_reviewer_a_header_mentions_coder_b(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("reviewer_a")
        assert "Reviewer A" in p
        assert "Coder B" in p

    def test_reviewer_b_header_mentions_coder_a(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("reviewer_b")
        assert "Reviewer B" in p
        assert "Coder A" in p

    def test_reviewer_a_and_b_share_same_body(self):
        from app.agents.models import load_system_prompt
        a = "\n".join(load_system_prompt("reviewer_a").splitlines()[2:])
        b = "\n".join(load_system_prompt("reviewer_b").splitlines()[2:])
        assert a == b, "reviewer_a and reviewer_b body diverged — edit reviewer_base.txt"

    def test_planner_prompt_has_json_output_contract(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("planner")
        assert '"plan"' in p
        assert "task_type" in p
        assert "acceptance_criteria" in p
        assert "STRICT JSON" in p

    def test_tester_prompt_documents_env_fix_routing(self):
        from app.agents.models import load_system_prompt
        p = load_system_prompt("tester")
        assert "ENVIRONMENT" in p
        assert "ModuleNotFoundError" in p

    def test_coder_prompts_have_issue_context_section(self):
        from app.agents.models import load_system_prompt
        for role in ("coder_a", "coder_b"):
            p = load_system_prompt(role)
            assert "Issue Context" in p, f"{role} prompt missing Issue Context section"

    def test_reviewer_prompts_mention_planner_gate(self):
        from app.agents.models import load_system_prompt
        for role in ("reviewer_a", "reviewer_b"):
            p = load_system_prompt(role)
            assert "Planner" in p, f"{role} prompt should mention the Planner final-review gate"

    def test_unknown_role_raises_value_error(self):
        from app.agents.models import load_system_prompt
        import pytest
        with pytest.raises(ValueError, match="Unknown agent role"):
            load_system_prompt("ghost_agent")


# ─────────────────────────────────────────────────────────────────────────────
# _os_note — platform-aware shell hint
# ─────────────────────────────────────────────────────────────────────────────

class TestOsNote:
    """_os_note() returns a correct shell hint for each platform."""

    def _get(self, plat: str) -> str:
        from app.core.nodes import _os_note
        return _os_note(plat)

    def test_windows_returns_powershell(self):
        note = self._get("Windows-10-10.0.19041-SP0")
        assert "Windows" in note
        assert "PowerShell" in note

    def test_macos_returns_bash_zsh(self):
        note = self._get("macOS-14.0-arm64")
        assert "macOS" in note
        assert "bash" in note.lower() or "zsh" in note.lower()

    def test_linux_returns_bash(self):
        note = self._get("Linux-6.1.0-amd64")
        assert "Linux" in note
        assert "bash" in note.lower()

    def test_empty_string_defaults_to_linux(self):
        note = self._get("")
        assert "Linux" in note

    def test_windows_does_not_say_bash(self):
        note = self._get("Windows-11")
        assert "bash" not in note.lower()

    def test_linux_does_not_say_powershell(self):
        note = self._get("Linux-5.15")
        assert "powershell" not in note.lower()
