"""LLM model configuration and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("agents.models")

AgentRole = Literal[
    "planner",
    "coder_a", "coder_b",
    "reviewer_a", "reviewer_b",
    "documenter",
    "tester",
]

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _make_anthropic(model: str, api_key: str, temperature: float = 0.1, max_tokens: int = 8192) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


def _make_openai(model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 8192) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


def _is_anthropic_model(model_name: str) -> bool:
    """Heuristic: Anthropic models contain 'claude'."""
    return "claude" in model_name.lower()


def _normalized_secret(value: str | None) -> str:
    return (value or "").strip()


def _require_openai_key(role: AgentRole, model: str, api_key: str) -> str:
    key = _normalized_secret(api_key)
    if key:
        return key
    raise ValueError(
        f"Missing OPENAI_API_KEY for role '{role}' with model '{model}'. "
        "Set OPENAI_API_KEY in .env or switch that role to an Anthropic model."
    )


def _require_anthropic_key(role: AgentRole, model: str, api_key: str) -> str:
    key = _normalized_secret(api_key)
    if key:
        return key
    raise ValueError(
        f"Missing ANTHROPIC_API_KEY for role '{role}' with model '{model}'. "
        "Set ANTHROPIC_API_KEY in .env or switch that role to an OpenAI model "
        "(for example CODER_A_MODEL=gpt-5.2)."
    )


def _build_for_model(
    role: AgentRole,
    model: str,
    openai_api_key: str,
    anthropic_api_key: str,
    temperature: float,
    max_tokens: int = 8192,
) -> BaseChatModel:
    if _is_anthropic_model(model):
        key = _require_anthropic_key(role, model, anthropic_api_key)
        return _make_anthropic(model, key, temperature=temperature, max_tokens=max_tokens)

    key = _require_openai_key(role, model, openai_api_key)
    return _make_openai(model, key, temperature=temperature, max_tokens=max_tokens)


def get_llm(role: AgentRole) -> BaseChatModel:
    """Create an LLM instance for the given agent role."""
    settings = get_settings()

    if role == "planner":
        model = settings.planner_model
        key = _require_openai_key(role, model, settings.openai_api_key)
        return _make_openai(model, key, temperature=0.2, max_tokens=4096)

    if role in ("coder_a", "reviewer_a"):
        return _build_for_model(
            role=role,
            model=settings.coder_a_model,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.1,
        )

    if role in ("coder_b", "reviewer_b"):
        return _build_for_model(
            role=role,
            model=settings.coder_b_model,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.1,
        )

    if role == "documenter":
        model = settings.documenter_model
        return _build_for_model(
            role=role,
            model=model,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.1,
            max_tokens=4096,
        )

    if role == "tester":
        model = settings.tester_model
        key = _require_openai_key(role, model, settings.openai_api_key)
        return _make_openai(model, key, temperature=0.0, max_tokens=4096)

    raise ValueError(f"Unknown agent role: {role}")


def load_system_prompt(role: AgentRole) -> str:
    """Load the system prompt file for a role."""
    mapping = {
        "planner": "supervisor_planner.txt",
        "coder_a": "coder_a.txt",
        "coder_b": "coder_b.txt",
        "reviewer_a": "peer_reviewer_a.txt",
        "reviewer_b": "peer_reviewer_b.txt",
        "documenter": "documenter.txt",
        "tester": "tester.txt",
    }
    prompt_file = PROMPTS_DIR / mapping[role]
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")
