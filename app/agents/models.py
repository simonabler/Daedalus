"""LLM model configuration and factory.

Each agent role maps to a specific provider + model.
Provides a unified interface: get_llm(role) -> BaseChatModel.

Roles:
  planner     — GPT-4o-mini (OpenAI)
  coder_a     — Claude (Anthropic) — primary coder
  coder_b     — gpt-5.2 (OpenAI)  — secondary coder
  reviewer_a  — same model as coder_a, loaded with peer-review prompt
  reviewer_b  — same model as coder_b, loaded with peer-review prompt
  tester      — GPT-4o-mini (OpenAI)
"""

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


def get_llm(role: AgentRole) -> BaseChatModel:
    """Create an LLM instance for the given agent role."""
    settings = get_settings()

    if role == "planner":
        return _make_openai(settings.planner_model, settings.openai_api_key, temperature=0.2, max_tokens=4096)

    elif role in ("coder_a", "reviewer_a"):
        # Coder A — Claude by default, but configurable
        model = settings.coder_a_model
        if _is_anthropic_model(model):
            return _make_anthropic(model, settings.anthropic_api_key, temperature=0.1)
        else:
            return _make_openai(model, settings.openai_api_key, temperature=0.1)

    elif role in ("coder_b", "reviewer_b"):
        # Coder B — gpt-5.2 by default, but configurable
        model = settings.coder_b_model
        if _is_anthropic_model(model):
            return _make_anthropic(model, settings.anthropic_api_key, temperature=0.1)
        else:
            return _make_openai(model, settings.openai_api_key, temperature=0.1)

    elif role == "tester":
        return _make_openai(settings.tester_model, settings.openai_api_key, temperature=0.0, max_tokens=4096)

    else:
        raise ValueError(f"Unknown agent role: {role}")


def load_system_prompt(role: AgentRole) -> str:
    """Load the system prompt file for a role.

    Mapping:
      planner              → supervisor_planner.txt
      coder_a              → coder_a.txt
      coder_b              → coder_b.txt
      reviewer_a           → peer_reviewer_a.txt  (Claude reviews Coder B's work)
      reviewer_b           → peer_reviewer_b.txt  (gpt-5.2 reviews Coder A's work)
      tester               → tester.txt
    """
    mapping = {
        "planner": "supervisor_planner.txt",
        "coder_a": "coder_a.txt",
        "coder_b": "coder_b.txt",
        "reviewer_a": "peer_reviewer_a.txt",
        "reviewer_b": "peer_reviewer_b.txt",
        "tester": "tester.txt",
    }
    prompt_file = PROMPTS_DIR / mapping[role]
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")
