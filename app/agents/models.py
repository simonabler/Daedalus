"""LLM model configuration and factory.

Provider routing is done via model name prefix:
  - "ollama:<model>"  → local Ollama  (e.g. "ollama:llama3.1:70b")
  - "claude-*"        → Anthropic API
  - anything else     → OpenAI API   (e.g. "gpt-4o", "gpt-4o-mini")

Set CODER_1_MODEL and CODER_2_MODEL in .env to choose freely.
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
    "coder_a", "coder_b",   # kept as internal role names; mapped from coder_1 / coder_2
    "reviewer_a", "reviewer_b",
    "documenter",
    "tester",
]

PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _is_ollama_model(model_name: str) -> bool:
    """Models prefixed with 'ollama:' are served by local Ollama."""
    return model_name.lower().startswith("ollama:")


def _is_anthropic_model(model_name: str) -> bool:
    """Anthropic models contain 'claude' in their name."""
    return "claude" in model_name.lower()


def _strip_ollama_prefix(model_name: str) -> str:
    """Return the bare model name without the 'ollama:' prefix."""
    return model_name[len("ollama:"):]


# ---------------------------------------------------------------------------
# LLM constructors
# ---------------------------------------------------------------------------

def _make_ollama(model: str, base_url: str, temperature: float = 0.1) -> BaseChatModel:
    """Create a ChatOllama instance. langchain-ollama must be installed."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    bare_model = _strip_ollama_prefix(model)
    logger.info("Using Ollama model '%s' at %s", bare_model, base_url)
    return ChatOllama(model=bare_model, base_url=base_url, temperature=temperature)


def _make_anthropic(model: str, api_key: str, temperature: float = 0.1, max_tokens: int = 8192) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    logger.info("Using Anthropic model '%s'", model)
    return ChatAnthropic(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


def _make_openai(model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 8192) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    logger.info("Using OpenAI model '%s'", model)
    return ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Key validation helpers
# ---------------------------------------------------------------------------

def _normalized_secret(value: str | None) -> str:
    return (value or "").strip()


def _require_openai_key(role: AgentRole, model: str, api_key: str) -> str:
    key = _normalized_secret(api_key)
    if key:
        return key
    raise ValueError(
        f"Missing OPENAI_API_KEY for role '{role}' with model '{model}'. "
        "Set OPENAI_API_KEY in .env or switch to an Ollama / Anthropic model."
    )


def _require_anthropic_key(role: AgentRole, model: str, api_key: str) -> str:
    key = _normalized_secret(api_key)
    if key:
        return key
    raise ValueError(
        f"Missing ANTHROPIC_API_KEY for role '{role}' with model '{model}'. "
        "Set ANTHROPIC_API_KEY in .env or switch to an OpenAI / Ollama model."
    )


# ---------------------------------------------------------------------------
# Core builder — routes to the correct provider
# ---------------------------------------------------------------------------

def _build_for_model(
    role: AgentRole,
    model: str,
    openai_api_key: str,
    anthropic_api_key: str,
    ollama_base_url: str,
    temperature: float,
    max_tokens: int = 8192,
) -> BaseChatModel:
    """Build an LLM for *any* supported provider based on the model string."""
    if _is_ollama_model(model):
        return _make_ollama(model, base_url=ollama_base_url, temperature=temperature)

    if _is_anthropic_model(model):
        key = _require_anthropic_key(role, model, anthropic_api_key)
        return _make_anthropic(model, key, temperature=temperature, max_tokens=max_tokens)

    # Default: OpenAI-compatible
    key = _require_openai_key(role, model, openai_api_key)
    return _make_openai(model, key, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm(role: AgentRole) -> BaseChatModel:
    """Create an LLM instance for the given agent role.

    The provider is determined entirely by the model string in .env —
    no provider is hardcoded.
    """
    settings = get_settings()

    kwargs = dict(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        ollama_base_url=settings.ollama_base_url,
    )

    if role == "planner":
        return _build_for_model(role=role, model=settings.planner_model, temperature=0.2, max_tokens=4096, **kwargs)

    # coder_a / reviewer_a → uses CODER_1_MODEL
    if role in ("coder_a", "reviewer_a"):
        return _build_for_model(role=role, model=settings.coder_1_model, temperature=0.1, **kwargs)

    # coder_b / reviewer_b → uses CODER_2_MODEL
    if role in ("coder_b", "reviewer_b"):
        return _build_for_model(role=role, model=settings.coder_2_model, temperature=0.1, **kwargs)

    if role == "documenter":
        return _build_for_model(role=role, model=settings.documenter_model, temperature=0.1, max_tokens=4096, **kwargs)

    if role == "tester":
        return _build_for_model(role=role, model=settings.tester_model, temperature=0.0, max_tokens=4096, **kwargs)

    raise ValueError(f"Unknown agent role: {role}")


def load_system_prompt(role: AgentRole) -> str:
    """Load the system prompt for a role.

    Coder and reviewer prompts are assembled from a shared base file
    (``coder_base.txt`` / ``reviewer_base.txt``) plus a role-specific header,
    keeping both variants in sync automatically.
    Edit the base file — not the individual coder_a/b or reviewer files.
    """
    _CODER_BASE = "coder_base.txt"
    _REVIEWER_BASE = "reviewer_base.txt"

    _HEADERS: dict[str, tuple[str, str]] = {
        "coder_a": (
            _CODER_BASE,
            "You are **Coder A** for the Daedalus system.\n"
            "After you implement, **Coder B** will review your work.\n\n",
        ),
        "coder_b": (
            _CODER_BASE,
            "You are **Coder B** for the Daedalus system.\n"
            "After you implement, **Coder A** will review your work.\n\n",
        ),
        "reviewer_a": (
            _REVIEWER_BASE,
            "You are **Peer Reviewer A** for the Daedalus system.\n"
            "Your job: review code changes produced by **Coder B**.\n\n",
        ),
        "reviewer_b": (
            _REVIEWER_BASE,
            "You are **Peer Reviewer B** for the Daedalus system.\n"
            "Your job: review code changes produced by **Coder A**.\n\n",
        ),
    }

    if role in _HEADERS:
        base_file, header = _HEADERS[role]
        base_path = PROMPTS_DIR / base_file
        if not base_path.exists():
            raise FileNotFoundError(f"Base prompt not found: {base_path}")
        return header + base_path.read_text(encoding="utf-8")

    _DIRECT: dict[str, str] = {
        "planner": "supervisor_planner.txt",
        "documenter": "documenter.txt",
        "tester": "tester.txt",
    }
    if role not in _DIRECT:
        raise ValueError(f"Unknown agent role: {role!r}")
    prompt_file = PROMPTS_DIR / _DIRECT[role]
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")