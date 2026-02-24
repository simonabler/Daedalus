"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for daedalus. Reads from .env automatically."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # LLM API keys (only required for the providers you actually use)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Ollama base URL — set this when using local Ollama models
    # Example: OLLAMA_BASE_URL=http://localhost:11434
    ollama_base_url: str = "http://localhost:11434"

    # Model identifiers — prefix determines the provider:
    #   "ollama:<model>"     → local Ollama  (e.g. "ollama:llama3.1:70b")
    #   "claude-*" / "claude" → Anthropic API
    #   anything else        → OpenAI API    (e.g. "gpt-4o", "gpt-4o-mini")
    planner_model: str = "gpt-4o-mini"
    coder_1_model: str = "gpt-4o-mini"   # Coder 1 — configure freely
    coder_2_model: str = "gpt-4o-mini"   # Coder 2 — configure freely
    documenter_model: str = "gpt-4o-mini"
    tester_model: str = "gpt-4o-mini"

    # Legacy aliases — kept so existing code that references coder_a/coder_b still works
    @property
    def coder_a_model(self) -> str:
        return self.coder_1_model

    @property
    def coder_b_model(self) -> str:
        return self.coder_2_model

    @property
    def coder_model(self) -> str:
        return self.coder_1_model

    # Telegram
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""

    @property
    def allowed_telegram_ids(self) -> list[int]:
        if not self.telegram_allowed_user_ids:
            return []
        return [int(uid.strip()) for uid in self.telegram_allowed_user_ids.split(",") if uid.strip()]

    # Web UI
    web_host: str = "127.0.0.1"
    web_port: int = 8420

    # Target repo
    target_repo_path: str = ""

    @field_validator("target_repo_path")
    @classmethod
    def _resolve_repo(cls, value: str) -> str:
        if value:
            return str(Path(value).expanduser().resolve())
        return value

    # Git
    git_author_name: str = "daedalus"
    git_author_email: str = "daedalus@local"

    # Safety
    max_iterations_per_item: int = 5
    max_rework_cycles_per_item: int = 3
    shell_timeout_seconds: int = 120
    max_output_chars: int = 12_000

    # Token budget / cost limits (USD). Set to 0.0 to disable.
    # Soft limit: emits a ⚠️ warning in the UI but workflow continues.
    # Hard limit: stops the workflow with stop_reason="budget_hard_limit_exceeded".
    token_budget_soft_limit_usd: float = 0.0
    token_budget_hard_limit_usd: float = 0.0

    # Context window management
    # Trigger message-history compression when this fraction of the model's
    # context window is estimated to be used. 0.0 = never compress.
    context_warn_fraction: float = 0.75
    # Number of most-recent messages always kept intact during compression.
    context_keep_recent_messages: int = 6
    # Hard limit on a single tool result's char count before it enters all_messages.
    # Prevents one large read_file call from consuming the context window.
    tool_result_max_chars: int = 8_000

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton accessor."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
