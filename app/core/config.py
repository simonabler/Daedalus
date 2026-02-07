"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for daedalus. Reads from .env automatically."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── LLM keys ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── LLM model identifiers ────────────────────────────────────────────
    planner_model: str = "gpt-4o-mini"
    coder_a_model: str = "claude-sonnet-4-20250514"   # Coder A — Claude
    coder_b_model: str = "gpt-5.2"                     # Coder B — gpt-5.2
    tester_model: str = "gpt-4o-mini"

    # Legacy alias (kept for backwards compat if referenced elsewhere)
    @property
    def coder_model(self) -> str:
        return self.coder_a_model

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""

    @property
    def allowed_telegram_ids(self) -> list[int]:
        if not self.telegram_allowed_user_ids:
            return []
        return [int(uid.strip()) for uid in self.telegram_allowed_user_ids.split(",") if uid.strip()]

    # ── Web UI ────────────────────────────────────────────────────────────
    web_host: str = "127.0.0.1"
    web_port: int = 8420

    # ── Target repo ───────────────────────────────────────────────────────
    target_repo_path: str = ""

    @field_validator("target_repo_path")
    @classmethod
    def _resolve_repo(cls, v: str) -> str:
        if v:
            return str(Path(v).expanduser().resolve())
        return v

    # ── Git ────────────────────────────────────────────────────────────────
    git_author_name: str = "daedalus"
    git_author_email: str = "daedalus@local"

    # ── Safety ─────────────────────────────────────────────────────────────
    max_iterations_per_item: int = 5
    shell_timeout_seconds: int = 120
    max_output_chars: int = 12_000

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton accessor."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
