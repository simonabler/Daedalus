"""Repository context models used by the codebase analyzer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class LanguageType(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    UNKNOWN = "unknown"


class PackageManager(StrEnum):
    PIP = "pip"
    POETRY = "poetry"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    CARGO = "cargo"
    GO = "go"
    UNKNOWN = "unknown"


class TechStack(BaseModel):
    language: LanguageType = LanguageType.UNKNOWN
    framework: str | None = None
    package_manager: PackageManager = PackageManager.UNKNOWN


class TestFramework(BaseModel):
    name: str
    unit_test_command: str | None = None


class CodeConventions(BaseModel):
    linting_tool: str | None = None
    formatting_tool: str | None = None
    max_line_length: int | None = None


class CICDConfig(BaseModel):
    platform: str
    config_files: list[str] = Field(default_factory=list)


class RepoContext(BaseModel):
    repo_path: str
    tech_stack: TechStack
    test_framework: TestFramework | None = None
    conventions: CodeConventions | None = None
    ci_cd_setup: CICDConfig | None = None
    dependencies: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    has_docker: bool = False

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)

    def get_test_command(self) -> str | None:
        if self.test_framework:
            return self.test_framework.unit_test_command
        return None
