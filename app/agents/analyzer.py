"""Static repository analyzer used by the context loader node."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.logging import get_logger
from app.core.repo_context import (
    CICDConfig,
    CodeConventions,
    LanguageType,
    PackageManager,
    RepoContext,
    TechStack,
    TestFramework,
)

logger = get_logger("agents.analyzer")

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


class CodebaseAnalyzer:
    """Lightweight, deterministic analyzer for repo tech facts."""

    def __init__(self, repo_path: Path):
        if not repo_path.exists() or not repo_path.is_dir():
            raise ValueError(f"Invalid repository path: {repo_path}")
        self.repo_path = repo_path.resolve()

    def analyze_repository(self) -> RepoContext:
        tech_stack = self._detect_tech_stack()
        test_framework = self._detect_test_framework(tech_stack.language)
        conventions = self._detect_conventions(tech_stack.language)
        ci_cd = self._detect_ci_cd()
        dependencies = self._collect_dependencies(tech_stack.language)
        entry_points = self._detect_entrypoints()

        return RepoContext(
            repo_path=str(self.repo_path),
            tech_stack=tech_stack,
            test_framework=test_framework,
            conventions=conventions,
            ci_cd_setup=ci_cd,
            dependencies=dependencies,
            entry_points=entry_points,
            has_docker=(self.repo_path / "Dockerfile").exists(),
        )

    def _detect_tech_stack(self) -> TechStack:
        if (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
            framework = self._detect_python_framework()
            manager = PackageManager.POETRY if (self.repo_path / "poetry.lock").exists() else PackageManager.PIP
            return TechStack(language=LanguageType.PYTHON, framework=framework, package_manager=manager)

        if (self.repo_path / "package.json").exists():
            pkg = self._read_json(self.repo_path / "package.json")
            dependencies = self._merged_npm_deps(pkg)
            language = (
                LanguageType.TYPESCRIPT
                if (self.repo_path / "tsconfig.json").exists()
                else LanguageType.JAVASCRIPT
            )
            if "typescript" in dependencies:
                language = LanguageType.TYPESCRIPT
            framework = self._detect_js_framework(dependencies)
            manager = PackageManager.NPM
            if (self.repo_path / "yarn.lock").exists():
                manager = PackageManager.YARN
            elif (self.repo_path / "pnpm-lock.yaml").exists():
                manager = PackageManager.PNPM
            return TechStack(language=language, framework=framework, package_manager=manager)

        if (self.repo_path / "go.mod").exists():
            return TechStack(language=LanguageType.GO, framework=None, package_manager=PackageManager.GO)
        if (self.repo_path / "Cargo.toml").exists():
            return TechStack(language=LanguageType.RUST, framework=None, package_manager=PackageManager.CARGO)

        return TechStack()

    def _detect_python_framework(self) -> str | None:
        deps = self._python_dependencies()
        if "fastapi" in deps:
            return "FastAPI"
        if "django" in deps:
            return "Django"
        if "flask" in deps:
            return "Flask"
        return None

    def _detect_js_framework(self, deps: dict) -> str | None:
        if "next" in deps:
            return "Next.js"
        if "react" in deps:
            return "React"
        if "vue" in deps:
            return "Vue"
        if "express" in deps:
            return "Express"
        if "@nestjs/core" in deps:
            return "NestJS"
        return None

    def _detect_test_framework(self, language: LanguageType) -> TestFramework | None:
        if language == LanguageType.PYTHON:
            if (self.repo_path / "pytest.ini").exists():
                return TestFramework(name="pytest", unit_test_command="python -m pytest -q")
            pyproject = self._read_toml(self.repo_path / "pyproject.toml")
            testpaths = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths")
            if testpaths:
                return TestFramework(name="pytest", unit_test_command="python -m pytest -q")
            return TestFramework(name="unittest", unit_test_command="python -m unittest discover")

        if language in {LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT}:
            pkg = self._read_json(self.repo_path / "package.json")
            scripts = pkg.get("scripts", {}) if isinstance(pkg, dict) else {}
            command = scripts.get("test", "npm test") if isinstance(scripts, dict) else "npm test"
            deps = self._merged_npm_deps(pkg)
            if "vitest" in deps:
                return TestFramework(name="vitest", unit_test_command=command)
            if "jest" in deps:
                return TestFramework(name="jest", unit_test_command=command)
            return TestFramework(name="npm-test", unit_test_command=command)

        return None

    def _detect_conventions(self, language: LanguageType) -> CodeConventions | None:
        if language == LanguageType.PYTHON:
            lint = None
            fmt = None
            line_length = None
            pyproject = self._read_toml(self.repo_path / "pyproject.toml")
            tool_cfg = pyproject.get("tool", {})
            if isinstance(tool_cfg, dict):
                ruff_cfg = tool_cfg.get("ruff")
                if isinstance(ruff_cfg, dict):
                    lint = "ruff"
                    line_length = ruff_cfg.get("line-length")
                black_cfg = tool_cfg.get("black")
                if isinstance(black_cfg, dict):
                    fmt = "black"
                    line_length = line_length or black_cfg.get("line-length")
            if (self.repo_path / "ruff.toml").exists():
                lint = lint or "ruff"
            if not fmt and lint == "ruff":
                fmt = "ruff"
            return CodeConventions(linting_tool=lint, formatting_tool=fmt, max_line_length=line_length)

        if language in {LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT}:
            lint = "eslint" if (self.repo_path / ".eslintrc.json").exists() else None
            if (self.repo_path / "eslint.config.js").exists():
                lint = "eslint"
            fmt = "prettier" if (self.repo_path / ".prettierrc").exists() else None
            return CodeConventions(linting_tool=lint, formatting_tool=fmt, max_line_length=None)

        return None

    def _detect_ci_cd(self) -> CICDConfig | None:
        gh = self.repo_path / ".github" / "workflows"
        if gh.exists() and gh.is_dir():
            files = sorted(str(p.relative_to(self.repo_path)) for p in gh.glob("*.y*ml"))
            return CICDConfig(platform="github_actions", config_files=files)
        gitlab = self.repo_path / ".gitlab-ci.yml"
        if gitlab.exists():
            return CICDConfig(platform="gitlab_ci", config_files=[".gitlab-ci.yml"])
        return None

    def _collect_dependencies(self, language: LanguageType) -> list[str]:
        if language == LanguageType.PYTHON:
            return sorted(self._python_dependencies().keys())[:200]
        if language in {LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT}:
            pkg = self._read_json(self.repo_path / "package.json")
            return sorted(self._merged_npm_deps(pkg).keys())[:200]
        return []

    def _python_dependencies(self) -> dict:
        deps: dict[str, str] = {}
        pyproject = self._read_toml(self.repo_path / "pyproject.toml")
        project = pyproject.get("project", {})
        project_deps = project.get("dependencies", []) if isinstance(project, dict) else []
        if isinstance(project_deps, list):
            for item in project_deps:
                name = str(item).split(" ", 1)[0].split("[", 1)[0].split("=", 1)[0]
                name = name.split("<", 1)[0].split(">", 1)[0].strip()
                if name:
                    deps[name.lower()] = ""
        poetry = pyproject.get("tool", {}).get("poetry", {}) if isinstance(pyproject.get("tool"), dict) else {}
        poetry_deps = poetry.get("dependencies", {}) if isinstance(poetry, dict) else {}
        if isinstance(poetry_deps, dict):
            for key in poetry_deps:
                if key.lower() != "python":
                    deps[str(key).lower()] = ""
        return deps

    def _merged_npm_deps(self, pkg: dict) -> dict:
        if not isinstance(pkg, dict):
            return {}
        dependencies = pkg.get("dependencies", {})
        dev_dependencies = pkg.get("devDependencies", {})
        merged = {}
        if isinstance(dependencies, dict):
            merged.update(dependencies)
        if isinstance(dev_dependencies, dict):
            merged.update(dev_dependencies)
        return merged

    def _detect_entrypoints(self) -> list[str]:
        candidates = [
            "app/main.py",
            "main.py",
            "src/main.ts",
            "src/index.ts",
            "src/index.js",
            "server.js",
            "server.ts",
        ]
        return [rel for rel in candidates if (self.repo_path / rel).exists()]

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Could not parse JSON %s: %s", path, exc)
            return {}

    def _read_toml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.debug("Could not parse TOML %s: %s", path, exc)
            return {}
