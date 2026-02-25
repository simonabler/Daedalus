"""Tests for the shared long-term memory system."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repo directory and patch settings to use it."""
    from app.core.active_repo import set_repo_root, clear_repo_root
    set_repo_root(str(tmp_path))
    with patch("app.core.memory.get_settings") as mock:
        mock.return_value.target_repo_path = str(tmp_path)
        yield tmp_path
    clear_repo_root()


class TestEnsureMemoryFiles:
    def test_creates_directory_and_seeds(self, temp_repo):
        from app.core.memory import ensure_memory_files, MEMORY_FILES
        ensure_memory_files()

        for key, rel_path in MEMORY_FILES.items():
            path = temp_repo / rel_path
            assert path.exists(), f"Missing: {rel_path}"
            content = path.read_text()
            assert len(content) > 10

    def test_idempotent(self, temp_repo):
        from app.core.memory import ensure_memory_files, load_memory, append_memory
        ensure_memory_files()
        append_memory("coding_style", "Use snake_case for functions")
        ensure_memory_files()  # should not overwrite
        content = load_memory("coding_style")
        assert "snake_case" in content


class TestLoadMemory:
    def test_load_existing(self, temp_repo):
        from app.core.memory import ensure_memory_files, load_memory
        ensure_memory_files()
        content = load_memory("coding_style")
        assert "Coding Style Guide" in content

    def test_load_missing(self, temp_repo):
        from app.core.memory import load_memory
        content = load_memory("coding_style")
        assert content == ""

    def test_load_all_memory(self, temp_repo):
        from app.core.memory import ensure_memory_files, load_all_memory
        ensure_memory_files()
        combined = load_all_memory()
        assert "SHARED LONG-TERM MEMORY" in combined
        assert "Coding Style Guide" in combined
        assert "Architecture Decisions" in combined
        assert "Shared Insights" in combined


class TestAppendMemory:
    def test_append_adds_entry(self, temp_repo):
        from app.core.memory import ensure_memory_files, append_memory, load_memory
        ensure_memory_files()
        append_memory("coding_style", "Always use type hints", item_id="item-001")
        content = load_memory("coding_style")
        assert "Always use type hints" in content
        assert "item-001" in content

    def test_append_multiple(self, temp_repo):
        from app.core.memory import ensure_memory_files, append_memory, load_memory
        ensure_memory_files()
        append_memory("insights", "Tests in /tests/integration need Docker")
        append_memory("insights", "The config module auto-loads .env")
        content = load_memory("insights")
        assert "Docker" in content
        assert "auto-loads" in content

    def test_append_creates_file_if_missing(self, temp_repo):
        from app.core.memory import append_memory, load_memory
        append_memory("architecture", "Use Repository pattern for data access")
        content = load_memory("architecture")
        assert "Repository pattern" in content


class TestMemoryStats:
    def test_stats_empty(self, temp_repo):
        from app.core.memory import ensure_memory_files, get_memory_stats
        ensure_memory_files()
        stats = get_memory_stats()
        assert "coding_style" in stats
        assert "architecture" in stats
        assert "insights" in stats
        for key, info in stats.items():
            assert "chars" in info
            assert "needs_compression" in info
            assert info["needs_compression"] is False

    def test_needs_compression(self, temp_repo):
        from app.core.memory import ensure_memory_files, append_memory, memory_needs_compression
        ensure_memory_files()
        # Append a lot of content
        for i in range(200):
            append_memory("coding_style", f"Rule {i}: " + "x" * 50)
        assert memory_needs_compression("coding_style") is True


class TestCompressionPrompt:
    def test_no_compression_needed(self, temp_repo):
        from app.core.memory import ensure_memory_files, build_compression_prompt
        ensure_memory_files()
        prompt = build_compression_prompt("coding_style")
        assert prompt is None

    def test_compression_prompt_generated(self, temp_repo):
        from app.core.memory import ensure_memory_files, append_memory, build_compression_prompt
        ensure_memory_files()
        for i in range(200):
            append_memory("coding_style", f"Rule {i}: " + "x" * 50)
        prompt = build_compression_prompt("coding_style")
        assert prompt is not None
        assert "Memory Compression Task" in prompt
        assert "Coding Style Guide" in prompt


class TestParseLearnings:
    def test_parse_valid_json(self):
        from app.core.nodes import _parse_learnings
        result = json.dumps({
            "coding_style": ["Use snake_case"],
            "architecture": [],
            "insights": ["Config auto-loads .env"],
        })
        parsed = _parse_learnings(result)
        assert len(parsed["coding_style"]) == 1
        assert len(parsed["insights"]) == 1

    def test_parse_json_in_markdown_fence(self):
        from app.core.nodes import _parse_learnings
        result = '```json\n{"coding_style": ["rule1"], "architecture": [], "insights": []}\n```'
        parsed = _parse_learnings(result)
        assert parsed["coding_style"] == ["rule1"]

    def test_parse_empty(self):
        from app.core.nodes import _parse_learnings
        result = '{"coding_style": [], "architecture": [], "insights": []}'
        parsed = _parse_learnings(result)
        assert all(len(v) == 0 for v in parsed.values())

    def test_parse_garbage(self):
        from app.core.nodes import _parse_learnings
        parsed = _parse_learnings("This is not JSON at all")
        assert parsed == {}

    def test_parse_json_embedded_in_text(self):
        from app.core.nodes import _parse_learnings
        result = 'Here are my findings:\n{"coding_style": ["test"], "architecture": [], "insights": []}\nDone.'
        parsed = _parse_learnings(result)
        assert parsed["coding_style"] == ["test"]
