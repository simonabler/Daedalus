"""Tests for the Documenter Agent node (Issue #28).

Covers:
- WorkflowPhase.DOCUMENTING presence
- _diff_needs_docs heuristic (true/false for various diffs)
- documenter_node: no-op path (no diff, uninteresting diff)
- documenter_node: LLM call path (documentation-worthy diff)
- Orchestrator wiring: _route_after_commit → documenter
- Orchestrator wiring: _route_after_documenter → coder | complete
- Graph contains documenter node
- documenter.txt prompt completeness
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.core.state import GraphState, WorkflowPhase


# ---------------------------------------------------------------------------
# 1. WorkflowPhase.DOCUMENTING
# ---------------------------------------------------------------------------

class TestWorkflowPhaseDocumenting:
    def test_documenting_in_enum(self):
        assert hasattr(WorkflowPhase, "DOCUMENTING")
        assert WorkflowPhase.DOCUMENTING == "documenting"

    def test_documenting_is_str(self):
        assert isinstance(WorkflowPhase.DOCUMENTING, str)


# ---------------------------------------------------------------------------
# 2. _diff_needs_docs heuristic
# ---------------------------------------------------------------------------

class TestDiffNeedsDocs:
    def _check(self, diff: str) -> bool:
        from app.core.nodes import _diff_needs_docs
        return _diff_needs_docs(diff)

    def test_new_public_function_triggers(self):
        diff = "+def calculate_total(items: list) -> float:\n"
        assert self._check(diff) is True

    def test_new_async_function_triggers(self):
        diff = "+async def fetch_data(url: str):\n"
        assert self._check(diff) is True

    def test_new_public_class_triggers(self):
        diff = "+class TokenBudget:\n"
        assert self._check(diff) is True

    def test_new_api_endpoint_triggers(self):
        diff = '+@app.get("/api/health")\n'
        assert self._check(diff) is True

    def test_new_router_endpoint_triggers(self):
        diff = '+@router.post("/api/tasks")\n'
        assert self._check(diff) is True

    def test_new_constant_triggers(self):
        diff = "+MAX_RETRIES = 5\n"
        assert self._check(diff) is True

    def test_private_function_does_not_trigger(self):
        diff = "+def _internal_helper():\n"
        assert self._check(diff) is False

    def test_test_only_change_does_not_trigger(self):
        diff = "+    assert result == 42\n+    mock.assert_called_once()\n"
        assert self._check(diff) is False

    def test_empty_diff_does_not_trigger(self):
        assert self._check("") is False

    def test_comment_only_change_does_not_trigger(self):
        diff = "+# Fix typo in variable name\n"
        assert self._check(diff) is False

    def test_multiline_diff_with_trigger(self):
        diff = (
            " context line\n"
            "+def process_batch(items):\n"
            "+    return [x * 2 for x in items]\n"
        )
        assert self._check(diff) is True

    def test_deletion_only_does_not_trigger(self):
        # Lines starting with '-' are removals — we only look at additions '+'
        diff = "-def old_function():\n-    pass\n"
        assert self._check(diff) is False


# ---------------------------------------------------------------------------
# 3. documenter_node — no-op paths
# ---------------------------------------------------------------------------

class TestDocumenterNodeNoOp:
    def _run_node(self, diff_output: str) -> dict:
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = diff_output
            result = documenter_node(state)
            clear_listeners()

        return result, mock_llm

    def test_empty_diff_returns_empty_dict(self):
        result, mock_llm = self._run_node("")
        assert result == {}
        mock_llm.assert_not_called()

    def test_non_doc_diff_no_llm_call(self):
        diff = "+    assert x == 1\n+    mock.reset_mock()\n"
        result, mock_llm = self._run_node(diff)
        assert result == {}
        mock_llm.assert_not_called()

    def test_git_command_failure_returns_empty(self):
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        state = GraphState(phase=WorkflowPhase.CODING)
        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.side_effect = RuntimeError("git error")
            result = documenter_node(state)
            clear_listeners()

        assert result == {}
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# 4. documenter_node — LLM call path
# ---------------------------------------------------------------------------

class TestDocumenterNodeLLMPath:
    def test_doc_worthy_diff_calls_llm(self):
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        diff = "+def new_public_api(x: int) -> str:\n+    return str(x)\n"
        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = diff
            mock_llm.return_value = "CHANGED FILES:\n- CHANGELOG.md: added entry\n"
            result = documenter_node(state)
            clear_listeners()

        mock_llm.assert_called_once()
        assert result.get("phase") is None  # documenter never mutates phase

    def test_llm_called_with_documenter_role(self):
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        diff = "+class NewFeature:\n+    pass\n"
        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = diff
            mock_llm.return_value = "done"
            documenter_node(state)
            clear_listeners()

        call_args = mock_llm.call_args
        assert call_args[0][0] == "documenter"

    def test_llm_called_with_documenter_tools(self):
        from app.core.nodes import documenter_node, DOCUMENTER_TOOLS
        from app.core.events import clear_listeners

        diff = "+@app.get('/health')\n+def health():\n+    return {'ok': True}\n"
        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = diff
            mock_llm.return_value = "done"
            documenter_node(state)
            clear_listeners()

        call_kwargs = mock_llm.call_args[0]
        # Third positional arg is tools
        assert call_kwargs[2] == DOCUMENTER_TOOLS

    def test_diff_truncated_to_8000_chars(self):
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        # Create a diff that's longer than 8000 chars but triggers docs
        long_diff = "+def new_func():\n" + ("+    x = 1\n" * 900)
        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = long_diff
            mock_llm.return_value = "done"
            documenter_node(state)
            clear_listeners()

        # The prompt passed to _invoke_agent should have truncated diff
        call_args = mock_llm.call_args
        messages = call_args[0][1]
        prompt_content = messages[0].content
        # 8000 chars for diff + surrounding text — total prompt < 10000
        assert len(prompt_content) < 10000

    def test_documenter_does_not_mutate_phase(self):
        """documenter_node never changes phase or workflow state — only updates token_budget."""
        from app.core.nodes import documenter_node
        from app.core.events import clear_listeners

        diff = "+def public_fn():\n    pass\n"
        state = GraphState(phase=WorkflowPhase.CODING)

        with patch("app.core.nodes.documenter.git_command") as mock_git, \
             patch("app.core.nodes._helpers._invoke_agent") as mock_llm:
            mock_git.invoke.return_value = diff
            mock_llm.return_value = "done"
            result = documenter_node(state)
            clear_listeners()

        # documenter must never set phase, stop_reason, or other workflow keys
        assert result.get("phase") is None
        assert result.get("stop_reason") is None


# ---------------------------------------------------------------------------
# 5. Orchestrator routing
# ---------------------------------------------------------------------------

class TestDocumenterOrchestration:
    @pytest.fixture(autouse=True)
    def _reset_shutdown(self):
        """Ensure the global shutdown event is clear before and after each test."""
        from app.core.orchestrator import reset_shutdown
        reset_shutdown()
        yield
        reset_shutdown()

    def test_commit_routes_to_documenter_when_more_items(self):
        from app.core.orchestrator import _route_after_commit
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_commit(state) == "documenter"

    def test_commit_routes_to_complete_when_done(self):
        from app.core.orchestrator import _route_after_commit
        state = GraphState(phase=WorkflowPhase.COMPLETE)
        assert _route_after_commit(state) == "complete"

    def test_documenter_routes_to_coder_when_more_items(self):
        from app.core.orchestrator import _route_after_documenter
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_documenter(state) == "coder"

    def test_documenter_routes_to_complete_when_done(self):
        from app.core.orchestrator import _route_after_documenter
        state = GraphState(phase=WorkflowPhase.COMPLETE)
        assert _route_after_documenter(state) == "complete"

    def test_documenter_routes_to_stopped_on_shutdown(self):
        from app.core.orchestrator import _route_after_documenter, reset_shutdown, request_shutdown
        reset_shutdown()
        request_shutdown()
        try:
            state = GraphState(phase=WorkflowPhase.CODING)
            assert _route_after_documenter(state) == "stopped"
        finally:
            reset_shutdown()

    def test_graph_contains_documenter_node(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        assert "documenter" in graph.nodes

    def test_graph_compiles_with_documenter(self):
        from app.core.orchestrator import compile_graph
        compiled = compile_graph()
        assert compiled is not None

    def test_resume_routes_to_documenter_when_documenting(self):
        from app.core.orchestrator import _route_after_resume
        state = GraphState(phase=WorkflowPhase.DOCUMENTING)
        assert _route_after_resume(state) == "documenter"


# ---------------------------------------------------------------------------
# 6. Prompt file completeness
# ---------------------------------------------------------------------------

class TestDocumenterPrompt:
    @pytest.fixture
    def prompt_text(self) -> str:
        prompt_path = Path(__file__).parent.parent / "app" / "agents" / "prompts" / "documenter.txt"
        return prompt_path.read_text()

    def test_prompt_file_exists(self):
        prompt_path = Path(__file__).parent.parent / "app" / "agents" / "prompts" / "documenter.txt"
        assert prompt_path.exists()

    def test_prompt_mentions_changelog(self, prompt_text):
        assert "CHANGELOG" in prompt_text

    def test_prompt_mentions_keep_a_changelog(self, prompt_text):
        assert "Keep a Changelog" in prompt_text or "keepachangelog" in prompt_text.lower()

    def test_prompt_mentions_readme(self, prompt_text):
        assert "README" in prompt_text

    def test_prompt_mentions_unreleased(self, prompt_text):
        assert "Unreleased" in prompt_text or "unreleased" in prompt_text.lower()

    def test_prompt_forbids_invention(self, prompt_text):
        # Should explicitly warn against inventing things
        assert "invent" in prompt_text.lower() or "do not invent" in prompt_text.lower()

    def test_prompt_has_output_format(self, prompt_text):
        assert "CHANGED FILES" in prompt_text or "Output format" in prompt_text

    def test_prompt_longer_than_original(self, prompt_text):
        # Original was ~200 chars — new prompt should be substantially richer
        assert len(prompt_text) > 800
