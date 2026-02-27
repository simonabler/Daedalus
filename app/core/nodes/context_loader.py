"""Context loader node and issue hydration helpers."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *
from ._intelligence_helpers import *
from ._prompt_enrichment import *

def _hydrate_issue(state: GraphState) -> tuple[str, dict]:
    """Fetch issue content from forge and return (enriched_request, extra_state).

    Called by context_loader_node when state.issue_ref is set.

    Returns:
        (enriched_user_request, extra_dict)
        extra_dict is merged into context_loader's return value.
    """
    issue_ref = state.issue_ref
    if not issue_ref:
        return state.user_request, {}

    emit_status(
        "planner",
        f"📋 Fetching issue #{issue_ref.issue_id} from {issue_ref.repo_ref}…",
        **_progress_meta(state, "planning"),
    )

    try:
        from infra.factory import get_forge_client
        from infra.forge import ForgeError

        # Build a URL for platform detection: use the repo_ref as-is if it
        # already looks like a URL, otherwise construct one from the repo_ref.
        ref = issue_ref.repo_ref
        if not ref.startswith("http"):
            # ref might be "github.com/owner/repo" or "owner/repo"
            if "/" in ref and not ref.startswith("http"):
                parts = ref.split("/")
                if len(parts) >= 3:
                    # has host prefix
                    url_for_detection = f"https://{ref}"
                else:
                    # short owner/repo — assume github
                    url_for_detection = f"https://github.com/{ref}"
            else:
                url_for_detection = f"https://github.com/{ref}"
        else:
            url_for_detection = ref

        client = get_forge_client(
            url_for_detection,
            platform=issue_ref.platform or None,
        )

        # The repo path for the client API is everything after the host
        if ref.startswith("http"):
            from urllib.parse import urlparse
            parsed = urlparse(ref)
            repo_path = parsed.path.strip("/")
        else:
            parts = ref.split("/")
            if len(parts) >= 3:
                repo_path = "/".join(parts[1:])   # strip host
            else:
                repo_path = ref

        issue = client.get_issue(repo_path, issue_ref.issue_id)

        # Build enriched task description
        labels_str = ", ".join(issue.labels) if issue.labels else "none"
        created_str = issue.created_at.strftime("%Y-%m-%d") if issue.created_at else "unknown"
        enriched = (
            f"Issue #{issue_ref.issue_id}: {issue.title}\n\n"
            f"{issue.description or '(no description)'}\n\n"
            f"Labels: {labels_str}\n"
            f"Reporter: {issue.author or 'unknown'}\n"
            f"Created: {created_str}\n"
            f"URL: {issue.url}"
        )

        emit_status(
            "planner",
            f"✅ Issue #{issue_ref.issue_id} loaded: \"{issue.title}\"",
            **_progress_meta(state, "planning"),
        )

        # Emit a dedicated event so UI and Telegram can show the issue card
        emit(WorkflowEvent(
            category=EventCategory.STATUS,
            agent="planner",
            title="issue_loaded",
            metadata={
                "issue_number": issue_ref.issue_id,
                "issue_title": issue.title,
                "repo_ref": issue_ref.repo_ref,
                "issue_url": issue.url,
                "platform": issue_ref.platform,
            },
        ))

        # Post "working on it" comment — best-effort, never blocks workflow
        try:
            client.post_comment(
                repo_path,
                issue_ref.issue_id,
                "🤖 **Daedalus** is working on this issue.",
            )
            logger.info("Posted working-on-it comment on issue #%d", issue_ref.issue_id)
        except Exception as comment_exc:
            logger.warning(
                "Could not post comment on issue #%d: %s",
                issue_ref.issue_id, comment_exc,
            )

        return enriched, {}

    except Exception as exc:
        logger.warning(
            "Issue hydration failed for #%d in %s: %s — using original request",
            issue_ref.issue_id, issue_ref.repo_ref, exc,
        )
        emit_status(
            "planner",
            f"⚠️ Could not fetch issue #{issue_ref.issue_id}: {exc} — continuing with original request",
            **_progress_meta(state, "planning"),
        )
        return state.user_request, {}


# ---------------------------------------------------------------------------
# NODE: context_loader (placeholder for Patch 02)
# ---------------------------------------------------------------------------

def context_loader_node(state: GraphState) -> dict:
    """Load repository context before planner execution."""
    emit_node_start("planner", "Context Loader", item_desc=state.user_request[:100])
    if state.context_loaded:
        emit_status("planner", "Context already loaded; skipping re-analysis", **_progress_meta(state, "planning"))
        emit_node_end("planner", "Context Loader", "Skipped (already loaded)")
        return {"input_intent": "code", "context_loaded": True}

    settings = get_settings()
    repo_root = (state.repo_root or settings.target_repo_path or "").strip()

    # ── Dynamic workspace: clone or pull if no static path is configured ──
    if not repo_root:
        repo_ref = (state.repo_ref or "").strip()
        if not repo_ref:
            emit_error("planner", "Context loader could not determine repository path.")
            emit_node_end("planner", "Context Loader", "Failed (repo path missing)")
            return {
                "repo_facts": {"error": "Missing repository path", "fallback": True},
                "context_loaded": False,
                "stop_reason": "context_repo_path_missing",
                "phase": WorkflowPhase.STOPPED,
            }
        # ── Registry guard — reject unknown repos before cloning ────────
        try:
            from infra.registry import get_registry
            _registry = get_registry()
            if len(_registry) > 0 and not _registry.is_allowed(repo_ref):
                emit_error(
                    "planner",
                    f"🚫 Repository {repo_ref!r} is not in repos.yaml. "
                    f"Known repos: {[e.name for e in _registry.list_repos()]}",
                )
                emit_node_end("planner", "Context Loader", "Failed (repo not in registry)")
                return {
                    "repo_facts": {"error": f"Repo not in registry: {repo_ref}", "fallback": True},
                    "context_loaded": False,
                    "stop_reason": "context_repo_not_in_registry",
                    "phase": WorkflowPhase.STOPPED,
                }
        except Exception as _reg_exc:
            logger.warning("Registry check skipped: %s", _reg_exc)

        try:
            from infra.workspace import WorkspaceManager
            workspace_dir = Path(settings.daedalus_workspace_dir).expanduser().resolve()
            workspace = WorkspaceManager(workspace_dir)
            emit_status(
                "planner",
                f"📥 Resolving workspace for {repo_ref!r} → {workspace_dir}",
                **_progress_meta(state, "planning"),
            )
            resolved_path = workspace.resolve(repo_ref)
            repo_root = str(resolved_path)
            emit_status(
                "planner",
                f"✅ Workspace ready: {repo_root}",
                **_progress_meta(state, "planning"),
            )
        except Exception as exc:
            emit_error("planner", f"Workspace resolver failed for {repo_ref!r}: {exc}")
            emit_node_end("planner", "Context Loader", "Failed (workspace error)")
            return {
                "repo_facts": {"error": f"Workspace error: {exc}", "fallback": True},
                "context_loaded": False,
                "stop_reason": "context_workspace_error",
                "phase": WorkflowPhase.STOPPED,
            }

    repo_path = Path(repo_root).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        emit_error("planner", f"Context loader repository path invalid: {repo_path}")
        emit_node_end("planner", "Context Loader", "Failed (repo path invalid)")
        return {
            "repo_facts": {"error": f"Invalid repository path: {repo_path}", "fallback": True},
            "context_loaded": False,
            "stop_reason": "context_repo_path_invalid",
            "phase": WorkflowPhase.STOPPED,
        }

    # Propagate resolved root into context var so all tools use it
    from app.core.active_repo import set_repo_root as _set_repo_root
    _set_repo_root(str(repo_path))

    # ── Issue hydration: fetch issue content and enrich user_request ──────
    hydrated_request = state.user_request
    hydration_extra: dict = {}
    if state.issue_ref:
        hydrated_request, hydration_extra = _hydrate_issue(state)

    emit_status("planner", "Reading repository documentation and structure", **_progress_meta(state, "planning"))

    # Daedalus' own root directory — determined once at import time.
    # AGENT.md files are ONLY read from the target repo, never from Daedalus itself.
    # This prevents Daedalus' own build-spec (AGENT.md) from leaking into tasks
    # targeting unrelated repositories.
    _daedalus_root = Path(__file__).parent.parent.parent.resolve()
    _is_self_referential = repo_path.resolve() == _daedalus_root

    doc_files = [
        "docs/AGENT.md",
        "AGENT.md",
        "AGENTS.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "CONTRIBUTING.rst",
        "README.md",
    ]

    # AGENT.md files are intentionally excluded when the target repo IS Daedalus itself.
    # When working on Daedalus, TARGET_REPO_PATH must point to a separate clone/copy —
    # in that case the copy's own AGENT.md will be read normally.
    _agent_md_files = {"docs/AGENT.md", "AGENT.md", "AGENTS.md"}

    max_chars = max(1000, int(settings.max_output_chars))
    instruction_chunks: list[str] = []
    for rel_path in doc_files:
        if _is_self_referential and rel_path in _agent_md_files:
            logger.info(
                "Context loader: skipping %s — target repo is Daedalus root. "
                "Set TARGET_REPO_PATH to a separate clone to enable self-improvement mode.",
                rel_path,
            )
            continue
        file_path = repo_path / rel_path
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            file_content = file_path.read_text(encoding="utf-8", errors="replace")
            trimmed = _truncate_context_text(file_content, limit=min(max_chars, 8000))
            instruction_chunks.append(f"=== {rel_path} ===\n{trimmed}")
        except Exception as exc:
            logger.warning("Could not read context file %s: %s", rel_path, exc)

    agent_instructions = "\n\n".join(instruction_chunks).strip()
    context_listing = _build_context_listing(repo_path, max_depth=2, max_entries=350)

    repo_facts: dict = {}
    try:
        from app.agents.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(repo_path)
        repo_context = analyzer.analyze_repository()
        repo_facts = repo_context.to_dict()
    except Exception as exc:
        logger.warning("Structured repository analysis failed; falling back to heuristics: %s", exc)
        repo_facts = {"error": str(exc), "fallback": True}

    if repo_facts.get("fallback"):
        repo_facts = _heuristic_analysis(repo_path)

    summary = _format_context_summary(repo_facts)

    # -- Static analysis (best-effort, never blocks the workflow) ----------
    static_issues: list[dict] = []
    try:
        from app.tools.static_analysis import run_static_analysis

        language = _extract_language(repo_facts)
        if language:
            emit_status("planner", f"Running static analysis ({language})…",
                        **_progress_meta(state, "planning"))
            raw_issues = run_static_analysis(repo_path, language)
            static_issues = [i.model_dump() for i in raw_issues]
            err_count = sum(1 for i in raw_issues if i.severity == "error")
            warn_count = sum(1 for i in raw_issues if i.severity == "warning")
            emit_status(
                "planner",
                f"Static analysis complete: {err_count} error(s), {warn_count} warning(s)",
                **_progress_meta(state, "planning"),
            )
    except Exception as exc:
        logger.warning("Static analysis failed (skipping): %s", exc)

    # -- Call graph analysis (best-effort, never blocks the workflow) -------
    call_graph: dict = {}
    try:
        from app.analysis.call_graph import CallGraphAnalyzer, format_call_graph_for_prompt

        emit_status("planner", "Building call graph…", **_progress_meta(state, "planning"))
        cg_analyzer = CallGraphAnalyzer(repo_path)
        cg = cg_analyzer.analyze()
        call_graph = cg.to_dict()
        emit_status(
            "planner",
            f"Call graph ready: {len(cg.all_functions())} functions, "
            f"{sum(len(v) for v in cg.callees.values())} edges",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Call graph analysis failed (skipping): %s", exc)

    # -- Dependency graph analysis (best-effort, never blocks the workflow) --
    dependency_graph: dict = {}
    dep_cycles: list = []
    try:
        from app.analysis.dependency_graph import DependencyAnalyzer, format_dep_graph_for_prompt

        emit_status("planner", "Building dependency graph…", **_progress_meta(state, "planning"))
        dep_analyzer = DependencyAnalyzer(repo_path)
        dg = dep_analyzer.analyze()
        dependency_graph = dg.to_dict()
        dep_cycles = dg.cycles
        cycle_msg = f", {len(dg.cycles)} cycle(s) detected" if dg.cycles else ""
        emit_status(
            "planner",
            f"Dependency graph ready: {len(dg.all_modules())} modules{cycle_msg}",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Dependency graph analysis failed (skipping): %s", exc)

    # -- Code smell detection (best-effort, never blocks the workflow) --------
    code_smells: list[dict] = []
    try:
        from app.analysis.smell_detector import SmellDetector, format_smells_for_prompt

        emit_status("planner", "Detecting code smells…", **_progress_meta(state, "planning"))
        detector = SmellDetector(repo_path, call_graph=call_graph)
        raw_smells = detector.detect()
        code_smells = [s.model_dump() for s in raw_smells]
        err_count  = sum(1 for s in raw_smells if s.severity == "error")
        warn_count = sum(1 for s in raw_smells if s.severity == "warning")
        emit_status(
            "planner",
            f"Code smell detection complete: {err_count} error(s), {warn_count} warning(s), {len(raw_smells)} total",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Code smell detection failed (skipping): %s", exc)

    emit_status(
        "planner",
        f"Repository context loaded: {summary}",
        **_progress_meta(state, "planning"),
    )
    emit_node_end("planner", "Context Loader", "Repository context ready for planner")

    return {
        "input_intent": "code",
        "agent_instructions": agent_instructions,
        "repo_facts": repo_facts,
        "context_listing": _truncate_context_text(context_listing, limit=min(max_chars, 10000)),
        "context_loaded": True,
        "repo_root": str(repo_path),  # persist resolved path (covers workspace case)
        "user_request": hydrated_request,  # enriched with issue content if issue_ref set
        **hydration_extra,               # e.g. issue_comment_posted
        "static_issues": static_issues,
        "call_graph": call_graph,
        "dependency_graph": dependency_graph,
        "dep_cycles": dep_cycles,
        "code_smells": code_smells,
    }
