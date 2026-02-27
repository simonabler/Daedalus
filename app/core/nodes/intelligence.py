"""Code intelligence node."""

from __future__ import annotations

from ._helpers import *
from ._intelligence_helpers import *

def code_intelligence_node(state: GraphState) -> dict:
    """Run all four analysis tools and cache results by git commit hash.

    Flow:
      1. Determine repo path.
      2. Get current git commit hash → cache key.
      3. If cache hit → restore all analysis fields instantly.
      4. If cache miss → run StaticAnalysis, CallGraph, DependencyGraph,
         SmellDetector sequentially; save to cache.
      5. Return updated state fields + emit intelligence_complete event.
    """
    from app.analysis.intelligence_cache import get_commit_hash, load_cache, save_cache

    settings = get_settings()
    repo_root = (state.repo_root or settings.target_repo_path or "").strip()
    if not repo_root:
        logger.warning("code_intelligence_node: no repo path, skipping")
        return {}

    repo_path = Path(repo_root).resolve()
    if not repo_path.is_dir():
        logger.warning("code_intelligence_node: invalid repo path %s, skipping", repo_path)
        return {}

    emit_node_start("planner", "Code Intelligence", item_desc="Analysing repository…")
    emit_status("planner", "Starting code intelligence analysis…", **_progress_meta(state, "analyzing"))

    # ── Cache lookup ────────────────────────────────────────────────────────
    cache_key = get_commit_hash(repo_path) or ""
    if cache_key:
        cached = load_cache(repo_path, cache_key)
        if cached:
            emit_status(
                "planner",
                f"Code intelligence loaded from cache (commit {cache_key})",
                **_progress_meta(state, "analyzing"),
            )
            _emit_intelligence_complete(cached)
            emit_node_end("planner", "Code Intelligence", "Cache hit — analysis restored")
            return {**cached, "intelligence_cache_key": cache_key, "intelligence_cached": True}

    # ── Run all four tools ──────────────────────────────────────────────────
    language = _extract_language(state.repo_facts)

    static_issues: list[dict] = []
    try:
        from app.tools.static_analysis import run_static_analysis
        emit_status("planner", "Running static analysis…", **_progress_meta(state, "analyzing"))
        raw = run_static_analysis(repo_path, language)
        static_issues = [i.model_dump() for i in raw]
    except Exception as exc:
        logger.warning("intelligence: static analysis failed: %s", exc)

    call_graph: dict = {}
    try:
        from app.analysis.call_graph import CallGraphAnalyzer
        emit_status("planner", "Building call graph…", **_progress_meta(state, "analyzing"))
        cg = CallGraphAnalyzer(repo_path).analyze()
        call_graph = cg.to_dict()
    except Exception as exc:
        logger.warning("intelligence: call graph failed: %s", exc)

    dependency_graph: dict = {}
    dep_cycles: list = []
    try:
        from app.analysis.dependency_graph import DependencyAnalyzer
        emit_status("planner", "Building dependency graph…", **_progress_meta(state, "analyzing"))
        dg = DependencyAnalyzer(repo_path).analyze()
        dependency_graph = dg.to_dict()
        dep_cycles = dg.cycles
    except Exception as exc:
        logger.warning("intelligence: dependency graph failed: %s", exc)

    code_smells: list[dict] = []
    try:
        from app.analysis.smell_detector import SmellDetector
        emit_status("planner", "Detecting code smells…", **_progress_meta(state, "analyzing"))
        smells = SmellDetector(repo_path, call_graph=call_graph).detect()
        code_smells = [s.model_dump() for s in smells]
    except Exception as exc:
        logger.warning("intelligence: smell detection failed: %s", exc)

    result = {
        "static_issues": static_issues,
        "call_graph": call_graph,
        "dependency_graph": dependency_graph,
        "dep_cycles": dep_cycles,
        "code_smells": code_smells,
    }

    # ── Save to cache ───────────────────────────────────────────────────────
    if cache_key:
        save_cache(repo_path, cache_key, result)

    # ── Summary emit ────────────────────────────────────────────────────────
    n_errors  = sum(1 for s in static_issues if s.get("severity") == "error")
    n_smells  = len(code_smells)
    n_cycles  = len(dep_cycles)
    summary   = f"{n_errors} static error(s) · {n_smells} smell(s) · {n_cycles} cycle(s)"

    emit_status("planner", f"Code intelligence complete: {summary}", **_progress_meta(state, "analyzing"))
    _emit_intelligence_complete(result)
    emit_node_end("planner", "Code Intelligence", summary)

    return {
        **result,
        "intelligence_cache_key": cache_key,
        "intelligence_cached": False,
        "phase": WorkflowPhase.ANALYZING,
    }

def _emit_intelligence_complete(data: dict) -> None:
    """Broadcast intelligence_complete event with summary counts via WorkflowEvent."""
    try:
        from app.core.events import emit, WorkflowEvent, EventCategory
        n_static_err  = sum(1 for s in data.get("static_issues", []) if s.get("severity") == "error")
        n_static_warn = sum(1 for s in data.get("static_issues", []) if s.get("severity") == "warning")
        n_smells      = len(data.get("code_smells", []))
        n_smell_err   = sum(1 for s in data.get("code_smells", []) if s.get("severity") == "error")
        n_cycles      = len(data.get("dep_cycles", []))
        n_funcs       = len(data.get("call_graph", {}).get("callees", {}))
        n_modules     = len(data.get("dependency_graph", {}).get("imports", {}))
        emit(WorkflowEvent(
            category=EventCategory.STATUS,
            agent="planner",
            title="intelligence_complete",
            detail=(
                f"{n_static_err} static error(s), {n_static_warn} warning(s) | "
                f"{n_smells} smell(s) ({n_smell_err} error) | "
                f"{n_cycles} cycle(s) | {n_funcs} functions | {n_modules} modules"
            ),
            metadata={
                "type": "intelligence_complete",
                "static_errors": n_static_err,
                "static_warnings": n_static_warn,
                "smells_total": n_smells,
                "smells_errors": n_smell_err,
                "cycles": n_cycles,
                "functions": n_funcs,
                "modules": n_modules,
            },
        ))
    except Exception as exc:
        logger.debug("_emit_intelligence_complete: %s", exc)
