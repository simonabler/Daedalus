"""Router node and routing-related helpers."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *

ROUTER_INTENTS = {"code", "status", "research", "resume"}

def _classify_request_intent(user_request: str) -> str:
    """Classify user input into planning intents."""
    text = (user_request or "").strip().lower()
    if not text:
        return "new_task"

    resume_markers = (
        "resume",
        "continue workflow",
        "continue task",
        "continue where",
        "fortsetzen",
        "weiterarbeiten",
        "weiter machen",
        "nach neustart",
        "wieder aufnehmen",
    )
    if any(marker in text for marker in resume_markers):
        return "resume_workflow"

    question_starters = (
        "was ",
        "wie ",
        "warum ",
        "wieso ",
        "welche ",
        "welcher ",
        "when ",
        "what ",
        "why ",
        "how ",
    )
    looks_like_question = text.endswith("?") or any(text.startswith(prefix) for prefix in question_starters)

    if looks_like_question and not is_programming_request(text):
        return "question_only"

    return "new_task"

def _heuristic_router_intent(user_request: str) -> str | None:
    """Fast intent heuristic for router gate."""
    text = (user_request or "").strip().lower()
    if not text:
        return "status"

    resume_markers = (
        "resume",
        "continue workflow",
        "continue task",
        "continue where",
        "fortsetzen",
        "weiterarbeiten",
        "weiter machen",
        "nach neustart",
        "wieder aufnehmen",
    )
    if any(marker in text for marker in resume_markers):
        return "resume"

    status_markers = (
        "status",
        "progress",
        "fortschritt",
        "current state",
        "aktueller stand",
        "wo stehen wir",
    )
    if any(marker in text for marker in status_markers):
        return "status"

    research_markers = (
        "research",
        "recherche",
        "investigate",
        "analyse",
        "analyze",
        "compare",
        "warum",
        "why",
    )
    if any(marker in text for marker in research_markers):
        return "research"

    if is_programming_request(text):
        return "code"

    return None

def _extract_repo_ref(user_request: str) -> str:
    """Extract a repository reference from free-form user text.

    Tries to detect:
    - Full HTTPS URL:        ``https://github.com/org/repo``
    - No-scheme forge URL:   ``github.com/org/repo``
    - owner/name or alias following keywords like "in", "for", "on"

    Returns the first match found, or ``""`` if nothing is detected.
    """
    raw = user_request.strip()

    # 1. Full HTTPS URL
    url_m = re.search(r"https?://[^\s/]+/[^\s/]+/[^\s]+", raw)
    if url_m:
        return url_m.group(0).rstrip(".,;:)")

    # 2. No-scheme forge URL  (github.com/..., gitlab.*/...)
    nscheme = re.search(
        r"\b(?:github\.com|gitlab\.[^\s/]+)/[^\s/]+/[^\s]+",
        raw,
        re.IGNORECASE,
    )
    if nscheme:
        return nscheme.group(0).rstrip(".,;:)")

    # 3. Keyword-anchored owner/name or alias
    kw = re.search(
        r"\b(?:in|for|on|repo|repository)\s+([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?)",
        raw,
        re.IGNORECASE,
    )
    if kw:
        return kw.group(1)

    return ""

def _parse_router_json(result: str) -> tuple[str | None, float]:
    """Parse strict JSON router output and validate intent."""
    try:
        parsed = json.loads((result or "").strip())
    except json.JSONDecodeError:
        return None, 0.0

    if not isinstance(parsed, dict):
        return None, 0.0

    intent = str(parsed.get("intent", "")).strip().lower()
    if intent not in ROUTER_INTENTS:
        return None, 0.0

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence

def _owner_to_agent(owner: str) -> str:
    owner_text = owner.lower()
    if "documenter" in owner_text:
        return "documenter"
    if "coder a" in owner_text:
        return "coder_a"
    if "coder b" in owner_text:
        return "coder_b"
    return ""

def _answer_question_directly(state: GraphState) -> str:
    """Answer non-task questions directly via planner without coder handoff."""
    prompt = (
        "You are the planner. The user asked a question, not a coding task.\n"
        "Answer directly and clearly. Do not create a plan. Do not modify files.\n\n"
        f"Question:\n{state.user_request}\n"
    )
    return _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)


# ---------------------------------------------------------------------------
# NODE: router
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> dict:
    """Intent gate before any planning/coding workflow starts."""
    emit_node_start("planner", "Router", item_desc=state.user_request[:100])
    emit_status("planner", "Classifying request intent", **_progress_meta(state, "planning"))

    # Extract repo reference from the request (used by context_loader + registry guard)
    repo_ref = state.repo_ref or _extract_repo_ref(state.user_request)

    # Detect issue reference — must happen before intent classification so the
    # issue URL/pattern influences routing (issue tasks are always "code" intent)
    issue_ref = state.issue_ref or parse_issue_ref(state.user_request, fallback_repo_ref=repo_ref)

    # If issue_ref provides a better repo_ref, prefer it
    if issue_ref and not repo_ref:
        repo_ref = issue_ref.repo_ref

    # Issue references are always coding tasks
    if issue_ref:
        emit_node_end("planner", "Router", f"Issue reference detected: #{issue_ref.issue_id} in {issue_ref.repo_ref}")
        return {"input_intent": "code", "repo_ref": repo_ref, "issue_ref": issue_ref}

    heuristic_intent = _heuristic_router_intent(state.user_request)
    if heuristic_intent in ROUTER_INTENTS:
        emit_node_end("planner", "Router", f"Heuristic intent: {heuristic_intent}")
        return {"input_intent": heuristic_intent, "repo_ref": repo_ref, "issue_ref": None}

    _router_prompt_file = Path(__file__).parent.parent / "agents" / "prompts" / "router.txt"
    if _router_prompt_file.exists():
        _router_system = _router_prompt_file.read_text(encoding="utf-8")
    else:
        _router_system = (
            "Classify the user request into ONE intent only.\n"
            "Allowed intents: code, status, research, resume.\n"
            "Return STRICT JSON only, no markdown:\n"
            '{"intent":"code|status|research|resume","confidence":0.0}\n'
        )
    llm_result = get_llm("planner").invoke(
        [SystemMessage(content=_router_system), HumanMessage(content=f"User request:\n{state.user_request}")]
    ).content
    llm_intent, confidence = _parse_router_json(llm_result)

    if llm_intent:
        emit_node_end("planner", "Router", f"LLM intent: {llm_intent} (confidence={confidence:.2f})")
        return {"input_intent": llm_intent, "repo_ref": repo_ref, "issue_ref": None}

    fallback = "code" if is_programming_request(state.user_request or "") else "research"
    emit_status(
        "planner",
        f"Router fallback intent: {fallback} (LLM output not parseable JSON)",
        **_progress_meta(state, "planning"),
    )
    emit_node_end("planner", "Router", f"Fallback intent: {fallback}")
    return {"input_intent": fallback, "repo_ref": repo_ref, "issue_ref": None}


# ---------------------------------------------------------------------------
# Issue hydration helper
# ---------------------------------------------------------------------------
