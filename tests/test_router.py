"""Tests for the intent router gate."""

from app.core.state import GraphState


def test_parse_router_json_valid():
    from app.core.nodes import _parse_router_json

    intent, confidence = _parse_router_json('{"intent":"research","confidence":0.91}')
    assert intent == "research"
    assert confidence == 0.91


def test_parse_router_json_invalid_returns_none():
    from app.core.nodes import _parse_router_json

    intent, confidence = _parse_router_json("not-json")
    assert intent is None
    assert confidence == 0.0


def test_router_node_uses_heuristic_for_code():
    from app.core.nodes import router_node

    state = GraphState(user_request="Implement a new API endpoint for auth")
    result = router_node(state)

    assert result["input_intent"] == "code"


def test_router_node_detects_resume():
    from app.core.nodes import router_node

    state = GraphState(user_request="Please resume workflow from previous run")
    result = router_node(state)

    assert result["input_intent"] == "resume"
