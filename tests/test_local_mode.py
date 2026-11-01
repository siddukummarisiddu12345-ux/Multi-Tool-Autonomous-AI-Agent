from src.config import Settings
from src.local_mode import plan, run_local
from src.registry import build_registry

REG = build_registry(Settings())


def tools(task):
    return [c[0] for c in plan(task)]


def test_routing():
    assert tools("Calculate (12.5 + 3) * 4") == ["calculator"]
    assert tools("Convert 5 km to miles") == ["unit_converter"]
    assert tools("100 F in celsius") == ["unit_converter"]
    assert tools("Days between 2025-01-01 and 2025-03-15") == ["datetime_tool"]
    assert tools("What time is it?") == ["datetime_tool"]
    assert tools('Validate this JSON: {"a": 1}') == ["json_validator"]
    assert tools('Analyze this text: "agents use tools and tools help agents"') == ["text_analyzer"]
    assert tools("search for python release") == ["web_research"]
    assert tools("hello there") == []


def test_dates_are_not_treated_as_arithmetic():
    assert tools("Days between 2025-01-01 and 2025-03-15") == ["datetime_tool"]


def test_multi_step_local():
    r = run_local("Calculate 6 * 7 and convert 10 km to miles", REG)
    assert r.status == "success" and [s.name for s in r.steps if s.kind == "tool"] == ["unit_converter", "calculator"] \
        or sorted(s.name for s in r.steps if s.kind == "tool") == ["calculator", "unit_converter"]
    assert "42" in r.answer and "6.2137" in r.answer


def test_results_and_labels():
    r = run_local("Convert 5 km to miles", REG)
    assert r.mode == "local" and r.steps[0].kind == "router" and "3.1068" in r.answer


def test_no_match_is_honest():
    r = run_local("tell me a joke", REG)
    assert r.status == "no_match" and "not an AI model" in r.answer


def test_tool_failure_and_missing_search_key():
    assert run_local("Calculate 1 / 0", REG).status == "error"
    r = run_local("search the web for cats", REG)
    assert r.status == "error" and "not configured" in r.answer


def test_invalid_input():
    assert run_local("   ", REG).status == "error"
    assert run_local("x" * 5000, REG).status == "error"


def test_invalid_json_reported():
    r = run_local('validate this JSON: {"a": }', REG)
    assert r.status == "success" and "Invalid JSON" in r.answer
