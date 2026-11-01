from src.config import Settings
from src.registry import build_registry
from src.security import redact, safe_error, truncate

REG = build_registry(Settings())


def test_all_six_tools_registered_with_schemas():
    assert REG.names() == ["calculator", "text_analyzer", "json_validator", "datetime_tool",
                           "unit_converter", "web_research"]
    assert all(s.description and s.input_model for s in REG.specs())


def test_each_tool_runs_through_registry():
    cases = [("calculator", {"expression": "2^8"}), ("text_analyzer", {"text": "one two three"}),
             ("json_validator", {"json_text": "{\"a\": 1}"}),
             ("datetime_tool", {"operation": "diff", "start": "2025-01-01", "end": "2025-01-31"}),
             ("unit_converter", {"value": 10, "from_unit": "km", "to_unit": "mi"})]
    for name, args in cases:
        out = REG.execute(name, args)
        assert out.ok and out.text and out.data, name


def test_validation_errors_are_readable():
    assert "Unknown tool" in REG.execute("shell", {}).error
    assert "Unknown tool" in REG.execute(None, {}).error
    assert "JSON object" in REG.execute("calculator", "1+1").error
    for args in [{}, {"expression": 5}, {"expression": "1", "x": 1}, {"expression": ""}, {"expression": "1" * 300}]:
        out = REG.execute("calculator", args)
        assert not out.ok and out.error.startswith("Invalid arguments"), args
    assert not REG.execute("unit_converter", {"value": "abc", "from_unit": "m", "to_unit": "km"}).ok
    assert not REG.execute("datetime_tool", {"operation": "later"}).ok


def test_tool_errors_are_contained():
    assert "zero" in REG.execute("calculator", {"expression": "1/0"}).error
    assert not REG.execute("web_research", {"query": "x"}).ok  # not configured
    assert "not configured" in REG.execute("web_research", {"query": "x"}).error


def test_status():
    states = {s.name: s.status()[0] for s in REG.specs()}
    assert states["web_research"] == "setup" and states["calculator"] == "ready"
    cfg = build_registry(Settings(tavily_api_key="tvly-abcdefghijk"))
    assert {s.name: s.status()[0] for s in cfg.specs()}["web_research"] == "ready"


def test_security_helpers():
    assert redact("k=AIzaSyABCDEFGHIJKLMNOPQRSTUV") == "k=[REDACTED]"
    assert redact("abc123456 here", ["abc123456"]) == "[REDACTED] here"
    assert "supersecret" not in safe_error(RuntimeError("bad supersecret"), ["supersecret"])
    assert truncate("abcdef", 4) == "abc…"
