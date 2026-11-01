import json

from src.agent import Agent, GeminiClient, LLMError, LLMReply, map_llm_error
from src.config import Settings
from src.registry import build_registry

KEY = "AIzaSyFAKEKEY1234567890abcdefgh"
SETTINGS = Settings(google_api_key=KEY)


def tc(name, args, cid="c1"):
    return LLMReply(tool_calls=[{"id": cid, "name": name, "args": args}])


class FakeLLM:
    """Scripted stand-in for the LLM; items are LLMReply objects or exceptions."""

    def __init__(self, script, default=None):
        self.script, self.calls, self.default = list(script), [], default

    def chat(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t.name for t in tools]})
        item = self.script.pop(0) if self.script else (self.default() if self.default else LLMReply(text="done"))
        if isinstance(item, Exception):
            raise item
        return item


def make(script, **kw):
    llm = FakeLLM(script, kw.pop("default", None))
    return Agent(build_registry(SETTINGS), llm, SETTINGS, **kw), llm


def test_direct_answer_without_tools():
    agent, llm = make([LLMReply(text="Hi!")])
    r = agent.run("hello")
    assert r.status == "success" and r.answer == "Hi!" and [s.kind for s in r.steps] == ["llm"]
    assert len(llm.calls[0]["tools"]) == 6  # all six tools are offered to the model


def test_model_selects_tool_and_result_returns_to_model():
    agent, llm = make([tc("unit_converter", {"value": 72, "from_unit": "F", "to_unit": "C"}),
                       LLMReply(text="72 F is about 22.2 C.")])
    r = agent.run("convert 72F to C")
    assert r.status == "success" and [s.name for s in r.steps if s.kind == "tool"] == ["unit_converter"]
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "22.22" in tool_msg["content"]
    assert llm.calls[1]["messages"][-2]["tool_calls"][0]["name"] == "unit_converter"


def test_multi_step_workflow_with_different_tools():
    agent, llm = make([
        tc("calculator", {"expression": "1250 * 1.18"}, "a"),
        tc("unit_converter", {"value": 1475, "from_unit": "km", "to_unit": "mi"}, "b"),
        tc("datetime_tool", {"operation": "diff", "start": "2025-01-01", "end": "2025-12-25"}, "c"),
        LLMReply(text="Done.")])
    r = agent.run("multi step task")
    assert [s.name for s in r.steps if s.kind == "tool"] == ["calculator", "unit_converter", "datetime_tool"]
    assert all(s.status == "success" for s in r.steps) and r.status == "success"
    assert [s.index for s in r.steps] == list(range(1, len(r.steps) + 1))
    assert len(llm.calls) == 4


def test_parallel_tool_calls_in_one_reply():
    reply = LLMReply(tool_calls=[{"id": "1", "name": "calculator", "args": {"expression": "1+1"}},
                                 {"id": "2", "name": "text_analyzer", "args": {"text": "a b c"}}])
    agent, llm = make([reply, LLMReply(text="ok")])
    r = agent.run("two things")
    assert [s.name for s in r.steps if s.kind == "tool"] == ["calculator", "text_analyzer"]
    assert [m["role"] for m in llm.calls[1]["messages"][-3:]] == ["assistant", "tool", "tool"]


def test_tool_selection_is_driven_by_the_llm_not_fixed():
    def picker(messages, wanted):
        return tc(wanted[0], wanted[1]) if messages[-1]["role"] == "user" else LLMReply(text="ok")

    for name, args in [("calculator", {"expression": "2+2"}), ("text_analyzer", {"text": "hello world"}),
                       ("json_validator", {"json_text": "[1]"})]:
        agent, llm = make([tc(name, args), LLMReply(text="ok")])
        assert [s.name for s in agent.run("x").steps if s.kind == "tool"] == [name]


def test_tool_error_is_fed_back_and_loop_continues():
    agent, llm = make([tc("calculator", {"expression": "1/0"}), LLMReply(text="Can't divide by zero.")])
    r = agent.run("1/0")
    assert r.status == "success" and r.steps[1].status == "error"
    assert llm.calls[1]["messages"][-1]["content"].startswith("ERROR:")


def test_unknown_tool_and_bad_arguments():
    agent, _ = make([tc("shell", {"cmd": "ls"}, "a"), tc("calculator", "bad", "b"),
                     tc("calculator", {"expression": ""}, "c"), LLMReply(text="ok")])
    r = agent.run("be bad")
    tools = [s for s in r.steps if s.kind == "tool"]
    assert [s.status for s in tools] == ["error"] * 3 and "Unknown tool" in tools[0].summary


def test_duplicate_calls_blocked():
    same = {"expression": "1+1"}
    agent, llm = make([tc("calculator", same, "a"), tc("calculator", same, "b"), LLMReply(text="2")])
    r = agent.run("1+1")
    assert [s.status for s in r.steps if s.kind == "tool"] == ["success", "skipped"]


def test_max_steps_limit():
    agent, llm = make([], max_steps=3, default=lambda: tc("calculator", {"expression": f"{len(llm.calls)}+1"}, str(len(llm.calls))))
    r = agent.run("loop forever")
    assert len([s for s in r.steps if s.kind == "tool"]) == 3
    assert llm.calls[-1]["tools"] == [] and len(llm.calls) <= 5
    assert r.status == "max_steps"


def test_limit_then_model_answers():
    agent, llm = make([tc("calculator", {"expression": "1+1"}, "a"), LLMReply(text="Partial answer.")], max_steps=1)
    r = agent.run("x")
    assert r.status == "max_steps" and r.answer == "Partial answer."
    assert llm.calls[1]["tools"] == [] and "limit" in llm.calls[1]["messages"][0]["content"]


def test_missing_api_key_gives_clear_message():
    r = Agent(build_registry(Settings()), None, Settings()).run("hello")
    assert r.status == "not_configured" and "GOOGLE_API_KEY" in r.answer
    try:
        GeminiClient(Settings())
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert exc.kind == "not_configured"


def test_api_failures_are_graceful_and_redacted():
    for exc, status in [(LLMError(f"rate limited {KEY}", "rate_limit"), "error"),
                        (RuntimeError(f"boom {KEY}"), "error")]:
        agent, _ = make([exc])
        r = agent.run("hello")
        assert r.status == status and KEY not in json.dumps(r.model_dump())
        assert r.steps[-1].status == "error"


def test_error_mapping():
    class E(Exception):
        def __init__(self, code=None, msg=""):
            super().__init__(msg)
            self.code = code
    assert map_llm_error(E(429))[1] == "rate_limit"
    assert map_llm_error(E(None, "RESOURCE_EXHAUSTED quota"))[1] == "rate_limit"
    assert map_llm_error(E(403))[1] == "auth" and map_llm_error(E(503))[1] == "unavailable"
    assert map_llm_error(E(None, "Deadline exceeded"))[1] == "timeout"
    msg, kind = map_llm_error(E(None, f"weird {KEY}"))
    assert KEY not in msg and kind == "error"


def test_malformed_model_replies():
    agent, _ = make([LLMReply(text="", tool_calls=[])])
    assert agent.run("hi").status == "error"
    agent, _ = make([tc(None, None), LLMReply(text="recovered")])
    assert agent.run("hi").answer == "recovered"


def test_input_validation_and_history():
    agent, llm = make([LLMReply(text="ok")])
    assert agent.run("  ").status == "error" and agent.run("x" * 5000).status == "error" and llm.calls == []
    hist = [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"}, {"role": "system", "content": "ignored"}]
    agent.run("now", hist)
    sent = llm.calls[0]["messages"]
    assert sent[0]["role"] == "system" and sent[1]["content"] == "earlier" and sent[-1]["content"] == "now"
    assert all(m["role"] in ("system", "user", "assistant") for m in sent)


def test_secrets_never_in_result():
    agent, _ = make([tc("calculator", {"expression": "1+1"}), LLMReply(text=f"key {KEY}")])
    assert KEY not in json.dumps(agent.run("x").model_dump())
