import urllib.error

import pytest

from src.tools.base import ToolError
from src.tools.web_research import web_research

KEY = "tvly-testkey-1234567890"
NOSLEEP = lambda s: None


def ok_post(url, payload, api_key, timeout):
    assert api_key == KEY and payload["query"] == "python release"
    return {"results": [
        {"title": "Python.org", "url": "https://www.python.org/", "content": "Latest  release " * 40},
        {"title": "Bad", "url": "javascript:alert(1)", "content": "x"},
        {"title": "Docs", "url": "https://docs.python.org/3/", "content": "Docs"}]}


def test_not_configured_never_invents_results():
    with pytest.raises(ToolError) as e:
        web_research("anything", "")
    assert "not configured" in str(e.value)


def test_success_returns_only_real_http_results():
    r = web_research("python release", KEY, http_post=ok_post, sleep=NOSLEEP)
    assert r["count"] == 2 and all(x["url"].startswith("https://") for x in r["results"])
    assert all(len(x["snippet"]) <= 300 for x in r["results"])


def test_empty_results():
    assert web_research("q", KEY, http_post=lambda *a: {"results": []}, sleep=NOSLEEP)["count"] == 0


def test_http_errors_mapped():
    def code(c):
        def f(*a):
            raise urllib.error.HTTPError("u", c, "x", {}, None)
        return f
    for c, text in [(401, "rejected"), (429, "rate limit"), (400, "HTTP 400")]:
        with pytest.raises(ToolError) as e:
            web_research("q", KEY, http_post=code(c), sleep=NOSLEEP)
        assert text in str(e.value)


def test_retries_then_succeeds_and_gives_up():
    calls = {"n": 0}

    def flaky(*a):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("down")
        return {"results": []}

    assert web_research("q", KEY, retries=2, http_post=flaky, sleep=NOSLEEP)["count"] == 0 and calls["n"] == 3
    calls["n"] = 0

    def dead(*a):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    with pytest.raises(ToolError) as e:
        web_research("q", KEY, retries=2, http_post=dead, sleep=NOSLEEP)
    assert "could not reach" in str(e.value) and calls["n"] == 3


def test_malformed_response_and_validation():
    with pytest.raises(ToolError):
        web_research("q", KEY, http_post=lambda *a: {"nope": 1}, sleep=NOSLEEP)
    for q in ["", "  ", "x" * 401, None]:
        with pytest.raises(ToolError):
            web_research(q, KEY, http_post=ok_post, sleep=NOSLEEP)
    with pytest.raises(ToolError):
        web_research("q", KEY, max_results=0, http_post=ok_post, sleep=NOSLEEP)
