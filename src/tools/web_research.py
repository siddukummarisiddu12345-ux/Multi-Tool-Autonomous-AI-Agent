"""Web research via the Tavily API. Real results only; never fabricated."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from src.tools.base import ToolError

TAVILY_URL = "https://api.tavily.com/search"
MAX_QUERY, MAX_BYTES, SNIPPET = 400, 1_000_000, 300


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # fixed https endpoint
        raw = resp.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("response too large")
    return json.loads(raw.decode())


def web_research(query, api_key: str, max_results: int = 5, timeout: float = 30.0,
                 retries: int = 2, http_post: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    if not api_key:
        raise ToolError("Web research is not configured. Set TAVILY_API_KEY to enable it; "
                        "no results can be provided without it.")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("Search query is empty.")
    query = query.strip()
    if len(query) > MAX_QUERY:
        raise ToolError(f"Search query too long (limit {MAX_QUERY} characters).")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 10:
        raise ToolError("max_results must be an integer between 1 and 10.")
    post = http_post or _post
    payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
    data, attempt = None, 0
    while True:
        try:
            data = post(TAVILY_URL, payload, api_key, timeout)
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ToolError("Web research failed: the Tavily API key was rejected.") from None
            if exc.code == 429:
                raise ToolError("Web research failed: Tavily rate limit or quota exceeded.") from None
            if exc.code < 500 or attempt >= retries:
                raise ToolError(f"Web research failed: Tavily returned HTTP {exc.code}.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= retries:
                raise ToolError("Web research failed: could not reach the search service.") from None
        except (ValueError, json.JSONDecodeError):
            raise ToolError("Web research failed: unreadable response from the search service.") from None
        attempt += 1
        sleep(min(2.0, 0.5 * attempt))
    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ToolError("Web research failed: unexpected response format.")
    results = []
    for it in items[:max_results]:
        if not isinstance(it, dict) or not str(it.get("url", "")).lower().startswith(("http://", "https://")):
            continue
        snippet = " ".join(str(it.get("content") or "").split())
        results.append({"title": str(it.get("title") or it["url"]).strip()[:120], "url": it["url"],
                        "snippet": snippet[:SNIPPET]})
    return {"query": query, "count": len(results), "results": results}
