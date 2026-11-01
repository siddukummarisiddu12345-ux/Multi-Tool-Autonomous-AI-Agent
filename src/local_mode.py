"""Deterministic local mode: a transparent RULE-BASED router (regex/keywords), not an LLM.

It works without any API key and is labelled as such in the UI and the results.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from src.registry import ToolRegistry
from src.schemas import AgentResult, ToolStep
from src.security import redact, truncate
from src.tools.unit_converter import is_known_unit

ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
EXPR = re.compile(r"[\d\s.()+\-*/^%×÷]{3,}")
CONVERT = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([A-Za-z]+(?:\s?oz)?)\s+(?:to|in|into|as)\s+°?([A-Za-z]+)", re.I)
CALC_WORDS = re.compile(r"\b(calc\w*|compute|evaluate|solve|what(?:'s| is)|how much is)\b|=", re.I)
TEXT_WORDS = re.compile(r"\b(analy[sz]e|word count|count (?:the )?words|how many words|keywords?|character count|sentence count)\b", re.I)
WEB_WORDS = re.compile(r"\b(search|look up|research|news|web)\b", re.I)
NOW_WORDS = re.compile(r"\b(time|date|today|now|current|what day)\b", re.I)
DIFF_WORDS = re.compile(r"\b(between|difference|diff|days|from|until|since|how long)\b", re.I)
TZ = re.compile(r"\b([A-Za-z]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)\b")


def _extract_text(task: str) -> Optional[str]:
    quoted = re.findall(r"\"([^\"]{3,})\"|“([^”]{3,})”|'([^']{10,})'", task)
    pieces = [next(p for p in q if p) for q in quoted]
    if pieces:
        return max(pieces, key=len)
    if ":" in task:
        rest = task.split(":", 1)[1].strip()
        return rest or None
    return None


def plan(task: str) -> list[tuple[str, dict]]:
    """Return the (tool, args) calls implied by the request. Pure function."""
    calls: list[tuple[str, dict]] = []
    lower = task.lower()
    scan = ISO.sub(" ", task)

    # JSON
    stripped = task.lstrip()
    if re.search(r"\bjson\b|\bvalidate\b", lower) or stripped.startswith(("{", "[")):
        starts = [i for i in (task.find("{"), task.find("[")) if i != -1]
        end = max(task.rfind("}"), task.rfind("]"))
        if starts and end > min(starts):
            calls.append(("json_validator", {"json_text": task[min(starts): end + 1]}))
            scan = scan.replace(task[min(starts): end + 1], " ")

    # Text analysis
    if TEXT_WORDS.search(task) and not calls:
        text = _extract_text(task)
        calls.append(("text_analyzer", {"text": text} if text else {"text": ""}))
        if text:
            scan = scan.replace(text, " ")

    # Unit conversion
    m = CONVERT.search(scan)
    if m and is_known_unit(m.group(2)) and is_known_unit(m.group(3)):
        calls.append(("unit_converter", {"value": float(m.group(1)), "from_unit": m.group(2), "to_unit": m.group(3)}))
        scan = scan.replace(m.group(0), " ")

    # Calculator
    for em in EXPR.finditer(scan):
        s = em.group().strip()
        if re.search(r"[\d)]\s*[+*/^%×÷-]\s*[\d(.\-]", s):
            density = len(re.sub(r"\s", "", s)) / max(1, len(re.sub(r"\s", "", scan)))
            if CALC_WORDS.search(scan) or density >= 0.6:
                calls.append(("calculator", {"expression": s}))
                break

    # Dates
    dates = ISO.findall(task)
    if len(dates) >= 2 and DIFF_WORDS.search(task):
        calls.append(("datetime_tool", {"operation": "diff", "start": dates[0], "end": dates[1]}))
    elif not calls and NOW_WORDS.search(task):
        tz = TZ.search(task)
        args = {"operation": "now"}
        if tz:
            args["timezone"] = tz.group(1)
        calls.append(("datetime_tool", args))

    # Web
    if WEB_WORDS.search(task) and not any(c[0] in ("json_validator", "text_analyzer") for c in calls):
        query = re.sub(r"(?i)^\s*(please\s+)?(search( the web)?( for)?|look up|research)\s*", "", task).strip() or task
        calls.append(("web_research", {"query": query[:400]}))
    return calls


def run_local(task: str, registry: ToolRegistry, secrets: tuple[str, ...] = ()) -> AgentResult:
    started = time.monotonic()
    text = task.strip() if isinstance(task, str) else ""
    if not text:
        return AgentResult(task="", mode="local", status="error", answer="Please enter a task.", error="Empty task.")
    if len(text) > 4000:
        err = "Task too long (limit 4000 characters)."
        return AgentResult(task=text[:200], mode="local", status="error", answer=err, error=err)
    calls = plan(text)
    steps: list[ToolStep] = []
    if not calls:
        steps.append(ToolStep(index=1, kind="router", name="rule-based router", status="skipped",
                              summary="No rule matched this request."))
        msg = ("Local mode is a rule-based tool router, not an AI model, so it could not map this request to a tool. "
               "Try: 'Calculate (12.5 + 3) * 4', 'Convert 5 km to miles', 'Days between 2025-01-01 and 2025-03-15', "
               "'Validate this JSON: {\"a\": 1}', or 'Analyze this text: \"...\"'. Add GOOGLE_API_KEY to use the AI agent.")
        return AgentResult(task=text, mode="local", status="no_match", answer=msg, steps=steps,
                           duration_ms=int((time.monotonic() - started) * 1000))
    steps.append(ToolStep(index=1, kind="router", name="rule-based router", status="success",
                          summary="Matched: " + ", ".join(c[0] for c in calls)))
    lines, errors, ok_count = [], [], 0
    for name, args in calls:
        t0 = time.monotonic()
        outcome = registry.execute(name, args)
        ok_count += outcome.ok
        summary = outcome.text if outcome.ok else (outcome.error or "")
        steps.append(ToolStep(index=len(steps) + 1, kind="tool", name=name,
                              input={k: truncate(str(v), 120) for k, v in args.items()},
                              status="success" if outcome.ok else "error",
                              duration_ms=int((time.monotonic() - t0) * 1000),
                              summary=redact(truncate(" ".join(summary.split()), 240), secrets)))
        if outcome.ok:
            lines.append(f"**{name}**: {outcome.text}")
        else:
            errors.append(f"{name}: {outcome.error}")
            lines.append(f"**{name}** failed: {outcome.error}")
    status = "success" if ok_count else "error"
    return AgentResult(task=text, mode="local", status=status, answer="\n\n".join(lines), steps=steps,
                       error="; ".join(errors) or None, duration_ms=int((time.monotonic() - started) * 1000))
