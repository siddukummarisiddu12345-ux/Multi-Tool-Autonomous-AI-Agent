"""Tool registry: discovery, Pydantic validation and contained execution."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from src import schemas
from src.config import Settings
from src.security import redact, safe_error, truncate
from src.tools.base import ToolError
from src.tools.calculator import calculate
from src.tools.datetime_tool import datetime_info
from src.tools.json_validator import validate_json
from src.tools.text_analyzer import analyze_text
from src.tools.unit_converter import convert
from src.tools.web_research import web_research

MAX_MODEL_OUTPUT = 6000


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[Any], dict]
    status: Callable[[], tuple[str, str]] = lambda: ("ready", "Ready")


@dataclass
class ToolOutcome:
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    text: str = ""       # human-readable
    validated: Optional[dict] = None

    def for_model(self, secrets=()) -> str:
        body = self.text if not self.ok else json.dumps(self.data, default=str, ensure_ascii=False)
        body = body if self.ok else f"ERROR: {self.error}"
        return redact(truncate(body, MAX_MODEL_OUTPUT), secrets)


def format_result(name: str, d: dict) -> str:
    """Readable summary of a successful tool result."""
    if name == "calculator":
        return f"{d['expression']} = {d['result']}"
    if name == "unit_converter":
        return d["formatted"]
    if name == "text_analyzer":
        kw = ", ".join(f"{k['word']} ({k['count']})" for k in d["keywords"]) or "none"
        return (f"{d['word_count']} words, {d['character_count']} characters "
                f"({d['character_count_no_spaces']} without spaces), {d['sentence_count']} sentences, "
                f"{d['paragraph_count']} paragraphs. Top keywords: {kw}.")
    if name == "json_validator":
        if not d["syntax_valid"]:
            e = d["error"]
            where = f" at line {e['line']}, column {e['column']}" if e.get("line") else ""
            return f"Invalid JSON: {e['message']}{where}."
        s = d["structure"]
        msg = f"Valid JSON ({s['top_level_type']}, depth {s['depth']})."
        if d["missing_keys"]:
            msg = f"JSON is syntactically valid but missing required keys: {', '.join(d['missing_keys'])}."
        return msg + "\n" + d["pretty"]
    if name == "datetime_tool":
        if d["operation"] == "now":
            return f"{d['weekday']}, {d['date']} {d['time']} ({d['timezone']}, UTC{d['utc_offset']})"
        c = d["calendar_difference"]
        return (f"{d['total_days']} days ({d['weeks']} weeks, {d['extra_days']} days; "
                f"{c['years']}y {c['months']}m {c['days']}d), {d['direction']}.")
    if name == "web_research":
        if not d["results"]:
            return f"No results found for '{d['query']}'."
        return "\n".join(f"{i}. {r['title']} - {r['url']}\n   {r['snippet']}" for i, r in enumerate(d["results"], 1))
    return json.dumps(d, default=str)


class ToolRegistry:
    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._secrets = secrets

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: Any) -> Optional[ToolSpec]:
        return self._tools.get(name) if isinstance(name, str) else None

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def execute(self, name: Any, args: Any) -> ToolOutcome:
        """Validate and run a tool. Never raises."""
        spec = self.get(name)
        if spec is None:
            return ToolOutcome(False, error=f"Unknown tool '{str(name)[:40]}'. Available: {', '.join(self._tools)}.")
        if not isinstance(args, dict):
            return ToolOutcome(False, error="Tool arguments must be a JSON object.")
        try:
            model = spec.input_model.model_validate(args)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e.get('loc', ())))}: {e.get('msg', 'invalid')}"
                               for e in exc.errors()[:4])
            return ToolOutcome(False, error=f"Invalid arguments for '{name}': {detail}")
        try:
            data = spec.run(model)
        except ToolError as exc:
            return ToolOutcome(False, error=redact(str(exc), self._secrets))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return ToolOutcome(False, error=f"Tool '{name}' failed: {safe_error(exc, self._secrets)}")
        text = redact(format_result(name, data), self._secrets)
        return ToolOutcome(True, data=data, text=text, validated=model.model_dump())


def build_registry(settings: Settings) -> ToolRegistry:
    reg = ToolRegistry(settings.secrets)
    reg.register(ToolSpec(
        "calculator",
        "Evaluate an arithmetic expression exactly (+ - * / // % ** ^, parentheses, decimals, sqrt/abs/round/floor/ceil, pi, e). "
        "Use for any calculation.",
        schemas.CalculatorInput, lambda m: calculate(m.expression)))
    reg.register(ToolSpec(
        "text_analyzer",
        "Count words, characters, sentences and paragraphs in a text and list its most frequent keywords.",
        schemas.TextAnalyzerInput, lambda m: analyze_text(m.text, m.top_n)))
    reg.register(ToolSpec(
        "json_validator",
        "Parse and validate JSON text, report syntax errors with line/column, check required top-level keys, "
        "and pretty-print valid JSON.",
        schemas.JSONValidatorInput, lambda m: validate_json(m.json_text, m.indent, m.required_keys)))
    reg.register(ToolSpec(
        "datetime_tool",
        "Get the current date/time (optionally in an IANA timezone) or compute the difference between two ISO dates/datetimes.",
        schemas.DateTimeInput, lambda m: datetime_info(m.operation, m.timezone, m.start, m.end)))
    reg.register(ToolSpec(
        "unit_converter",
        "Convert between common length (mm cm m km in ft yd mi), weight (mg g kg t oz lb), "
        "temperature (C F K) and volume (ml l tsp tbsp floz cup pt qt gal) units.",
        schemas.UnitConverterInput, lambda m: convert(m.value, m.from_unit, m.to_unit)))
    reg.register(ToolSpec(
        "web_research",
        "Search the live web for current information or facts you cannot know. Returns real titles, snippets and URLs. "
        "Fails with a clear message if search is not configured.",
        schemas.WebResearchInput,
        lambda m: web_research(m.query, settings.tavily_api_key, min(m.max_results, settings.search_max_results * 2),
                               settings.request_timeout, settings.max_retries),
        status=lambda: ("ready", "Ready") if settings.has_search else ("setup", "Set TAVILY_API_KEY")))
    return reg
