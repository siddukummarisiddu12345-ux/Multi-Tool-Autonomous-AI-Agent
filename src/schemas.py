"""Pydantic schemas: tool inputs (also used as LangChain tool args_schema) and agent outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalculatorInput(_Strict):
    expression: str = Field(..., min_length=1, max_length=200,
                            description="Arithmetic expression, e.g. '(12.5 + 3) * 4 / 2'.")


class TextAnalyzerInput(_Strict):
    text: str = Field(..., min_length=1, max_length=100_000, description="The text to analyze.")
    top_n: int = Field(10, ge=1, le=50, description="How many top keywords to return.")


class JSONValidatorInput(_Strict):
    json_text: str = Field(..., min_length=1, max_length=200_000, description="The JSON text to validate.")
    indent: int = Field(2, ge=0, le=8, description="Indent width for pretty-printing.")
    required_keys: Optional[list[str]] = Field(None, max_length=50,
                                               description="Top-level keys that must exist.")


class DateTimeInput(_Strict):
    operation: Literal["now", "diff"] = Field(..., description="'now' for current date/time, 'diff' for a date difference.")
    timezone: Optional[str] = Field(None, max_length=64, description="IANA timezone for 'now', e.g. 'Asia/Kolkata'.")
    start: Optional[str] = Field(None, max_length=40, description="ISO start date/datetime for 'diff'.")
    end: Optional[str] = Field(None, max_length=40, description="ISO end date/datetime for 'diff'.")


class UnitConverterInput(_Strict):
    value: float = Field(..., description="Numeric value to convert.")
    from_unit: str = Field(..., min_length=1, max_length=30, description="Source unit, e.g. 'km', 'lb', 'F', 'gal'.")
    to_unit: str = Field(..., min_length=1, max_length=30, description="Target unit.")


class WebResearchInput(_Strict):
    query: str = Field(..., min_length=1, max_length=400, description="A concise web search query.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results.")


class ToolStep(BaseModel):
    index: int
    kind: Literal["llm", "router", "tool"]
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error", "skipped"]
    duration_ms: int = 0
    summary: str = ""


class AgentResult(BaseModel):
    task: str
    mode: Literal["agent", "local"]
    status: Literal["success", "error", "max_steps", "not_configured", "no_match"]
    answer: str
    steps: list[ToolStep] = Field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
