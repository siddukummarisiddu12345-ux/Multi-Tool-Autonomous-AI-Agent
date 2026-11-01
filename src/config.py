"""Configuration from environment variables (or Streamlit secrets exposed as env vars)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

DEFAULT_MODEL = "gemini-2.5-flash"


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int((env.get(name) or "").strip() or default)
    except ValueError:
        v = default
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Settings:
    google_api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    tavily_api_key: str = field(default="", repr=False)
    max_steps: int = 6          # max tool executions per task
    max_retries: int = 2        # LLM / search retry limit
    request_timeout: float = 30.0
    search_max_results: int = 5

    @property
    def has_llm(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_search(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.google_api_key, self.tavily_api_key) if s)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            except ImportError:
                pass
            env = os.environ
        key = (env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY") or "").strip()
        return cls(
            google_api_key=key,
            model=(env.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL,
            tavily_api_key=(env.get("TAVILY_API_KEY") or "").strip(),
            max_steps=_int(env, "MAX_STEPS", 6, 1, 12),
            max_retries=_int(env, "MAX_RETRIES", 2, 0, 5),
        )
