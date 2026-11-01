"""JSON parsing, structure validation and pretty-printing."""
from __future__ import annotations

import json
from typing import Optional

from src.tools.base import ToolError

MAX_CHARS = 200_000


def _bad_constant(name):
    raise ValueError(f"Invalid JSON constant '{name}' (NaN/Infinity are not valid JSON)")


def _depth(obj) -> int:
    best, stack = 0, [(obj, 1)]
    while stack:
        cur, d = stack.pop()
        best = max(best, d)
        if isinstance(cur, dict):
            stack.extend((v, d + 1) for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend((v, d + 1) for v in cur if isinstance(v, (dict, list)))
    return best


def validate_json(json_text, indent: int = 2, required_keys: Optional[list] = None) -> dict:
    """Return a validation report. Invalid JSON is reported, not raised."""
    if not isinstance(json_text, str) or not json_text.strip():
        raise ToolError("JSON text is empty.")
    if len(json_text) > MAX_CHARS:
        raise ToolError(f"JSON too long (limit {MAX_CHARS:,} characters).")
    if isinstance(indent, bool) or not isinstance(indent, int) or not 0 <= indent <= 8:
        raise ToolError("indent must be an integer between 0 and 8.")
    if required_keys is not None and (
        not isinstance(required_keys, list) or not all(isinstance(k, str) for k in required_keys)
    ):
        raise ToolError("required_keys must be a list of strings.")
    try:
        obj = json.loads(json_text, parse_constant=_bad_constant)
    except json.JSONDecodeError as exc:
        return {"valid": False, "syntax_valid": False,
                "error": {"message": exc.msg, "line": exc.lineno, "column": exc.colno},
                "pretty": None, "structure": None, "missing_keys": []}
    except ValueError as exc:
        return {"valid": False, "syntax_valid": False,
                "error": {"message": str(exc), "line": None, "column": None},
                "pretty": None, "structure": None, "missing_keys": []}
    except RecursionError:
        return {"valid": False, "syntax_valid": False,
                "error": {"message": "JSON is nested too deeply", "line": None, "column": None},
                "pretty": None, "structure": None, "missing_keys": []}
    kind = {dict: "object", list: "array", str: "string", bool: "boolean", type(None): "null"}.get(
        type(obj), "number")
    structure = {"top_level_type": kind, "depth": _depth(obj)}
    if isinstance(obj, dict):
        structure.update(key_count=len(obj), keys=list(obj)[:20])
    elif isinstance(obj, list):
        structure["length"] = len(obj)
    missing = []
    if required_keys:
        if not isinstance(obj, dict):
            missing = list(required_keys)
        else:
            missing = [k for k in required_keys if k not in obj]
    return {"valid": not missing, "syntax_valid": True, "error": None,
            "pretty": json.dumps(obj, indent=indent, ensure_ascii=False),
            "structure": structure, "missing_keys": missing}
