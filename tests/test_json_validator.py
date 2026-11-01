import json

import pytest

from src.tools.base import ToolError
from src.tools.json_validator import validate_json


def test_valid_object_pretty_and_structure():
    r = validate_json('{"b":[1,2,{"c":null}],"a":"x"}', indent=4)
    assert r["valid"] and r["syntax_valid"] and r["error"] is None
    assert json.loads(r["pretty"]) == {"b": [1, 2, {"c": None}], "a": "x"}
    assert r["pretty"].splitlines()[1].startswith("    ")
    assert r["structure"] == {"top_level_type": "object", "depth": 3, "key_count": 2, "keys": ["b", "a"]}


def test_valid_array_and_scalar():
    assert validate_json("[1,2,3]")["structure"]["length"] == 3
    assert validate_json("42")["structure"]["top_level_type"] == "number"
    assert validate_json("true")["structure"]["top_level_type"] == "boolean"


def test_syntax_errors_report_location():
    r = validate_json('{"a": 1,\n "b": }')
    assert not r["valid"] and not r["syntax_valid"]
    assert r["error"]["line"] == 2 and r["error"]["column"] >= 1 and r["pretty"] is None


def test_nan_and_trailing_comma_invalid():
    assert not validate_json('{"a": NaN}')["valid"]
    assert not validate_json("[1,2,]")["valid"]
    assert not validate_json("{'a': 1}")["valid"]


def test_required_keys():
    r = validate_json('{"a": 1}', required_keys=["a", "b"])
    assert r["syntax_valid"] and not r["valid"] and r["missing_keys"] == ["b"]
    assert validate_json('{"a": 1}', required_keys=["a"])["valid"]
    assert validate_json("[1]", required_keys=["a"])["missing_keys"] == ["a"]


def test_deep_nesting_does_not_crash():
    r = validate_json("[" * 100_000 + "]" * 100_000)
    assert not r["valid"]


def test_invalid_arguments():
    for bad in ["", "  ", None, 5]:
        with pytest.raises(ToolError):
            validate_json(bad)
    with pytest.raises(ToolError):
        validate_json("{}", indent=9)
    with pytest.raises(ToolError):
        validate_json("{}", required_keys="a")
    with pytest.raises(ToolError):
        validate_json("x" * 200_001)
