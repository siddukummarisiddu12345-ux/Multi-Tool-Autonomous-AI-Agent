import pytest

from src.tools.base import ToolError
from src.tools.calculator import calculate, evaluate


def test_valid_expressions():
    cases = {"1 + 2": 3, "10 - 4 * 2": 2, "(10 - 4) * 2": 12, "2 ** 10": 1024, "2 ^ 3": 8, "7 / 2": 3.5,
             "7 // 2": 3, "7 % 3": 1, "0.1 + 0.2 * 3": 0.7, "-2 ** 2": -4, "sqrt(16) + abs(-3)": 7,
             "round(3.14159, 2)": 3.14, "12.5 * 4 / 2": 25}
    for expr, expected in cases.items():
        assert evaluate(expr) == pytest.approx(expected), expr


def test_calculate_output():
    assert calculate("(12.5 + 3) * 4") == {"expression": "(12.5 + 3) * 4", "result": "62"}


def test_invalid_expressions():
    for expr in ["", "   ", "2 +", "(1 + 2", "abc", "1 2", "sqrt()", "sqrt(-1)", None, 5]:
        with pytest.raises(ToolError):
            evaluate(expr)


def test_division_by_zero():
    for expr in ["1 / 0", "5 // 0", "5 % 0", "0 ** -1"]:
        with pytest.raises(ToolError):
            evaluate(expr)


def test_dangerous_expressions_rejected():
    for expr in ["__import__('os').system('ls')", "eval('1+1')", "open('/etc/passwd')", "().__class__",
                 "[x for x in range(3)]", "lambda: 1", "'a' * 3", "exec('x=1')", "(1).real", "abs.__call__(1)"]:
        with pytest.raises(ToolError):
            evaluate(expr)


def test_resource_limits():
    for expr in ["9 ** 9 ** 9", "10 ** 100000", "1" * 201, "-" * 100 + "1", "10.0 ** 999 * 10 ** 999"]:
        with pytest.raises(ToolError):
            evaluate(expr)
