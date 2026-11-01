"""Safe calculator using a restricted AST evaluator (no eval/exec)."""
from __future__ import annotations

import ast
import math
import operator
import re

from src.tools.base import ToolError

MAX_LEN, MAX_DEPTH, MAX_EXP, MAX_BITS = 200, 40, 1000, 10_000
_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/%().,\s^×÷=]+$")
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": (math.sqrt, 1, 1), "abs": (abs, 1, 1), "round": (round, 1, 2),
          "floor": (math.floor, 1, 1), "ceil": (math.ceil, 1, 1)}
_CONST = {"pi": math.pi, "e": math.e}


def _check(v):
    if isinstance(v, (bool, complex)):
        raise ToolError("Result is not a real number.")
    if isinstance(v, float) and not math.isfinite(v):
        raise ToolError("Result is not a finite number.")
    if isinstance(v, int) and v.bit_length() > MAX_BITS:
        raise ToolError("Result is too large.")
    return v


def _binary(op, a, b):
    if op not in _BIN:
        raise ToolError("Unsupported operator.")
    if op in (ast.Div, ast.FloorDiv, ast.Mod) and b == 0:
        raise ToolError("Division by zero is not allowed.")
    if op is ast.Pow:
        if abs(b) > MAX_EXP:
            raise ToolError(f"Exponent too large (limit {MAX_EXP}).")
        if a == 0 and b < 0:
            raise ToolError("Division by zero is not allowed.")
        if isinstance(a, int) and isinstance(b, int) and b >= 0 and abs(a).bit_length() * b > MAX_BITS:
            raise ToolError("Result is too large.")
    try:
        return _check(_BIN[op](a, b))
    except OverflowError:
        raise ToolError("Result is too large.") from None
    except ZeroDivisionError:
        raise ToolError("Division by zero is not allowed.") from None


def _eval(node, depth=0):
    if depth > MAX_DEPTH:
        raise ToolError("Expression is nested too deeply.")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("Only numbers are allowed.")
        return _check(node.value)
    if isinstance(node, ast.Name):
        if node.id in _CONST:
            return _CONST[node.id]
        raise ToolError(f"Unknown name '{node.id}'.")
    if isinstance(node, ast.BinOp):
        return _binary(type(node.op), _eval(node.left, depth + 1), _eval(node.right, depth + 1))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _check(_UN[type(node.op)](_eval(node.operand, depth + 1)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ToolError("Unsupported function call.")
        fn, lo, hi = _FUNCS[node.func.id]
        if not lo <= len(node.args) <= hi:
            raise ToolError(f"Wrong number of arguments for '{node.func.id}'.")
        args = [_eval(a, depth + 1) for a in node.args]
        if node.func.id == "round" and len(args) == 2 and (not isinstance(args[1], int) or abs(args[1]) > 15):
            raise ToolError("round() digits must be a small integer.")
        try:
            return _check(fn(*args))
        except (ValueError, OverflowError):
            raise ToolError("Math domain error or result out of range.") from None
    raise ToolError("Unsupported syntax in expression.")


def evaluate(expression) -> int | float:
    if not isinstance(expression, str):
        raise ToolError("Expression must be text.")
    expr = expression.strip().rstrip("=").strip()
    if not expr:
        raise ToolError("Expression is empty.")
    if len(expr) > MAX_LEN:
        raise ToolError(f"Expression too long (limit {MAX_LEN} characters).")
    if not _CHARS.match(expr):
        raise ToolError("Expression contains unsupported characters.")
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ToolError("Could not parse the expression.") from None
    try:
        return _eval(tree)
    except RecursionError:
        raise ToolError("Expression is nested too deeply.") from None


def format_number(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else format(v, ".12g")
    return str(v)


def calculate(expression: str) -> dict:
    return {"expression": expression.strip(), "result": format_number(evaluate(expression))}
