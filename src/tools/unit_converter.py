"""Length, weight, temperature and volume conversions."""
from __future__ import annotations

import math

from src.tools.base import ToolError

# category -> {canonical unit: factor to base unit}
_FACTORS = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048,
               "yd": 0.9144, "mi": 1609.344},
    "weight": {"mg": 1e-6, "g": 1e-3, "kg": 1.0, "t": 1000.0, "oz": 0.028349523125, "lb": 0.45359237},
    "volume": {"ml": 0.001, "l": 1.0, "tsp": 0.00492892159375, "tbsp": 0.01478676478125,
               "floz": 0.0295735295625, "cup": 0.2365882365, "pt": 0.473176473,
               "qt": 0.946352946, "gal": 3.785411784},
    "temperature": {"c": None, "f": None, "k": None},
}
_ALIASES = {
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "kms": "km",
    "inch": "in", "inches": "in", "foot": "ft", "feet": "ft", "yard": "yd", "yards": "yd",
    "mile": "mi", "miles": "mi",
    "milligram": "mg", "milligrams": "mg", "gram": "g", "grams": "g", "kilogram": "kg",
    "kilograms": "kg", "kgs": "kg", "tonne": "t", "tonnes": "t", "ton": "t", "ounce": "oz",
    "ounces": "oz", "pound": "lb", "pounds": "lb", "lbs": "lb",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "teaspoon": "tsp", "teaspoons": "tsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "fl oz": "floz", "fluid ounce": "floz", "fluid ounces": "floz", "cups": "cup",
    "pint": "pt", "pints": "pt", "quart": "qt", "quarts": "qt", "gallon": "gal", "gallons": "gal",
    "celsius": "c", "centigrade": "c", "fahrenheit": "f", "kelvin": "k",
}
_CATEGORY = {u: cat for cat, units in _FACTORS.items() for u in units}
_ABS_ZERO = {"c": -273.15, "f": -459.67, "k": 0.0}


def normalize_unit(unit) -> str:
    if not isinstance(unit, str):
        raise ToolError("Unit must be text.")
    u = unit.strip().lower().replace("°", "").replace("degrees", "").replace("degree", "").strip()
    return _ALIASES.get(u, u)


def is_known_unit(unit) -> bool:
    try:
        return normalize_unit(unit) in _CATEGORY
    except ToolError:
        return False


def _to_celsius(v, u):
    return v if u == "c" else (v - 32) * 5 / 9 if u == "f" else v - 273.15


def _from_celsius(v, u):
    return v if u == "c" else v * 9 / 5 + 32 if u == "f" else v + 273.15


def convert(value, from_unit, to_unit) -> dict:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolError("Value must be a finite number.")
    src, dst = normalize_unit(from_unit), normalize_unit(to_unit)
    for label, u, raw in (("from", src, from_unit), ("to", dst, to_unit)):
        if u not in _CATEGORY:
            raise ToolError(f"Unknown {label} unit '{str(raw)[:30]}'. Supported: {', '.join(sorted(_CATEGORY))}.")
    if _CATEGORY[src] != _CATEGORY[dst]:
        raise ToolError(f"Cannot convert {_CATEGORY[src]} ({src}) to {_CATEGORY[dst]} ({dst}).")
    cat = _CATEGORY[src]
    if cat == "temperature":
        if value < _ABS_ZERO[src]:
            raise ToolError("Temperature is below absolute zero.")
        result = _from_celsius(_to_celsius(value, src), dst)
    else:
        result = value * _FACTORS[cat][src] / _FACTORS[cat][dst]
    if not math.isfinite(result):
        raise ToolError("Result is out of range.")
    return {"category": cat, "value": value, "from_unit": src, "to_unit": dst,
            "result": float(format(result, ".10g")),
            "formatted": f"{format(value, '.10g')} {src} = {format(result, '.10g')} {dst}"}
