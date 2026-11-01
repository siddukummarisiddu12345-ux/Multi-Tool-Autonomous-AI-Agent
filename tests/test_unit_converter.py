import pytest

from src.tools.base import ToolError
from src.tools.unit_converter import convert, is_known_unit


def r(v, a, b):
    return convert(v, a, b)["result"]


def test_length():
    assert r(1, "km", "m") == 1000 and r(1, "mi", "km") == pytest.approx(1.609344)
    assert r(12, "in", "ft") == 1 and r(100, "cm", "m") == 1


def test_weight():
    assert r(1, "kg", "g") == 1000 and r(1, "lb", "oz") == pytest.approx(16)
    assert r(1, "lb", "kg") == pytest.approx(0.45359237)


def test_volume():
    assert r(1, "l", "ml") == 1000 and r(1, "gal", "l") == pytest.approx(3.785411784)
    assert r(1, "cup", "floz") == pytest.approx(8, rel=1e-3)


def test_temperature():
    assert r(100, "c", "f") == 212 and r(32, "f", "c") == 0 and r(0, "c", "k") == 273.15
    assert r(-40, "c", "f") == -40


def test_aliases_and_degree_symbol():
    assert r(5, "Kilometers", "Miles") == pytest.approx(3.106855961)
    assert r(212, "°F", "°C") == 100 and is_known_unit("Pounds") and not is_known_unit("parsec")


def test_formatted_output():
    assert convert(100, "C", "F")["formatted"] == "100 c = 212 f"


def test_invalid_conversions():
    bad = [(1, "kg", "m"), (1, "parsec", "m"), (1, "m", "furlong"), (-1, "k", "c"), (-500, "c", "f"),
           (float("nan"), "m", "km"), (float("inf"), "m", "km"), ("5", "m", "km"), (True, "m", "km"),
           (1, None, "m"), (1e308, "km", "mm")]
    for v, a, b in bad:
        with pytest.raises(ToolError):
            convert(v, a, b)
