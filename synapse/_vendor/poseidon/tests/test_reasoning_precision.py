"""Exact verification must preserve the numeric literal supplied by the user."""
from fractions import Fraction

import pytest

from poseidon.reasoning import MathError, extract_math, solve, solve_quadratic


@pytest.mark.parametrize("expression, expected", [
    ("9007199254740993.0 - 9007199254740992.0", "1"),
    ("1.00000000000000001 - 1", "1/100000000000000000"),
    ("0.10000000000000001 + 0.2", "30000000000000001/100000000000000000"),
    ("x + 1.00000000000000001 = 2", "99999999999999999/100000000000000000"),
    ("det([[1.00000000000000001,1],[1,1]])", "1/100000000000000000"),
    ("diff(1.00000000000000001*x, x)", "100000000000000001/100000000000000000"),
    ("integrate(1, x, 1, 1.00000000000000001)", "1/100000000000000000"),
    ("diff(x^2, x, at=1.00000000000000001)", "100000000000000001/50000000000000000"),
    ("1e-17 + 1.00000000000000001", "50000000000000001/50000000000000000"),
    ("det([[1e-17,0],[0,1]])", "1/100000000000000000"),
])
def test_exact_results_retain_literal_precision(expression, expected):
    result = solve(expression)
    assert result["answer"] == expected
    assert result["verified"] is True


@pytest.mark.parametrize("expression, expected", [
    ("x+y=1.00000000000000001, x-y=0", "100000000000000001/200000000000000000"),
    ("x+y+z=1.00000000000000001, y=0, z=0", "100000000000000001/100000000000000000"),
])
def test_linear_systems_retain_literal_precision(expression, expected):
    result = solve(expression)
    assert result["x"] == expected
    assert result["verified"] is True


def test_direct_quadratic_retains_literal_precision():
    result = solve_quadratic("x^2 - 1.00000000000000001*x = 0")
    assert {Fraction(root) for root in result["roots"]} == {
        Fraction(0), Fraction(100000000000000001, 100000000000000000)}
    assert result["verified"] is True


def test_scientific_notation_is_extracted_without_changing_literals():
    expression = "1.00000000000000001e0 - 1e0"
    assert extract_math("Compute " + expression) == expression
    assert solve(expression)["answer"] == "1/100000000000000000"


def test_scientific_notation_inside_function_is_extracted():
    expression = "det([[1e-17,0],[0,1]])"
    assert extract_math("Compute " + expression) == expression


@pytest.mark.parametrize("expression, expected", [
    ("1e80", "1" + "0" * 80),
    ("1e-80", "1/1" + "0" * 80),
    ("0e999999999999999999999999", "0"),
    ("100e-82", "1/1" + "0" * 80),
])
def test_numeric_budget_keeps_exact_boundary_values(expression, expected):
    assert solve(expression)["answer"] == expected


@pytest.mark.parametrize("expression", [
    "1e999999999999999999999999", "1e-999999999999999999999999",
    "det([[1e999999999999999999999999]])", "1e81", "1e-81",
])
def test_literal_exponents_respect_numeric_budget(expression):
    with pytest.raises(MathError):
        solve(expression)


@pytest.mark.parametrize("expression", ["gcd(1.0,2)", "gcd(1e0,2)", "sqrt(True)"])
def test_decimal_fix_does_not_admit_noninteger_number_theory_or_booleans(expression):
    with pytest.raises(MathError):
        solve(expression)
