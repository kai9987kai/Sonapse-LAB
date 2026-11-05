import pytest
from poseidon.reasoning import MathError, extract_math, solve, solve_quadratic, solve_system


def test_quadratic_exact_rational_roots():
    res = solve("x^2 - 5*x + 6 = 0")
    assert res["roots"] == ["3", "2"]
    assert res["verified"]
    assert res["backend"] == "exact-rational-quadratic-solver"


def test_quadratic_double_root():
    res = solve("x^2 - 6*x + 9 = 0")
    assert res["roots"] == ["3"]
    assert res["verified"]
    assert "double root" in res["text"]


def test_quadratic_radical_roots():
    res = solve("x^2 - 2 = 0")
    assert len(res["roots"]) == 2
    assert res["verified"]
    assert res["backend"] == "exact-radical-quadratic-solver"


def test_quadratic_complex_roots():
    res = solve("x^2 + 4 = 0")
    assert res["roots"] == ["0 + 2i", "0 - 2i"]
    assert res["verified"]
    assert res["backend"] == "exact-complex-quadratic-solver"


def test_linear_system_exact_solution():
    res = solve("2*x + 3*y = 13, x - y = 4")
    assert res["x"] == "5"
    assert res["y"] == "1"
    assert res["verified"]
    assert res["backend"] == "exact-rational-linear-system-solver"


def test_linear_system_inconsistent_and_dependent():
    with pytest.raises(MathError, match="no solution"):
        solve("x + y = 1, x + y = 2")
    with pytest.raises(MathError, match="infinitely many"):
        solve("2*x + 2*y = 4, x + y = 2")


def test_math_extraction_supports_quadratics_and_systems():
    assert extract_math("Please solve x^2 - 4 = 0") == "x^2 - 4 = 0"
    assert extract_math("Compute 3*x + y = 10, x - y = 2") == "3*x + y = 10, x - y = 2"
