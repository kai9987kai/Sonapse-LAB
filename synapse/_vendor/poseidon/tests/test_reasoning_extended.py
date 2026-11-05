import pytest
from poseidon.reasoning import solve, MathError, extract_math, solve_linear_system, solve_polynomial


def test_polynomial_cubic_three_rational_roots():
    res = solve("x^3 - 6*x^2 + 11*x - 6 = 0")
    assert res["roots"] == ["1", "2", "3"]
    assert res["verified"] is True
    assert res["exact"] is True
    assert res["backend"] == "exact-rational-polynomial-solver"


def test_polynomial_factored_input():
    res = solve("(x - 1)*(x + 2)*(x - 3) = 0")
    assert "1" in res["roots"]
    assert "-2" in res["roots"]
    assert "3" in res["roots"]
    assert res["verified"] is True


def test_polynomial_repeated_root():
    res = solve("x^3 - 3*x^2 + 3*x - 1 = 0")
    assert res["roots"] == ["1"]
    assert res["multiplicities"] == [3]
    assert res["verified"] is True


def test_polynomial_quartic_mixed():
    res = solve("x^4 - 16 = 0")
    assert "2" in res["roots"]
    assert "-2" in res["roots"]
    assert res["verified"] is True


def test_polynomial_degree_rejection():
    with pytest.raises(MathError, match=r"(exponents between 0 and 8|degree exceeds)"):
        solve("x^9 = 1")
    with pytest.raises(MathError, match=r"(exponents between 0 and 8|degree exceeds)"):
        solve("x^5 * x^4 = 1")


def test_linear_system_3x3_gaussian():
    res = solve("x + y + z = 6, 2*x - y + z = 3, x + 2*y - z = 2")
    assert res["x"] == "1"
    assert res["y"] == "2"
    assert res["z"] == "3"
    assert res["verified"] is True
    assert res["backend"] == "exact-rational-gaussian-elimination"


def test_linear_system_4x4_gaussian():
    eqs = "x + y + z + w = 10, x - y + z - w = 0, 2*x + y - z = 3, x + 2*w = 7"
    res = solve(eqs)
    assert res["verified"] is True
    assert res["backend"] == "exact-rational-gaussian-elimination"
    assert set(res["variables"]) == {"x", "y", "z", "w"}


def test_linear_system_inconsistent_3x3():
    with pytest.raises(MathError, match="no solution"):
        solve("x + y + z = 1, x + y + z = 2, 2*x + y = 3")


def test_linear_system_dependent_3x3():
    with pytest.raises(MathError, match="infinitely many"):
        solve("x + y + z = 3, 2*x + 2*y + 2*z = 6, x - y = 0")


def test_extract_math_extended():
    assert extract_math("solve x^3 - 8 = 0") == "x^3 - 8 = 0"
    assert extract_math("compute x + y + z = 6, x - y = 0, y - z = -1") == "x + y + z = 6, x - y = 0, y - z = -1"
    assert extract_math("What are you doing today?") is None
    assert extract_math("How are you?") is None


# ---------------------------------------------------------------------------
# 1. Symbolic Polynomial Calculus Tests (exact-symbolic-calculus)
# ---------------------------------------------------------------------------

def test_calculus_differentiation_first_and_higher_order():
    res1 = solve("diff(x^3 - 4*x^2 + 5*x - 2, x)")
    assert res1["answer"] == "3*x^2 - 8*x + 5"
    assert res1["derivative"] == "3*x^2 - 8*x + 5"
    assert res1["order"] == 1
    assert res1["verified"] is True
    assert res1["backend"] == "exact-symbolic-calculus"

    res2 = solve("diff(x^4 - 2*x^2, x, 2)")
    assert res2["answer"] == "12*x^2 - 4"
    assert res2["order"] == 2
    assert res2["verified"] is True
    assert res2["backend"] == "exact-symbolic-calculus"

    res_ddx = solve("d/dx (x^3 - 4*x^2 + 5*x - 2)")
    assert res_ddx["answer"] == "3*x^2 - 8*x + 5"
    assert res_ddx["verified"] is True
    assert res_ddx["backend"] == "exact-symbolic-calculus"

    res_d2dx2 = solve("d^2/dx^2 (x^4 - 2*x^2)")
    assert res_d2dx2["answer"] == "12*x^2 - 4"
    assert res_d2dx2["verified"] is True


def test_calculus_differentiation_at_point():
    res = solve("diff(x^3 - 4*x, x, at=2)")
    assert res["answer"] == "8"
    assert res["derivative"] == "3*x^2 - 4"
    assert res["at"] == "2"
    assert res["verified"] is True
    assert res["backend"] == "exact-symbolic-calculus"

    res_frac = solve("diff(x^3 - 4*x, x, 2, at=1/2)")
    assert res_frac["answer"] == "3"
    assert res_frac["verified"] is True


def test_calculus_indefinite_integration():
    res = solve("integrate(3*x^2 - 8*x + 5, x)")
    assert res["answer"] == "x^3 - 4*x^2 + 5*x + C"
    assert res["antiderivative"] == "x^3 - 4*x^2 + 5*x"
    assert res["verified"] is True
    assert res["backend"] == "exact-symbolic-calculus"

    res_frac = solve("integrate(x^2 + x, x)")
    assert res_frac["answer"] == "1/3*x^3 + 1/2*x^2 + C"
    assert res_frac["verified"] is True


def test_calculus_definite_integration():
    res = solve("integrate(3*x^2 + 2*x, x, 0, 2)")
    assert res["answer"] == "12"
    assert res["verified"] is True
    assert res["backend"] == "exact-symbolic-calculus"

    res_frac = solve("integrate(x^3 - x, x, -1, 2)")
    assert res_frac["answer"] == "9/4"
    assert res_frac["verified"] is True


def test_calculus_errors():
    with pytest.raises(MathError):
        solve("diff(x^2, y)")
    with pytest.raises(MathError):
        solve("diff(x^2, x, -1)")
    with pytest.raises(MathError):
        solve("integrate(x^2, x, 0)")


# ---------------------------------------------------------------------------
# 2. Exact Matrix & Linear Algebra Tests (exact-fraction-linear-algebra)
# ---------------------------------------------------------------------------

def test_matrix_det_and_trace():
    res_det = solve("det([[1, 2], [3, 4]])")
    assert res_det["answer"] == "-2"
    assert res_det["verified"] is True
    assert res_det["backend"] == "exact-fraction-linear-algebra"

    res_det3 = solve("det([[6, 1, 1], [4, -2, 5], [2, 8, 7]])")
    assert res_det3["answer"] == "-306"
    assert res_det3["verified"] is True

    res_tr = solve("trace([[1, 2], [3, 4]])")
    assert res_tr["answer"] == "5"
    assert res_tr["verified"] is True
    assert res_tr["backend"] == "exact-fraction-linear-algebra"


def test_matrix_rank():
    res = solve("rank([[1, 2, 3], [2, 4, 6], [1, 0, 1]])")
    assert res["answer"] == "2"
    assert res["rank"] == 2
    assert res["verified"] is True
    assert res["backend"] == "exact-fraction-linear-algebra"

    res_rect = solve("rank([[1, 2, 3, 4], [2, 4, 6, 8]])")
    assert res_rect["answer"] == "1"
    assert res_rect["verified"] is True


def test_matrix_inverse_and_singular_error():
    res = solve("inv([[4, 7], [2, 6]])")
    assert res["answer"] == "[[3/5, -7/10], [-1/5, 2/5]]"
    assert res["verified"] is True
    assert res["backend"] == "exact-fraction-linear-algebra"

    res3 = solve("inv([[1, 0, 2], [0, 1, 0], [0, 0, 1]])")
    assert res3["answer"] == "[[1, 0, -2], [0, 1, 0], [0, 0, 1]]"
    assert res3["verified"] is True

    with pytest.raises(MathError, match="Matrix is singular and cannot be inverted."):
        solve("inv([[1, 2], [2, 4]])")


def test_matrix_eigenvalues_2x2_and_3x3():
    res2 = solve("eigenvalues([[4, 1], [2, 3]])")
    assert res2["answer"] == "2, 5"
    assert res2["eigenvalues"] == ["2", "5"]
    assert res2["trace"] == "7"
    assert res2["determinant"] == "10"
    assert res2["verified"] is True
    assert res2["backend"] == "exact-fraction-linear-algebra"

    res3 = solve("eigenvalues([[2, 0, 0], [0, 3, 4], [0, 4, 9]])")
    assert res3["answer"] == "1, 2, 11"
    assert res3["verified"] is True

    res_rep = solve("eigenvalues([[2, 1], [0, 2]])")
    assert res_rep["answer"] == "2"
    assert res_rep["multiplicities"] == [2]
    assert res_rep["verified"] is True


def test_matrix_validation_errors():
    with pytest.raises(MathError):
        solve("det([[1, 2, 3], [4, 5, 6]])")
    with pytest.raises(MathError):
        solve("trace([[1, 2], [3]])")
    with pytest.raises(MathError):
        solve("rank([[1, 2, 3, 4, 5, 6]])")


# ---------------------------------------------------------------------------
# 3. Exact Number Theory & Modular Arithmetic Tests (exact-number-theory)
# ---------------------------------------------------------------------------

def test_number_theory_gcd_and_lcm():
    res_g = solve("gcd(48, 180)")
    assert res_g["answer"] == "12"
    assert res_g["verified"] is True
    assert res_g["backend"] == "exact-number-theory"

    res_g3 = solve("gcd(48, 180, 300)")
    assert res_g3["answer"] == "12"
    assert res_g3["verified"] is True

    res_l = solve("lcm(12, 18)")
    assert res_l["answer"] == "36"
    assert res_l["verified"] is True

    res_l3 = solve("lcm(12, 15, 20)")
    assert res_l3["answer"] == "60"
    assert res_l3["verified"] is True


def test_number_theory_factor_and_totient():
    res_f = solve("factor(360)")
    assert res_f["answer"] == "2^3 * 3^2 * 5"
    assert res_f["factors"] == {"2": 3, "3": 2, "5": 1}
    assert res_f["verified"] is True
    assert res_f["backend"] == "exact-number-theory"

    res_prime = solve("factor(1000000007)")
    assert res_prime["answer"] == "1000000007"
    assert res_prime["verified"] is True

    res_tot = solve("totient(36)")
    assert res_tot["answer"] == "12"
    assert res_tot["verified"] is True
    assert res_tot["backend"] == "exact-number-theory"

    res_phi = solve("phi(36)")
    assert res_phi["answer"] == "12"
    assert res_phi["verified"] is True

    assert solve("totient(1)")["answer"] == "1"


def test_number_theory_modinv_and_crt():
    res_inv = solve("modinv(3, 11)")
    assert res_inv["answer"] == "4"
    assert res_inv["verified"] is True
    assert res_inv["backend"] == "exact-number-theory"

    res_inv2 = solve("modinv(17, 3120)")
    assert res_inv2["answer"] == "2753"
    assert res_inv2["verified"] is True

    with pytest.raises(MathError, match="gcd"):
        solve("modinv(6, 9)")

    res_crt1 = solve("crt([(2, 3), (3, 5), (2, 7)])")
    assert res_crt1["answer"] == "23"
    assert res_crt1["modulus"] == 105
    assert res_crt1["verified"] is True
    assert res_crt1["backend"] == "exact-number-theory"

    res_crt2 = solve("crt(2, 3, 3, 5, 2, 7)")
    assert res_crt2["answer"] == "23"
    assert res_crt2["modulus"] == 105
    assert res_crt2["verified"] is True

    res_crt_noncoprime = solve("crt([(1, 4), (3, 6)])")
    assert res_crt_noncoprime["answer"] == "9"
    assert res_crt_noncoprime["modulus"] == 12
    assert res_crt_noncoprime["verified"] is True

    with pytest.raises(MathError, match="incompatible"):
        solve("crt([(1, 4), (2, 6)])")


def test_number_theory_bounds_and_errors():
    with pytest.raises(MathError):
        solve("factor(1)")
    with pytest.raises(MathError):
        solve("factor(10000000000000)")
    with pytest.raises(MathError):
        solve("totient(0)")


# ---------------------------------------------------------------------------
# 4. Safe Math Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_math_special_functions_and_english_rejection():
    assert extract_math("Please compute diff(x^3 - 4*x^2 + 5*x - 2, x)") == "diff(x^3 - 4*x^2 + 5*x - 2, x)"
    assert extract_math("What is integrate(3*x^2 + 2*x, x, 0, 2)?") == "integrate(3*x^2 + 2*x, x, 0, 2)"
    assert extract_math("Evaluate d/dx (x^3 - 4*x)") == "d/dx (x^3 - 4*x)"
    assert extract_math("Calculate det([[1, 2], [3, 4]]).") == "det([[1, 2], [3, 4]])"
    assert extract_math("What is inv([[4, 7], [2, 6]])?") == "inv([[4, 7], [2, 6]])"
    assert extract_math("Find eigenvalues([[4, 1], [2, 3]])") == "eigenvalues([[4, 1], [2, 3]])"
    assert extract_math("Compute rank([[1, 2, 3], [2, 4, 6], [1, 0, 1]])") == "rank([[1, 2, 3], [2, 4, 6], [1, 0, 1]])"
    assert extract_math("What is trace([[1, 2], [3, 4]])?") == "trace([[1, 2], [3, 4]])"
    assert extract_math("What is factor(360)?") == "factor(360)"
    assert extract_math("Calculate totient(36)") == "totient(36)"
    assert extract_math("Compute modinv(3, 11)") == "modinv(3, 11)"
    assert extract_math("Solve crt([(2, 3), (3, 5), (2, 7)])") == "crt([(2, 3), (3, 5), (2, 7)])"
    assert extract_math("gcd(48, 180)") == "gcd(48, 180)"
    assert extract_math("lcm(12, 18)") == "lcm(12, 18)"

    # Normal English sentences must return None
    assert extract_math("Explain why seasons change in two sentences.") is None
    assert extract_math("Should we integrate the new module or factor in the delay?") is None
    assert extract_math("Can we trace the rank of each item in the list?") is None
    assert extract_math("Should we diff(the two files) or factor(in the delay)?") is None


def test_solve_sqrt_exact_and_approximation():
    res1 = solve("sqrt(144)")
    assert res1["answer"] == "12"
    assert res1["verified"] is True
    assert res1["backend"] == "exact-radical-solver"

    res2 = solve("sqrt(4/9)")
    assert res2["answer"] == "2/3"
    assert res2["verified"] is True

    res3 = solve("sqrt(2)")
    assert res3["verified"] is False
    assert res3["backend"] == "numeric-radical-approximation"
    assert "1.414" in res3["answer"]

    with pytest.raises(MathError, match="negative number"):
        solve("sqrt(-9)")


def test_natural_language_math_extraction_and_solving():
    # n01: linear equation with implicit mult and "If ..., what is x?"
    assert solve(extract_math("If 2x + 3 = 11, what is x?"))["answer"] == "4"

    # n02: percentage
    assert solve(extract_math("What is 15% of 240?"))["answer"] == "36"

    # n03: natural-language derivative
    d_res = solve(extract_math("What is the derivative of x^3 - 4x?"))
    assert d_res["answer"] == "3*x^2 - 4"

    # n04: natural-language definite integral
    assert solve(extract_math("Find the integral of 2x from 0 to 3"))["answer"] == "9"

    # n05: system with 'and'
    sys_res = solve(extract_math("Solve x + y = 10 and x - y = 2"))
    assert sys_res["x"] == "6"
    assert sys_res["y"] == "4"

    # n06: greatest common divisor
    assert solve(extract_math("What is the greatest common divisor of 84 and 36?"))["answer"] == "12"

    # n07: factorize
    fac_res = solve(extract_math("Factorize 1001"))
    assert all(p in fac_res["answer"] for p in ("7", "11", "13"))

    # n08: matrix determinant
    assert solve(extract_math("What is the determinant of [[2, 1], [5, 3]]?"))["answer"] == "1"

    # n09: power
    assert solve(extract_math("What is 2 to the power of 10?"))["answer"] == "1024"

    # n10: square root
    assert solve(extract_math("What is the square root of 144?"))["answer"] == "12"

    # n11: word arithmetic
    assert solve(extract_math("Calculate 7 times 8 plus 6"))["answer"] == "62"

    # n12: least common multiple
    assert solve(extract_math("What is the least common multiple of 4, 6 and 10?"))["answer"] == "60"


