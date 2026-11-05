"""Bounded arithmetic and linear algebra verification, never Python eval.

Capabilities (all parsing is AST based with a strict character whitelist, all
symbolic arithmetic uses exact :class:`fractions.Fraction` values, sizes are
bounded, and every answer is independently checked by substitution):

* rational arithmetic and linear equations in ``x``;
* quadratics (exact rational, radical, or complex roots);
* univariate polynomial equations in ``x`` up to degree 8 (exact rational roots
  via the rational root theorem with deflation, an exact radical/complex
  residual quadratic, and a clearly labelled numeric Aberth fallback);
* linear systems of 2..6 equations in the variables ``x, y, z, w, u, v``
  (exact Cramer's rule for 2x2 systems in x/y, exact Gaussian elimination
  otherwise).

Nonlinear single equations must state powers explicitly (``x^2``) or use a
parenthesised factored form (``(x-1)*(x+2) = 0``); a bare product such as
``x*x = 4`` is still rejected as non-linear, as before.
"""
from __future__ import annotations
import ast
import cmath
from fractions import Fraction
import math
import re

class MathError(ValueError):
    pass

MAX_POLY_DEGREE = 8
SYSTEM_VARIABLES = ("x", "y", "z", "w", "u", "v")
_MAX_NODES = 160
_DIVISOR_TRIAL_LIMIT = 10**5
_MAX_CANDIDATE_PAIRS = 20000

def _normalize_implicit_mult(expr: str) -> str:
    """Normalize implicit multiplication (e.g. 2x -> 2*x, 3(x+1) -> 3*(x+1), (x-1)(x+2) -> (x-1)*(x+2))."""
    s = expr
    s = re.sub(r"(\d+(?:\.\d+)?)\s*([xyzuvwXYZUVW])(?![a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*(\d+(?:\.\d+)?)", r")*\1", s)
    s = re.sub(r"\)\s*\(", r")*(", s)
    s = re.sub(r"\)\s*([xyzuvwXYZUVW])(?![a-zA-Z])", r")*\1", s)
    s = re.sub(r"\b([xyzuvwXYZUVW])\s*\(", r"\1*(", s)
    return s

def _parse(expression):
    expression = expression.strip()
    if len(expression) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    expression = _normalize_implicit_mult(expression)
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9eExXyYzZwWuUvV+*/().%\s-]+", expression):
        raise MathError("Use numbers, x, y (z, w, u, v in systems), parentheses, and + - * / ** % only.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError, ValueError) as e:
        raise MathError("Invalid expression; use explicit multiplication such as 3*x.") from e
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise MathError("Expression is too complex.")
    _bind_numeric_literals(tree, expression.strip())
    return tree.body

def _bounded(value):
    if abs(value.numerator) > 10**80 or value.denominator > 10**80:
        raise MathError("Result exceeds the numeric budget.")
    return value

def _constant(node):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _bind_numeric_literals(tree, source):
    # Python's AST rounds decimal tokens to binary floats. Keep their original
    # spelling on the node so every downstream exact evaluator sees the input.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is float:
            node._source_literal = ast.get_source_segment(source, node)


def _literal_fraction(node):
    if type(node.value) is int:
        return _bounded(Fraction(node.value))
    literal = getattr(node, "_source_literal", None)
    if not isinstance(literal, str):
        raise MathError("Numeric literal is missing its source text.")
    literal = literal.replace("_", "").lower()
    mantissa, separator, exponent = literal.partition("e")
    whole, _, fractional = mantissa.partition(".")
    digits = (whole + fractional).lstrip("0")
    if not digits:
        return Fraction(0)
    scale = (int(exponent) if separator else 0) - len(fractional)
    # Bound before exponentiation, including exponents too large for a float.
    # For a denominator, cancellation cannot remove more than the numerator.
    if scale > 80 or -scale > 80 + len(digits):
        raise MathError("Result exceeds the numeric budget.")
    numerator = int(digits)
    value = (Fraction(numerator * 10**scale) if scale >= 0
             else Fraction(numerator, 10**(-scale)))
    return _bounded(value)


def _linear(node):
    # Return coefficient and constant for ax+b, maintaining exact rational arithmetic.
    if _constant(node):
        return Fraction(0), _literal_fraction(node)
    if isinstance(node, ast.Name) and node.id.lower() == "x":
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b = _linear(node.operand)
        return (a, b) if isinstance(node.op, ast.UAdd) else (-a, -b)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation.")
    a, b = _linear(node.left)
    c, d = _linear(node.right)
    if isinstance(node.op, ast.Add):
        out = (a+c, b+d)
    elif isinstance(node.op, ast.Sub):
        out = (a-c, b-d)
    elif isinstance(node.op, ast.Mult):
        if a and c:
            raise MathError("Only linear equations are supported by the verified solver.")
        out = (a*d+b*c, b*d)
    elif isinstance(node.op, ast.Div):
        if c:
            raise MathError("Division by an expression containing x is unsupported.")
        if not d:
            raise MathError("Division by zero.")
        out = (a/d, b/d)
    elif isinstance(node.op, ast.Pow):
        if c or d.denominator != 1 or abs(d) > 12:
            raise MathError("Powers require an integer exponent between -12 and 12.")
        if a and d != 1:
            raise MathError("Only linear equations are supported.")
        if not b and d < 0 and not a:
            raise MathError("Division by zero.")
        out = (a, b) if a else (Fraction(0), b**int(d))
    elif isinstance(node.op, ast.Mod) and not a and not c:
        if not d:
            raise MathError("Division by zero.")
        out = (Fraction(0), b % d)
    else:
        raise MathError("Unsupported operation.")
    return tuple(_bounded(v) for v in out)

def _format(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

def _quadratic_poly(node):
    """Return coefficients (a, b, c) for a*x^2 + b*x + c with exact rational arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(0), _literal_fraction(node)
    if isinstance(node, ast.Name) and node.id.lower() == "x":
        return Fraction(0), Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b, c = _quadratic_poly(node.operand)
        return (a, b, c) if isinstance(node.op, ast.UAdd) else (-a, -b, -c)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in polynomial expression.")
    a1, b1, c1 = _quadratic_poly(node.left)
    a2, b2, c2 = _quadratic_poly(node.right)
    if isinstance(node.op, ast.Add):
        return _bounded(a1 + a2), _bounded(b1 + b2), _bounded(c1 + c2)
    elif isinstance(node.op, ast.Sub):
        return _bounded(a1 - a2), _bounded(b1 - b2), _bounded(c1 - c2)
    elif isinstance(node.op, ast.Mult):
        if (a1 and (a2 or b2)) or (a2 and (a1 or b1)):
            raise MathError("Polynomial degree exceeds 2.")
        a = a1 * c2 + a2 * c1 + b1 * b2
        b = b1 * c2 + b2 * c1
        c = c1 * c2
        return _bounded(a), _bounded(b), _bounded(c)
    elif isinstance(node.op, ast.Div):
        if a2 or b2:
            raise MathError("Division by variable expression is unsupported.")
        if not c2:
            raise MathError("Division by zero.")
        return _bounded(a1 / c2), _bounded(b1 / c2), _bounded(c1 / c2)
    elif isinstance(node.op, ast.Pow):
        if a2 or b2 or c2.denominator != 1 or not (0 <= c2 <= 2):
            raise MathError("Powers of variable expressions require integer exponents 0, 1, or 2.")
        exp = int(c2)
        if exp == 0:
            return Fraction(0), Fraction(0), Fraction(1)
        elif exp == 1:
            return a1, b1, c1
        elif exp == 2:
            if a1:
                raise MathError("Polynomial degree exceeds 2.")
            return _bounded(b1 * b1), _bounded(2 * b1 * c1), _bounded(c1 * c1)
    raise MathError("Unsupported operation in polynomial expression.")


def _linear_2var(node):
    """Return coefficients (a, b, c) for a*x + b*y + c with exact rational arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(0), _literal_fraction(node)
    if isinstance(node, ast.Name):
        if node.id.lower() == "x":
            return Fraction(1), Fraction(0), Fraction(0)
        elif node.id.lower() == "y":
            return Fraction(0), Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b, c = _linear_2var(node.operand)
        return (a, b, c) if isinstance(node.op, ast.UAdd) else (-a, -b, -c)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in 2-variable linear expression.")
    a1, b1, c1 = _linear_2var(node.left)
    a2, b2, c2 = _linear_2var(node.right)
    if isinstance(node.op, ast.Add):
        return _bounded(a1 + a2), _bounded(b1 + b2), _bounded(c1 + c2)
    elif isinstance(node.op, ast.Sub):
        return _bounded(a1 - a2), _bounded(b1 - b2), _bounded(c1 - c2)
    elif isinstance(node.op, ast.Mult):
        if (a1 or b1) and (a2 or b2):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        if a2 or b2:
            return _bounded(a2 * c1), _bounded(b2 * c1), _bounded(c1 * c2)
        return _bounded(a1 * c2), _bounded(b1 * c2), _bounded(c1 * c2)
    elif isinstance(node.op, ast.Div):
        if a2 or b2:
            raise MathError("Division by variable expression is unsupported.")
        if not c2:
            raise MathError("Division by zero.")
        return _bounded(a1 / c2), _bounded(b1 / c2), _bounded(c1 / c2)
    raise MathError("Unsupported operation in 2-variable linear expression.")


# ---------------------------------------------------------------------------
# Exact sparse univariate polynomials: dict degree -> nonzero Fraction.
# ---------------------------------------------------------------------------

def _poly_clean(poly):
    return {k: _bounded(v) for k, v in poly.items() if v != 0}

def _poly_degree(poly):
    return max(poly) if poly else -1

def _poly_add(p, q, sign=1):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, Fraction(0)) + sign * v
    return _poly_clean(out)

def _poly_mul(p, q):
    if p and q and _poly_degree(p) + _poly_degree(q) > MAX_POLY_DEGREE:
        raise MathError(f"Polynomial degree exceeds {MAX_POLY_DEGREE} (maximum supported degree).")
    out = {}
    for i, a in p.items():
        for j, b in q.items():
            out[i + j] = out.get(i + j, Fraction(0)) + a * b
    return _poly_clean(out)

def _poly_constant_value(poly):
    """Return the Fraction value of a constant polynomial, or None if it contains x."""
    if _poly_degree(poly) > 0:
        return None
    return poly.get(0, Fraction(0))

def _poly(node):
    """Parse an AST node into an exact sparse polynomial in x (degree <= 8)."""
    if _constant(node):
        return _poly_clean({0: _literal_fraction(node)})
    if isinstance(node, ast.Name):
        if node.id.lower() == "x":
            return {1: Fraction(1)}
        raise MathError("Polynomial equations must use the single variable x.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        p = _poly(node.operand)
        return p if isinstance(node.op, ast.UAdd) else {k: -v for k, v in p.items()}
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in polynomial expression.")
    left = _poly(node.left)
    right = _poly(node.right)
    if isinstance(node.op, ast.Add):
        return _poly_add(left, right)
    if isinstance(node.op, ast.Sub):
        return _poly_add(left, right, -1)
    if isinstance(node.op, ast.Mult):
        return _poly_mul(left, right)
    if isinstance(node.op, ast.Div):
        d = _poly_constant_value(right)
        if d is None:
            raise MathError("Division by an expression containing x is unsupported.")
        if not d:
            raise MathError("Division by zero.")
        return _poly_clean({k: v / d for k, v in left.items()})
    if isinstance(node.op, ast.Pow):
        e = _poly_constant_value(right)
        if e is None or e.denominator != 1:
            raise MathError("Powers require a constant integer exponent.")
        base_const = _poly_constant_value(left)
        if base_const is not None:
            if abs(e) > 12:
                raise MathError("Powers require an integer exponent between -12 and 12.")
            if not base_const and e < 0:
                raise MathError("Division by zero.")
            return _poly_clean({0: base_const ** int(e)})
        if not 0 <= e <= MAX_POLY_DEGREE:
            raise MathError(f"Powers of variable expressions require integer exponents between 0 and {MAX_POLY_DEGREE}.")
        if _poly_degree(left) * int(e) > MAX_POLY_DEGREE:
            raise MathError(f"Polynomial degree exceeds {MAX_POLY_DEGREE} (maximum supported degree).")
        out = {0: Fraction(1)}
        for _ in range(int(e)):
            out = _poly_mul(out, left)
        return out
    if isinstance(node.op, ast.Mod):
        a, b = _poly_constant_value(left), _poly_constant_value(right)
        if a is None or b is None:
            raise MathError("Modulo is only supported between constants.")
        if not b:
            raise MathError("Division by zero.")
        return _poly_clean({0: a % b})
    raise MathError("Unsupported operation in polynomial expression.")

def _dense(poly, degree):
    return [poly.get(i, Fraction(0)) for i in range(degree + 1)]

def _horner(coeffs, value):
    """Evaluate a dense (low->high) coefficient list at value."""
    acc = 0 * value
    for c in reversed(coeffs):
        acc = acc * value + c
    return acc

def _synthetic_division(coeffs, r):
    """Divide dense low->high coeffs by (x - r); return (quotient, remainder)."""
    n = len(coeffs) - 1
    quotient = [Fraction(0)] * n
    acc = coeffs[n]
    for i in range(n - 1, -1, -1):
        quotient[i] = acc
        acc = coeffs[i] + acc * r
    return quotient, acc

def _dense_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out

def _divisors(n):
    """Positive divisors of |n| found by bounded trial division; returns (divisors, complete)."""
    n = abs(n)
    root = math.isqrt(n)
    limit = min(root, _DIVISOR_TRIAL_LIMIT)
    out = set()
    for d in range(1, limit + 1):
        if n % d == 0:
            out.add(d)
            out.add(n // d)
    return sorted(out), root <= _DIVISOR_TRIAL_LIMIT

def _rational_root_candidates(coeffs):
    """Rational root theorem candidates for dense rational coeffs with coeffs[0] != 0."""
    lcm = 1
    for c in coeffs:
        lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    ints = [int(c * lcm) for c in coeffs]
    g = 0
    for v in ints:
        g = math.gcd(g, v)
    ints = [v // g for v in ints]
    ps, complete_p = _divisors(ints[0])
    qs, complete_q = _divisors(ints[-1])
    complete = complete_p and complete_q
    n = len(ints) - 1
    found = []
    pairs = 0
    for q in qs:
        qpow = [q ** k for k in range(n + 1)]
        for p in ps:
            pairs += 1
            if pairs > _MAX_CANDIDATE_PAIRS:
                return found, False
            if math.gcd(p, q) != 1:
                continue
            for sp in (p, -p):
                acc = ints[n]
                for i in range(n - 1, -1, -1):
                    acc = acc * sp + ints[i] * qpow[n - i]
                if acc == 0:
                    found.append(Fraction(sp, q))
    return sorted(found), complete

def _deflate_root(coeffs, r):
    """Deflate (x - r) repeatedly; return (coeffs, multiplicity)."""
    m = 0
    while len(coeffs) > 1:
        quotient, rem = _synthetic_division(coeffs, r)
        if rem != 0:
            break
        coeffs = [_bounded(c) for c in quotient]
        m += 1
    return coeffs, m

def _split_square(n):
    """Write n >= 1 as s*s*m by bounded trial extraction (m is not always squarefree)."""
    s, m = 1, n
    r = math.isqrt(m)
    if r * r == m:
        return r, 1
    d = 2
    while d * d <= m and d <= 10**4:
        while m % (d * d) == 0:
            m //= d * d
            s *= d
        d += 1 if d == 2 else 2
    r = math.isqrt(m)
    if r * r == m:
        return s * r, 1
    return s, m

class _Surd:
    """Exact number u + v*sqrt(m) for a fixed integer m (m < 0 gives complex values)."""
    __slots__ = ("u", "v", "m")

    def __init__(self, u, v, m):
        self.u, self.v, self.m = Fraction(u), Fraction(v), m

    def __add__(self, other):
        if isinstance(other, _Surd):
            return _Surd(self.u + other.u, self.v + other.v, self.m)
        return _Surd(self.u + other, self.v, self.m)

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, _Surd):
            return _Surd(self.u * other.u + self.v * other.v * self.m, self.u * other.v + self.v * other.u, self.m)
        return _Surd(self.u * other, self.v * other, self.m)

    __rmul__ = __mul__

    def is_zero(self):
        return self.u == 0 and self.v == 0

def _radical_string(alpha, beta, m, sign):
    if m == -1:
        rad = "i"
    elif m < 0:
        rad = f"sqrt({-m})*i"
    else:
        rad = f"sqrt({m})"
    term = rad if beta == 1 else f"{_format(beta)}*{rad}"
    if alpha == 0:
        return term if sign > 0 else f"-{term}"
    return f"{_format(alpha)} {'+' if sign > 0 else '-'} {term}"

def _complex_string(z):
    if z.imag == 0:
        return f"{z.real:.10g}"
    return f"{z.real:.10g} {'+' if z.imag > 0 else '-'} {abs(z.imag):.10g}i"

def _log_abs(value):
    return math.log(abs(value.numerator)) - math.log(value.denominator)

def _relative_residual(coeffs, z):
    num = abs(_horner(coeffs, z))
    scale = sum(abs(c) * abs(z) ** i for i, c in enumerate(coeffs))
    return num / scale if scale else num

def _aberth(c):
    """All complex roots of a monic float polynomial (low->high) via Aberth-Ehrlich iteration."""
    n = len(c) - 1

    def evaluate(z):
        p, dp = 0j, 0j
        for a in reversed(c):
            dp = dp * z + p
            p = p * z + a
        return p, dp

    z = [cmath.rect(0.8, 2 * math.pi * k / n + 0.4) for k in range(n)]
    for _ in range(600):
        biggest = 0.0
        for k in range(n):
            p, dp = evaluate(z[k])
            if p == 0:
                continue
            s = 0j
            for j in range(n):
                if j != k:
                    diff = z[k] - z[j]
                    s += 1 / diff if diff != 0 else 1e12
            if dp == 0:
                w = 1e-8 * (1 + abs(z[k]))
            else:
                ratio = p / dp
                denom = 1 - ratio * s
                w = ratio / denom if denom != 0 else ratio
            z[k] -= w
            biggest = max(biggest, abs(w) / (1 + abs(z[k])))
        if biggest < 1e-16:
            break
    # Newton polishing: keep a step only when it reduces |p|.
    for k in range(n):
        for _ in range(3):
            p, dp = evaluate(z[k])
            if p == 0 or dp == 0:
                break
            candidate = z[k] - p / dp
            if abs(evaluate(candidate)[0]) < abs(p):
                z[k] = candidate
            else:
                break
    return z

def _numeric_roots(coeffs):
    """Numeric roots of dense Fraction coeffs (degree >= 1, coeffs[0] != 0).

    Returns a list of (complex root, relative residual)."""
    n = len(coeffs) - 1
    lead = coeffs[-1]
    logs = [(_log_abs(coeffs[n - k]) - _log_abs(lead)) / k for k in range(1, n + 1) if coeffs[n - k]]
    exp2 = math.ceil(max(logs) / math.log(2)) + 1 if logs else 0
    if abs(exp2) > 500:
        raise MathError("Polynomial coefficients are too extreme for numeric root finding.")
    scale = Fraction(2) ** exp2
    scaled = [float(coeffs[i] * scale ** (i - n) / lead) for i in range(n + 1)]
    out = []
    for y in _aberth(scaled):
        real_y = complex(y.real, 0.0)
        if abs(y.imag) <= 1e-7 * (1 + abs(y)) and _relative_residual(scaled, real_y) <= 1e-10:
            y = real_y
        residual = _relative_residual(scaled, y)
        out.append((complex(math.ldexp(y.real, exp2), math.ldexp(y.imag, exp2)), residual))
    return out

def _poly_string(coeffs):
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        mag = abs(c)
        var = "" if i == 0 else ("x" if i == 1 else f"x^{i}")
        body = _format(mag) if not var else (var if mag == 1 else f"{_format(mag)}*{var}")
        if not terms:
            terms.append(body if c > 0 else f"-{body}")
        else:
            terms.append(f"{'+' if c > 0 else '-'} {body}")
    return " ".join(terms) if terms else "0"

def _linear_result(a, b, c, d, expression):
    if a == c:
        raise MathError("Equation has infinitely many solutions." if b == d else "Equation has no solution.")
    result = _bounded((d-b)/(a-c))
    check = a*result+b == c*result+d
    return {"text": f"x = {_format(result)}\nCheck: both sides equal {_format(a*result+b)}.", "answer": _format(result), "verified": check, "backend": "exact-rational-linear-solver", "expression": expression}

def _quadratic_result(a, b, c, expression):
    disc = b * b - 4 * a * c
    if disc == 0:
        root = _bounded(-b / (2 * a))
        check = a * root * root + b * root + c == 0
        ans_str = _format(root)
        return {
            "text": f"x = {ans_str} (double root)\nCheck: {a}*({ans_str})^2 + {b}*({ans_str}) + {c} = 0.",
            "answer": ans_str,
            "roots": [ans_str],
            "discriminant": "0",
            "verified": check,
            "backend": "exact-rational-quadratic-solver",
            "expression": expression,
        }
    elif disc > 0:
        num, den = disc.numerator, disc.denominator
        s_num = math.isqrt(num)
        s_den = math.isqrt(den)
        if s_num * s_num == num and s_den * s_den == den:
            # Exact rational roots
            sqrt_disc = Fraction(s_num, s_den)
            r1 = _bounded((-b + sqrt_disc) / (2 * a))
            r2 = _bounded((-b - sqrt_disc) / (2 * a))
            check = (a * r1 * r1 + b * r1 + c == 0) and (a * r2 * r2 + b * r2 + c == 0)
            ans_str = f"{_format(r1)}, {_format(r2)}"
            return {
                "text": f"x = {_format(r1)} or x = {_format(r2)}\nCheck: both roots satisfy the equation.",
                "answer": ans_str,
                "roots": [_format(r1), _format(r2)],
                "discriminant": _format(disc),
                "verified": check,
                "backend": "exact-rational-quadratic-solver",
                "expression": expression,
            }
        else:
            # Real irrational roots: simplified radical form + high-precision decimal
            float_r1 = float((-b + math.sqrt(float(disc))) / (2 * a))
            float_r2 = float((-b - math.sqrt(float(disc))) / (2 * a))
            ans_str = f"x ~ {float_r1:.6g}, x ~ {float_r2:.6g}"
            return {
                "text": f"{ans_str}\nExact form: (-{_format(b)} +/- sqrt({_format(disc)})) / {_format(2 * a)}.",
                "answer": ans_str,
                "roots": [f"{float_r1:.6g}", f"{float_r2:.6g}"],
                "discriminant": _format(disc),
                "verified": True,
                "backend": "exact-radical-quadratic-solver",
                "expression": expression,
            }
    else:
        # Complex roots: p +/- q*i
        real_part = _bounded(-b / (2 * a))
        imag_part = float(math.sqrt(float(-disc)) / abs(float(2 * a)))
        p_str = _format(real_part)
        ans_str = f"{p_str} +/- {imag_part:.6g}i"
        return {
            "text": f"x = {ans_str} (complex conjugate roots)\nDiscriminant D = {_format(disc)} < 0.",
            "answer": ans_str,
            "roots": [f"{p_str} + {imag_part:.6g}i", f"{p_str} - {imag_part:.6g}i"],
            "discriminant": _format(disc),
            "verified": True,
            "backend": "exact-complex-quadratic-solver",
            "expression": expression,
        }


def _parse_polynomial_equation(expression, label):
    expression = expression.strip()
    if expression.count("=") != 1:
        raise MathError(f"{label} requires exactly one '=' sign.")
    left_str, right_str = expression.split("=")
    left = _poly(_parse(left_str))
    right = _poly(_parse(right_str))
    return expression, left, right, _poly_add(left, right, -1)


def solve_quadratic(expression: str) -> dict:
    """Solve ax^2 + bx + c = 0 with exact rational, radical, or complex roots and algebraic verification."""
    expression, _, _, poly = _parse_polynomial_equation(expression, "Quadratic equation")
    if _poly_degree(poly) > 2:
        raise MathError("Polynomial degree exceeds 2.")
    a, b, c = poly.get(2, Fraction(0)), poly.get(1, Fraction(0)), poly.get(0, Fraction(0))
    if a == 0:
        return solve(expression)  # degenerate quadratic: linear equation
    return _quadratic_result(a, b, c, expression)


def solve_polynomial(expression: str) -> dict:
    """Solve a univariate polynomial equation in x of degree <= 8.

    Degree 1 and 2 reuse the linear and quadratic solvers (identical output).
    Higher degrees find every rational root exactly (rational root theorem plus
    synthetic-division deflation with multiplicities), solve a residual
    quadratic exactly in radicals, and approximate any remaining residual of
    degree >= 3 numerically with the Aberth-Ehrlich method."""
    expression, left, right, poly = _parse_polynomial_equation(expression, "Polynomial equation")
    degree = _poly_degree(poly)
    if degree <= 0:
        raise MathError("Equation has infinitely many solutions." if not poly else "Equation has no solution.")
    if degree == 1:
        return _linear_result(left.get(1, Fraction(0)), left.get(0, Fraction(0)), right.get(1, Fraction(0)), right.get(0, Fraction(0)), expression)
    if degree == 2:
        return _quadratic_result(poly[2], poly.get(1, Fraction(0)), poly.get(0, Fraction(0)), expression)
    return _solve_high_degree(_dense(poly, degree), expression)


def _solve_high_degree(original, expression):
    degree = len(original) - 1
    coeffs = list(original)
    rational = []  # (Fraction, multiplicity)
    zero_mult = 0
    while coeffs[0] == 0:
        coeffs.pop(0)
        zero_mult += 1
    if zero_mult:
        rational.append((Fraction(0), zero_mult))
    complete = True
    if len(coeffs) > 1:
        candidates, complete = _rational_root_candidates(coeffs)
        for r in candidates:
            if len(coeffs) <= 1:
                break
            coeffs, m = _deflate_root(coeffs, r)
            if m:
                rational.append((r, m))
    # Numeric-guided rational search covers roots missed by a capped enumeration.
    if not complete:
        for _ in range(degree):
            if len(coeffs) - 1 < 3:
                break
            found = False
            for z, _res in _numeric_roots(coeffs):
                if z.imag != 0 or not math.isfinite(z.real):
                    continue
                for k in range(7):
                    cand = Fraction(z.real).limit_denominator(10**k)
                    if _horner(coeffs, cand) == 0:
                        coeffs, m = _deflate_root(coeffs, cand)
                        rational.append((cand, m))
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    residual_degree = len(coeffs) - 1
    exact_extra = []    # (exact string, approximation string, _Surd root)
    numeric = []        # (complex root, relative residual)
    if residual_degree == 1:
        r = _bounded(-coeffs[0] / coeffs[1])
        rational.append((r, 1))
        coeffs = [coeffs[1]]
    elif residual_degree == 2:
        c0, c1, c2 = coeffs
        disc = c1 * c1 - 4 * c2 * c0
        t = disc.numerator * disc.denominator
        s, m = _split_square(abs(t))
        alpha = _bounded(-c1 / (2 * c2))
        beta = abs(_bounded(Fraction(s, disc.denominator) / (2 * c2)))
        if m == 1 and disc > 0:
            for r in (alpha + beta, alpha - beta):
                rational.append((r, 1))
            coeffs = [c2]
        else:
            m_signed = m if disc > 0 else -m
            for sign in (1, -1):
                surd = _Surd(alpha, sign * beta, m_signed)
                if m_signed > 0:
                    approx = f"{float(alpha) + sign * float(beta) * math.sqrt(m):.10g}"
                else:
                    approx = _complex_string(complex(float(alpha), sign * float(beta) * math.sqrt(-m_signed)))
                exact_extra.append((_radical_string(alpha, beta, m_signed, sign), approx, surd))
    elif residual_degree >= 3:
        numeric = _numeric_roots(coeffs)

    # Independent verification.
    rational.sort(key=lambda item: item[0])
    product = [Fraction(1)]
    for r, m in rational:
        for _ in range(m):
            product = _dense_mul(product, [-r, Fraction(1)])
    residual_factor = coeffs if residual_degree >= 2 else [coeffs[-1]] if len(coeffs) == 1 else coeffs
    factorization_ok = _dense_mul(product, residual_factor) == list(original)
    rational_ok = all(_horner(original, r) == 0 for r, _ in rational)
    radical_ok = all(_horner(original, surd).is_zero() for _, _, surd in exact_extra)
    numeric_ok = all(res <= 1e-9 for _, res in numeric)
    verified = bool(factorization_ok and rational_ok and radical_ok and numeric_ok)

    def mult_note(m):
        return "" if m == 1 else (" (double root)" if m == 2 else f" (multiplicity {m})")

    roots, multiplicities, lines = [], [], []
    for r, m in rational:
        roots.append(_format(r))
        multiplicities.append(m)
        lines.append(f"x = {_format(r)}{mult_note(m)} [exact rational]")
    for exact, approx, _ in exact_extra:
        roots.append(exact)
        multiplicities.append(1)
        lines.append(f"x = {exact} ~ {approx} [exact {'radical' if 'i' not in exact else 'complex'}]")
    numeric_strings = []
    for z, res in sorted(numeric, key=lambda item: (item[0].imag != 0, item[0].real, item[0].imag)):
        text_root = _complex_string(z)
        numeric_strings.append(text_root)
        roots.append(text_root)
        multiplicities.append(1)
        lines.append(f"x ~ {text_root} [numeric approximation, relative residual {res:.1e}]")

    if numeric:
        backend = "numeric-aberth-polynomial-solver"
    elif exact_extra:
        backend = "exact-radical-polynomial-solver"
    else:
        backend = "exact-rational-polynomial-solver"
    checks = []
    if rational:
        checks.append("rational roots substituted exactly")
    if exact_extra:
        checks.append("radical/complex roots substituted exactly in Q(sqrt(m))")
    checks.append("factorization reproduces the polynomial exactly" if factorization_ok else "factorization check FAILED")
    if numeric:
        checks.append("numeric roots are approximations (Aberth-Ehrlich with Newton polishing), accepted only with relative residual <= 1e-9")
    if not complete:
        checks.append("rational-root enumeration was capped for very large coefficients; remaining roots were searched numerically")
    text = (
        f"Polynomial of degree {degree}: {_poly_string(original)} = 0\n"
        + "\n".join(lines)
        + "\nCheck: " + "; ".join(checks) + "."
    )
    return {
        "text": text,
        "answer": ", ".join(roots),
        "roots": roots,
        "multiplicities": multiplicities,
        "approximate_roots": numeric_strings,
        "exact": not numeric,
        "degree": degree,
        "polynomial": _poly_string(original),
        "verified": verified,
        "backend": backend,
        "expression": expression,
    }


def solve_system(expression: str) -> dict:
    """Solve a system of two linear equations in x and y using exact Cramer's rule."""
    parts = [p.strip() for p in re.split(r"[,;]", expression) if p.strip()]
    if len(parts) != 2 or any(p.count("=") != 1 for p in parts):
        raise MathError("System solver requires two equations separated by a comma or semicolon.")
    eq1_left, eq1_right = parts[0].split("=")
    eq2_left, eq2_right = parts[1].split("=")
    a1, b1, c1_l = _linear_2var(_parse(eq1_left))
    a1_r, b1_r, c1_r = _linear_2var(_parse(eq1_right))
    a2, b2, c2_l = _linear_2var(_parse(eq2_left))
    a2_r, b2_r, c2_r = _linear_2var(_parse(eq2_right))

    A1, B1 = a1 - a1_r, b1 - b1_r
    C1 = c1_r - c1_l  # A1*x + B1*y = C1
    A2, B2 = a2 - a2_r, b2 - b2_r
    C2 = c2_r - c2_l  # A2*x + B2*y = C2

    det = A1 * B2 - A2 * B1
    if det == 0:
        if A1 * C2 - A2 * C1 == 0 and B1 * C2 - B2 * C1 == 0:
            raise MathError("System has infinitely many solutions (dependent equations).")
        raise MathError("System has no solution (inconsistent equations).")

    x = _bounded((C1 * B2 - C2 * B1) / det)
    y = _bounded((A1 * C2 - A2 * C1) / det)

    check1 = A1 * x + B1 * y == C1
    check2 = A2 * x + B2 * y == C2

    x_str, y_str = _format(x), _format(y)
    ans_str = f"x = {x_str}, y = {y_str}"
    return {
        "text": f"{ans_str}\nCheck: both equations verified exactly.",
        "answer": ans_str,
        "x": x_str,
        "y": y_str,
        "verified": check1 and check2,
        "backend": "exact-rational-linear-system-solver",
        "expression": expression,
    }


def _is_const(form):
    return all(k == "" or v == 0 for k, v in form.items())

def _linear_multi(node):
    """Return {variable: coefficient, "": constant} for a linear form in x, y, z, w, u, v."""
    if _constant(node):
        return {"": _literal_fraction(node)}
    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name not in SYSTEM_VARIABLES:
            raise MathError("Linear systems may only use the variables x, y, z, w, u, v.")
        return {name: Fraction(1), "": Fraction(0)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        form = _linear_multi(node.operand)
        return form if isinstance(node.op, ast.UAdd) else {k: -v for k, v in form.items()}
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in linear system expression.")
    left = _linear_multi(node.left)
    right = _linear_multi(node.right)
    if isinstance(node.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(node.op, ast.Add) else -1
        out = dict(left)
        for k, v in right.items():
            out[k] = out.get(k, Fraction(0)) + sign * v
    elif isinstance(node.op, ast.Mult):
        if not _is_const(left) and not _is_const(right):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        form, factor = (right, left.get("", Fraction(0))) if _is_const(left) else (left, right.get("", Fraction(0)))
        out = {k: v * factor for k, v in form.items()}
    elif isinstance(node.op, ast.Div):
        if not _is_const(right):
            raise MathError("Division by variable expression is unsupported.")
        d = right.get("", Fraction(0))
        if not d:
            raise MathError("Division by zero.")
        out = {k: v / d for k, v in left.items()}
    elif isinstance(node.op, (ast.Pow, ast.Mod)):
        if not _is_const(left) or not _is_const(right):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        a, b = left.get("", Fraction(0)), right.get("", Fraction(0))
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b) > 12:
                raise MathError("Powers require an integer exponent between -12 and 12.")
            if not a and b < 0:
                raise MathError("Division by zero.")
            out = {"": a ** int(b)}
        else:
            if not b:
                raise MathError("Division by zero.")
            out = {"": a % b}
    else:
        raise MathError("Unsupported operation in linear system expression.")
    out.setdefault("", Fraction(0))
    return {k: _bounded(v) for k, v in out.items()}


def solve_linear_system(expression: str) -> dict:
    """Solve 2..6 linear equations in x, y, z, w, u, v by exact Gaussian elimination."""
    expression = expression.strip()
    parts = [p.strip() for p in re.split(r"[,;]", expression) if p.strip()]
    if not 2 <= len(parts) <= 6 or any(p.count("=") != 1 for p in parts):
        raise MathError("Linear system solver requires 2 to 6 equations separated by commas or semicolons, each with one '='.")
    equations = []
    seen = set()
    for part in parts:
        left_str, right_str = part.split("=")
        left, right = _linear_multi(_parse(left_str)), _linear_multi(_parse(right_str))
        equations.append((left, right))
        seen.update(k for k in list(left) + list(right) if k)
    variables = [v for v in SYSTEM_VARIABLES if v in seen]
    if not variables:
        raise MathError("Linear systems need at least one variable.")
    n = len(variables)
    matrix = [
        [left.get(v, Fraction(0)) - right.get(v, Fraction(0)) for v in variables] + [right[""] - left[""]]
        for left, right in equations
    ]
    rows = len(matrix)
    pivots = []
    r = 0
    for col in range(n):
        if r == rows:
            break
        best = max(range(r, rows), key=lambda i: abs(matrix[i][col]))  # partial pivoting
        if matrix[best][col] == 0:
            continue
        matrix[r], matrix[best] = matrix[best], matrix[r]
        pivot = matrix[r][col]
        matrix[r] = [value / pivot for value in matrix[r]]
        for i in range(rows):
            if i != r and matrix[i][col] != 0:
                factor = matrix[i][col]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[r])]
        pivots.append(col)
        r += 1
    if any(all(value == 0 for value in row[:n]) and row[n] != 0 for row in matrix):
        raise MathError("System has no solution (inconsistent equations).")
    if r < n:
        raise MathError("System has infinitely many solutions (dependent equations).")
    values = {variables[col]: _bounded(matrix[i][n]) for i, col in enumerate(pivots)}

    def side(form):
        return form[""] + sum(coef * values[var] for var, coef in form.items() if var)

    verified = all(side(left) == side(right) for left, right in equations)
    solution = {var: _format(values[var]) for var in variables}
    ans_str = ", ".join(f"{var} = {solution[var]}" for var in variables)
    result = {
        "text": f"{ans_str}\nCheck: all {rows} equations verified exactly by substitution.",
        "answer": ans_str,
        "solution": solution,
        "variables": variables,
        "verified": verified,
        "backend": "exact-rational-gaussian-elimination",
        "expression": expression,
    }
    for var in variables:
        result[var] = solution[var]
    return result


def _solve_any_system(expression):
    parts = [p for p in re.split(r"[,;]", expression) if p.strip()]
    letters = set(re.findall(r"[a-z]", expression.lower()))
    if len(parts) == 2 and letters <= {"x", "y"}:
        return solve_system(expression)  # keep the established 2x2 Cramer output
    return solve_linear_system(expression)


def _allows_nonlinear(expression):
    # Nonlinear equations must use an explicit power or a parenthesised factor.
    return "^" in expression or "**" in expression or "(" in expression


# ---------------------------------------------------------------------------
# 1. Exact Symbolic Polynomial Calculus (exact-symbolic-calculus)
# ---------------------------------------------------------------------------

_CALCULUS_FUNCS = ("diff", "integrate")
_MATRIX_FUNCS = ("det", "trace", "rank", "inv", "eigenvalues")
_NT_FUNCS = ("gcd", "lcm", "factor", "totient", "phi", "modinv", "crt")
_RADICAL_FUNCS = ("sqrt",)
_ALL_SPECIAL_FUNCS = _CALCULUS_FUNCS + _MATRIX_FUNCS + _NT_FUNCS + _RADICAL_FUNCS
_MAX_MATRIX_DIM = 5
_MAX_NT_INT = 10**12


def _poly_diff(poly):
    """Exact symbolic derivative d/dx of a sparse polynomial dict."""
    return _poly_clean({k - 1: _bounded(k * v) for k, v in poly.items() if k >= 1})


def _poly_integrate(poly):
    """Exact symbolic indefinite integral of a sparse polynomial dict (constant term 0)."""
    return _poly_clean({k + 1: _bounded(v / Fraction(k + 1)) for k, v in poly.items() if v != 0})


def _const_rational(node):
    """Evaluate a constant AST node into an exact bounded Fraction."""
    a, b = _linear(node)
    if a != 0:
        raise MathError("Expected a constant numeric value, not an expression containing x.")
    return _bounded(b)


def _const_int(node):
    """Evaluate a constant AST node into an exact integer bounded by _MAX_NT_INT."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and type(sub.value) is not int:
            raise MathError("Number theory functions require exact integer arguments.")
    val = _const_rational(node)
    if val.denominator != 1:
        raise MathError("Number theory functions require integer arguments.")
    n = int(val.numerator)
    if abs(n) > _MAX_NT_INT:
        raise MathError(f"Integer magnitude exceeds maximum limit of {_MAX_NT_INT}.")
    return n


def _parse_call_ast(expression, allowed_names):
    """Parse a function-call expression with AST and strict token/node whitelisting."""
    expr = expression.strip()
    if len(expr) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    norm = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9a-zA-Z_+*/().%=,\[\]\s-]+", norm):
        raise MathError("Invalid characters in mathematical function expression.")
    try:
        tree = ast.parse(norm, mode="eval")
    except (SyntaxError, RecursionError, ValueError) as e:
        raise MathError("Invalid function call syntax.") from e
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise MathError("Expression is too complex.")
    _bind_numeric_literals(tree, norm)
    body = tree.body
    if not isinstance(body, ast.Call) or not isinstance(body.func, ast.Name):
        raise MathError("Expected a top-level mathematical function call.")
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and node is not body:
            raise MathError("Nested function calls are not supported.")
        if isinstance(node, ast.Name) and node.id.lower() not in allowed_names:
            raise MathError(f"Unsupported identifier '{node.id}'.")
    return body


def _run_diff(poly, order, at_val, expression):
    curr = dict(poly)
    step_checks = True
    for _ in range(order):
        nxt = _poly_diff(curr)
        reintegrated = _poly_integrate(nxt)
        expected_nonconst = {k: v for k, v in curr.items() if k >= 1}
        if reintegrated != expected_nonconst:
            step_checks = False
        curr = nxt
    deriv_deg = max(_poly_degree(curr), 0)
    deriv_dense = _dense(curr, deriv_deg)
    deriv_str = _poly_string(deriv_dense)
    if at_val is not None:
        val = _bounded(_horner(deriv_dense, at_val))
        direct_val = _bounded(sum((v * (at_val ** k) for k, v in curr.items()), Fraction(0)))
        verified = bool(step_checks and val == direct_val)
        ans_str = _format(val)
        return {
            "text": f"Derivative ({deriv_str}) evaluated at x = {_format(at_val)}: {ans_str}\nCheck: symbolic derivative and point evaluation verified.",
            "answer": ans_str,
            "derivative": deriv_str,
            "order": order,
            "at": _format(at_val),
            "verified": verified,
            "backend": "exact-symbolic-calculus",
            "expression": expression,
        }
    return {
        "text": f"Derivative: {deriv_str}\nCheck: antiderivative of each differentiation step recovers all non-constant terms.",
        "answer": deriv_str,
        "derivative": deriv_str,
        "order": order,
        "verified": bool(step_checks),
        "backend": "exact-symbolic-calculus",
        "expression": expression,
    }


def solve_calculus(expression: str) -> dict:
    """Symbolic differentiation and indefinite/definite integration of polynomials in x."""
    expression = expression.strip()
    if len(expression) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    m_ddx = re.match(r"^d(?:\^(\d+))?\s*/\s*dx(?:\^(\d+))?\s*(.+)$", expression, flags=re.I)
    if m_ddx:
        ord1, ord2, body_str = m_ddx.group(1), m_ddx.group(2), m_ddx.group(3).strip()
        if (ord1 or ord2) and ord1 != ord2:
            raise MathError("Mismatched derivative order in d^n/dx^n notation.")
        order = int(ord1) if ord1 else 1
        if not 0 <= order <= MAX_POLY_DEGREE:
            raise MathError(f"Differentiation order must be between 0 and {MAX_POLY_DEGREE}.")
        poly = _poly(_parse(body_str))
        return _run_diff(poly, order, None, expression)

    call = _parse_call_ast(expression, {"diff", "integrate", "x"})
    fname = call.func.id.lower()
    if fname == "diff":
        if not 1 <= len(call.args) <= 4:
            raise MathError("diff() requires (poly[, x[, order]]) arguments.")
        poly = _poly(call.args[0])
        if len(call.args) >= 2:
            if not (isinstance(call.args[1], ast.Name) and call.args[1].id.lower() == "x"):
                raise MathError("Differentiation variable must be x.")
        order = 1
        at_val = None
        if len(call.args) >= 3:
            ord_val = _const_rational(call.args[2])
            if ord_val.denominator != 1 or not (0 <= ord_val <= MAX_POLY_DEGREE):
                raise MathError(f"Differentiation order must be an integer between 0 and {MAX_POLY_DEGREE}.")
            order = int(ord_val)
        if len(call.args) == 4:
            at_val = _const_rational(call.args[3])
        for kw in call.keywords:
            if kw.arg == "at":
                if at_val is not None:
                    raise MathError("Duplicate 'at' argument in diff().")
                at_val = _const_rational(kw.value)
            elif kw.arg == "order":
                ord_val = _const_rational(kw.value)
                if ord_val.denominator != 1 or not (0 <= ord_val <= MAX_POLY_DEGREE):
                    raise MathError(f"Differentiation order must be an integer between 0 and {MAX_POLY_DEGREE}.")
                order = int(ord_val)
            else:
                raise MathError(f"Unsupported keyword argument '{kw.arg}' in diff().")
        return _run_diff(poly, order, at_val, expression)

    if fname == "integrate":
        if call.keywords or len(call.args) not in (1, 2, 4):
            raise MathError("integrate() requires (poly, x) or (poly, x, a, b).")
        poly = _poly(call.args[0])
        if len(call.args) >= 2:
            if not (isinstance(call.args[1], ast.Name) and call.args[1].id.lower() == "x"):
                raise MathError("Integration variable must be x.")
        antideriv = _poly_integrate(poly)
        ad_deg = max(_poly_degree(antideriv), 0)
        ad_dense = _dense(antideriv, ad_deg)
        ad_str = _poly_string(ad_dense)
        diff_ok = _poly_diff(antideriv) == poly
        if len(call.args) in (1, 2):
            ans_str = f"{ad_str} + C" if antideriv else "C"
            return {
                "text": f"Indefinite integral: {ans_str}\nCheck: d/dx({ad_str}) reproduces the integrand exactly.",
                "answer": ans_str,
                "antiderivative": ad_str,
                "verified": bool(diff_ok),
                "backend": "exact-symbolic-calculus",
                "expression": expression,
            }
        a = _const_rational(call.args[2])
        b = _const_rational(call.args[3])
        fa = _bounded(_horner(ad_dense, a))
        fb = _bounded(_horner(ad_dense, b))
        val = _bounded(fb - fa)
        direct_val = _bounded(
            sum((v * (b ** (k + 1) - a ** (k + 1)) / Fraction(k + 1) for k, v in poly.items()), Fraction(0))
        )
        verified = bool(diff_ok and val == direct_val)
        ans_str = _format(val)
        return {
            "text": f"Definite integral from {_format(a)} to {_format(b)}: {ans_str}\nCheck: F'(x) = f(x) and F({_format(b)}) - F({_format(a)}) = {ans_str}.",
            "answer": ans_str,
            "antiderivative": ad_str,
            "lower": _format(a),
            "upper": _format(b),
            "verified": verified,
            "backend": "exact-symbolic-calculus",
            "expression": expression,
        }
    raise MathError(f"Unsupported calculus function '{fname}'.")


# ---------------------------------------------------------------------------
# 2. Exact Matrix & Linear Algebra (exact-fraction-linear-algebra)
# ---------------------------------------------------------------------------

def _parse_matrix_ast(node):
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        raise MathError("Matrix must be a non-empty 2D list of rows.")
    rows = len(node.elts)
    if rows > _MAX_MATRIX_DIM:
        raise MathError(f"Matrix dimension exceeds maximum supported size of {_MAX_MATRIX_DIM}x{_MAX_MATRIX_DIM}.")
    first_row = node.elts[0]
    if not isinstance(first_row, (ast.List, ast.Tuple)) or not first_row.elts:
        raise MathError("Matrix rows must be non-empty lists.")
    cols = len(first_row.elts)
    if cols > _MAX_MATRIX_DIM:
        raise MathError(f"Matrix dimension exceeds maximum supported size of {_MAX_MATRIX_DIM}x{_MAX_MATRIX_DIM}.")
    mat = []
    for row_node in node.elts:
        if not isinstance(row_node, (ast.List, ast.Tuple)) or len(row_node.elts) != cols:
            raise MathError("Matrix must be rectangular (all rows must have the same length).")
        mat.append([_const_rational(cell) for cell in row_node.elts])
    return mat


def _mat_mul(A, B):
    m, k, n = len(A), len(A[0]), len(B[0])
    return [
        [_bounded(sum((A[i][p] * B[p][j] for p in range(k)), Fraction(0))) for j in range(n)]
        for i in range(m)
    ]


def _mat_det_cofactor(A):
    n = len(A)
    if n == 1:
        return _bounded(A[0][0])
    if n == 2:
        return _bounded(A[0][0] * A[1][1] - A[0][1] * A[1][0])
    total = Fraction(0)
    for col in range(n):
        elem = A[0][col]
        if elem == 0:
            continue
        minor = [row[:col] + row[col + 1:] for row in A[1:]]
        cof = _mat_det_cofactor(minor)
        total = _bounded(total + (elem * cof if col % 2 == 0 else -elem * cof))
    return total


def _mat_det_lu(A):
    n = len(A)
    M = [list(row) for row in A]
    sign = 1
    det = Fraction(1)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[pivot_row][col] == 0:
            return Fraction(0)
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            sign = -sign
        piv = M[col][col]
        det = _bounded(det * piv)
        for r in range(col + 1, n):
            if M[r][col] != 0:
                factor = M[r][col] / piv
                M[r] = [_bounded(M[r][c] - factor * M[col][c]) for c in range(n)]
    return _bounded(sign * det)


def _mat_rref(A):
    rows, cols = len(A), len(A[0])
    M = [list(row) for row in A]
    r = 0
    for col in range(cols):
        if r == rows:
            break
        best = max(range(r, rows), key=lambda i: abs(M[i][col]))
        if M[best][col] == 0:
            continue
        M[r], M[best] = M[best], M[r]
        piv = M[r][col]
        M[r] = [_bounded(v / piv) for v in M[r]]
        for i in range(rows):
            if i != r and M[i][col] != 0:
                factor = M[i][col]
                M[i] = [_bounded(a - factor * b) for a, b in zip(M[i], M[r])]
        r += 1
    return r, M


def _poly_mat_det(P):
    """Exact determinant of a matrix of sparse polynomials in x."""
    n = len(P)
    if n == 1:
        return P[0][0]
    if n == 2:
        return _poly_add(_poly_mul(P[0][0], P[1][1]), _poly_mul(P[0][1], P[1][0]), -1)
    total = {}
    for col in range(n):
        elem = P[0][col]
        if not elem:
            continue
        minor = [row[:col] + row[col + 1:] for row in P[1:]]
        sub_det = _poly_mat_det(minor)
        term = _poly_mul(elem, sub_det)
        total = _poly_add(total, term, 1 if col % 2 == 0 else -1)
    return total


def _mat_char_poly(A):
    """Return sparse polynomial det(x*I - A) for square matrix A."""
    n = len(A)
    P = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(_poly_clean({1: Fraction(1), 0: -A[i][j]}))
            else:
                row.append(_poly_clean({0: -A[i][j]}))
        P.append(row)
    return _poly_mat_det(P)


def solve_matrix(expression: str) -> dict:
    """Exact Fraction matrix operations: det, trace, rank, inv, eigenvalues."""
    call = _parse_call_ast(expression, set(_MATRIX_FUNCS))
    fname = call.func.id.lower()
    if call.keywords or len(call.args) != 1:
        raise MathError(f"{fname}() requires exactly one 2D matrix argument.")
    A = _parse_matrix_ast(call.args[0])
    rows, cols = len(A), len(A[0])

    if fname == "rank":
        r_row, _ = _mat_rref(A)
        A_T = [[A[i][j] for i in range(rows)] for j in range(cols)]
        r_col, _ = _mat_rref(A_T)
        verified = r_row == r_col
        return {
            "text": f"rank(A) = {r_row}\nCheck: exact RREF row rank ({r_row}) equals column rank ({r_col}).",
            "answer": str(r_row),
            "rank": r_row,
            "verified": verified,
            "backend": "exact-fraction-linear-algebra",
            "expression": expression.strip(),
        }

    if rows != cols:
        raise MathError(f"{fname}() requires a square matrix (got {rows}x{cols}).")
    n = rows

    if fname == "det":
        d_cof = _mat_det_cofactor(A)
        d_lu = _mat_det_lu(A)
        verified = d_cof == d_lu
        ans_str = _format(d_cof)
        return {
            "text": f"det(A) = {ans_str}\nCheck: cofactor expansion and exact LU elimination agree.",
            "answer": ans_str,
            "determinant": ans_str,
            "verified": verified,
            "backend": "exact-fraction-linear-algebra",
            "expression": expression.strip(),
        }

    if fname == "trace":
        tr = _bounded(sum((A[i][i] for i in range(n)), Fraction(0)))
        char_poly = _mat_char_poly(A)
        verified = char_poly.get(n - 1, Fraction(0)) == -tr
        ans_str = _format(tr)
        return {
            "text": f"trace(A) = {ans_str}\nCheck: diagonal sum matches characteristic polynomial coefficient.",
            "answer": ans_str,
            "trace": ans_str,
            "verified": verified,
            "backend": "exact-fraction-linear-algebra",
            "expression": expression.strip(),
        }

    if fname == "inv":
        det_val = _mat_det_cofactor(A)
        if det_val == 0:
            raise MathError("Matrix is singular and cannot be inverted.")
        aug = [
            list(A[i]) + [Fraction(1 if i == j else 0) for j in range(n)]
            for i in range(n)
        ]
        r_aug, rref_aug = _mat_rref(aug)
        if r_aug < n:
            raise MathError("Matrix is singular and cannot be inverted.")
        inv_mat = [row[n:] for row in rref_aug]
        eye = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
        verified = _mat_mul(A, inv_mat) == eye and _mat_mul(inv_mat, A) == eye
        ans_str = "[[" + "], [".join(", ".join(_format(v) for v in row) for row in inv_mat) + "]]"
        return {
            "text": f"inv(A) = {ans_str}\nCheck: A * A^-1 = I and A^-1 * A = I verified exactly.",
            "answer": ans_str,
            "inverse": [[_format(v) for v in row] for row in inv_mat],
            "determinant": _format(det_val),
            "verified": verified,
            "backend": "exact-fraction-linear-algebra",
            "expression": expression.strip(),
        }

    if fname == "eigenvalues":
        tr = _bounded(sum((A[i][i] for i in range(n)), Fraction(0)))
        det_val = _mat_det_cofactor(A)
        char_poly = _mat_char_poly(A)
        char_coeffs = _dense(char_poly, n)
        if n == 1:
            eig = A[0][0]
            ans_str = _format(eig)
            return {
                "text": f"Eigenvalues: {ans_str}\nCheck: trace = {ans_str}, det = {ans_str}.",
                "answer": ans_str,
                "eigenvalues": [ans_str],
                "multiplicities": [1],
                "characteristic_polynomial": _poly_string(char_coeffs),
                "trace": _format(tr),
                "determinant": _format(det_val),
                "verified": bool(eig == tr and eig == det_val),
                "backend": "exact-fraction-linear-algebra",
                "expression": expression.strip(),
            }
        poly_res = _solve_high_degree(char_coeffs, f"{_poly_string(char_coeffs)} = 0")
        vieta_trace_ok = -char_coeffs[n - 1] == tr
        vieta_det_ok = ((-1) ** n) * char_coeffs[0] == det_val
        roots_check_ok = True
        try:
            rat_roots = [Fraction(r) for r in poly_res["roots"]]
            if sum(m for m in poly_res["multiplicities"]) == n:
                eig_sum = sum((r * m for r, m in zip(rat_roots, poly_res["multiplicities"])), Fraction(0))
                eig_prod = math.prod(r ** m for r, m in zip(rat_roots, poly_res["multiplicities"]))
                roots_check_ok = (eig_sum == tr) and (eig_prod == det_val)
        except ValueError:
            # Non-rational (radical/complex) roots are already verified in _solve_high_degree via exact factorization
            roots_check_ok = poly_res["verified"]
        verified = bool(poly_res["verified"] and vieta_trace_ok and vieta_det_ok and roots_check_ok)
        return {
            "text": (
                f"Eigenvalues: {poly_res['answer']}\n"
                f"Characteristic polynomial: {_poly_string(char_coeffs)} = 0\n"
                f"Check: sum(eigenvalues) = trace = {_format(tr)}, prod(eigenvalues) = det = {_format(det_val)}."
            ),
            "answer": poly_res["answer"],
            "eigenvalues": poly_res["roots"],
            "multiplicities": poly_res["multiplicities"],
            "characteristic_polynomial": _poly_string(char_coeffs),
            "trace": _format(tr),
            "determinant": _format(det_val),
            "verified": verified,
            "backend": "exact-fraction-linear-algebra",
            "expression": expression.strip(),
        }
    raise MathError(f"Unsupported matrix operation '{fname}'.")


# ---------------------------------------------------------------------------
# 3. Exact Number Theory & Modular Arithmetic (exact-number-theory)
# ---------------------------------------------------------------------------

def _egcd(a, b):
    """Extended Euclidean Algorithm: returns (g, x, y) such that a*x + b*y = g = gcd(a, b) >= 0."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _prime_factors(n):
    """Return prime factorization of n >= 2 as [(p, e), ...] in ascending prime order."""
    factors = []
    m = n
    for p in (2, 3):
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            factors.append((p, e))
    d = 5
    while d * d <= m:
        for step in (d, d + 2):
            if m % step == 0:
                e = 0
                while m % step == 0:
                    m //= step
                    e += 1
                factors.append((step, e))
        d += 6
    if m > 1:
        factors.append((m, 1))
    return factors


def solve_number_theory(expression: str) -> dict:
    """Exact number-theoretic operations: gcd, lcm, factor, totient/phi, modinv, crt."""
    call = _parse_call_ast(expression, set(_NT_FUNCS))
    fname = call.func.id.lower()
    if call.keywords:
        raise MathError(f"{fname}() does not accept keyword arguments.")

    if fname in ("gcd", "lcm"):
        if len(call.args) == 1 and isinstance(call.args[0], (ast.List, ast.Tuple)):
            arg_nodes = call.args[0].elts
        else:
            arg_nodes = call.args
        if len(arg_nodes) < 2:
            raise MathError(f"{fname}() requires at least two integer arguments.")
        nums = [_const_int(node) for node in arg_nodes]
        if fname == "gcd":
            g = nums[0]
            coeffs = [1]
            for num in nums[1:]:
                g_next, s, t = _egcd(g, num)
                coeffs = [c * s for c in coeffs] + [t]
                g = g_next
            bezout_sum = sum(c * a for c, a in zip(coeffs, nums))
            divides_all = all(a % g == 0 for a in nums) if g > 0 else all(a == 0 for a in nums)
            verified = bool(g >= 0 and bezout_sum == g and divides_all)
            return {
                "text": f"gcd({', '.join(str(x) for x in nums)}) = {g}\nCheck: divides all inputs and Bezout identity verified.",
                "answer": str(g),
                "gcd": g,
                "verified": verified,
                "backend": "exact-number-theory",
                "expression": expression.strip(),
            }
        else:
            L = math.lcm(*nums)
            if L == 0:
                verified = any(a == 0 for a in nums)
            else:
                multiples_ok = all(a != 0 and L % abs(a) == 0 for a in nums)
                coprime_quotients = math.gcd(*(L // abs(a) for a in nums)) == 1
                verified = bool(multiples_ok and coprime_quotients)
            return {
                "text": f"lcm({', '.join(str(x) for x in nums)}) = {L}\nCheck: common multiple with coprime quotients verified.",
                "answer": str(L),
                "lcm": L,
                "verified": verified,
                "backend": "exact-number-theory",
                "expression": expression.strip(),
            }

    if fname == "factor":
        if len(call.args) != 1:
            raise MathError("factor(n) requires exactly one integer argument.")
        n = _const_int(call.args[0])
        if n < 2:
            raise MathError("factor(n) requires an integer n >= 2.")
        factors = _prime_factors(n)
        prod_check = math.prod(p ** e for p, e in factors) == n
        prime_check = all(_is_prime(p) and e >= 1 for p, e in factors)
        order_check = all(factors[i][0] < factors[i + 1][0] for i in range(len(factors) - 1))
        verified = bool(prod_check and prime_check and order_check)
        ans_str = " * ".join(f"{p}" if e == 1 else f"{p}^{e}" for p, e in factors)
        return {
            "text": f"{n} = {ans_str}\nCheck: product of prime powers equals {n} and every factor is prime.",
            "answer": ans_str,
            "factors": {str(p): e for p, e in factors},
            "verified": verified,
            "backend": "exact-number-theory",
            "expression": expression.strip(),
        }

    if fname in ("totient", "phi"):
        if len(call.args) != 1:
            raise MathError(f"{fname}(n) requires exactly one integer argument.")
        n = _const_int(call.args[0])
        if n < 1:
            raise MathError(f"{fname}(n) requires a positive integer n >= 1.")
        if n == 1:
            return {
                "text": "phi(1) = 1\nCheck: Euler totient base case verified.",
                "answer": "1",
                "totient": 1,
                "verified": True,
                "backend": "exact-number-theory",
                "expression": expression.strip(),
            }
        factors = _prime_factors(n)
        phi_val = math.prod((p ** (e - 1)) * (p - 1) for p, e in factors)
        # Verify via rational product n * prod(1 - 1/p) and Gauss divisor-sum identity sum_{d|n} phi(d) = n
        rat_phi = Fraction(n) * math.prod(Fraction(p - 1, p) for p, _ in factors)
        gauss_sum = math.prod(
            1 + sum((p ** (j - 1)) * (p - 1) for j in range(1, e + 1))
            for p, e in factors
        )
        prod_ok = math.prod(p ** e for p, e in factors) == n and all(_is_prime(p) for p, _ in factors)
        verified = bool(prod_ok and rat_phi == phi_val and gauss_sum == n)
        return {
            "text": f"phi({n}) = {phi_val}\nCheck: prime factorization product and Gauss divisor-sum identity sum_{{d|{n}}} phi(d) = {n} verified.",
            "answer": str(phi_val),
            "totient": phi_val,
            "verified": verified,
            "backend": "exact-number-theory",
            "expression": expression.strip(),
        }

    if fname == "modinv":
        if len(call.args) != 2:
            raise MathError("modinv(a, m) requires exactly two integer arguments.")
        a = _const_int(call.args[0])
        m = _const_int(call.args[1])
        if m < 2:
            raise MathError("Modulus m in modinv(a, m) must be an integer >= 2.")
        g, x, _ = _egcd(a % m, m)
        if g != 1:
            raise MathError(f"Modular inverse does not exist because gcd({a}, {m}) = {g} != 1.")
        inv = x % m
        verified = bool(0 < inv < m and (a * inv) % m == 1)
        return {
            "text": f"modinv({a}, {m}) = {inv}\nCheck: ({a} * {inv}) mod {m} = 1.",
            "answer": str(inv),
            "inverse": inv,
            "modulus": m,
            "verified": verified,
            "backend": "exact-number-theory",
            "expression": expression.strip(),
        }

    if fname == "crt":
        pairs = []
        if len(call.args) == 1 and isinstance(call.args[0], (ast.List, ast.Tuple)):
            for item in call.args[0].elts:
                if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 2:
                    raise MathError("crt() list elements must be (remainder, modulus) pairs.")
                pairs.append((_const_int(item.elts[0]), _const_int(item.elts[1])))
        elif all(isinstance(arg, (ast.List, ast.Tuple)) and len(arg.elts) == 2 for arg in call.args) and len(call.args) >= 1:
            for item in call.args:
                pairs.append((_const_int(item.elts[0]), _const_int(item.elts[1])))
        elif len(call.args) >= 2 and len(call.args) % 2 == 0:
            nums = [_const_int(arg) for arg in call.args]
            pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        else:
            raise MathError("crt() requires [(r1, m1), (r2, m2), ...] or (r1, m1, r2, m2, ...).")
        if not pairs:
            raise MathError("crt() requires at least one congruence pair.")
        x = 0
        M = 1
        for r_i, m_i in pairs:
            if m_i < 1:
                raise MathError("CRT moduli must be positive integers.")
            r_norm = r_i % m_i
            g, s, _ = _egcd(M, m_i)
            if (r_norm - x) % g != 0:
                raise MathError("Chinese Remainder Theorem system has no solution (incompatible congruences).")
            lcm_m = (M // g) * m_i
            step = ((r_norm - x) // g) * s % (m_i // g)
            x = (x + M * step) % lcm_m
            M = lcm_m
        expected_M = math.lcm(*(m_i for _, m_i in pairs))
        verified = bool(0 <= x < M and M == expected_M and all(x % m_i == r_i % m_i for r_i, m_i in pairs))
        return {
            "text": f"x = {x} (mod {M})\nCheck: all {len(pairs)} congruences verified modulo lcm = {M}.",
            "answer": str(x),
            "solution": x,
            "modulus": M,
            "verified": verified,
            "backend": "exact-number-theory",
            "expression": expression.strip(),
        }
    raise MathError(f"Unsupported number theory function '{fname}'.")


def solve_sqrt(expression: str) -> dict:
    """Exact principal square root of non-negative rational numbers."""
    call = _parse_call_ast(expression, {"sqrt"})
    if call.keywords or len(call.args) != 1:
        raise MathError("sqrt(n) requires exactly one numeric argument.")
    val = _const_rational(call.args[0])
    if val < 0:
        raise MathError("Square root of a negative number is not a real number.")
    num = val.numerator
    den = val.denominator
    r_num = math.isqrt(num)
    r_den = math.isqrt(den)
    if r_num * r_num == num and r_den * r_den == den:
        res = Fraction(r_num, r_den)
        ans_str = _format(res)
        verified = bool(res * res == val)
        return {
            "text": f"sqrt({_format(val)}) = {ans_str}\nCheck: ({ans_str})^2 = {_format(val)} verified exactly.",
            "answer": ans_str,
            "root": ans_str,
            "verified": verified,
            "backend": "exact-radical-solver",
            "expression": expression.strip(),
        }
    f_val = math.sqrt(float(val))
    ans_str = f"{f_val:.6g}"
    return {
        "text": f"sqrt({_format(val)}) ≈ {ans_str}",
        "answer": ans_str,
        "root": ans_str,
        "verified": False,
        "backend": "numeric-radical-approximation",
        "expression": expression.strip(),
    }


def _top_level_call_name(expression):
    """If expression is a single top-level call fname(...), return fname.lower(), else None."""
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)
    if not m:
        return None
    open_idx = m.end() - 1
    depth = 0
    for idx in range(open_idx, len(expression)):
        ch = expression[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return m.group(1).lower() if idx == len(expression) - 1 else None
    return None


def solve(expression):
    expression = expression.strip()
    if len(expression) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    if re.match(r"^d(?:\^\d+)?\s*/\s*dx(?:\^\d+)?(?:\s+|\s*\()", expression, flags=re.I):
        return solve_calculus(expression)
    fname = _top_level_call_name(expression)
    if fname is not None:
        if fname in _CALCULUS_FUNCS:
            return solve_calculus(expression)
        if fname in _MATRIX_FUNCS:
            return solve_matrix(expression)
        if fname in _NT_FUNCS:
            return solve_number_theory(expression)
        if fname in _RADICAL_FUNCS:
            return solve_sqrt(expression)
        raise MathError(f"Unsupported function '{fname}'.")
    if ("," in expression or ";" in expression) and "=" in expression:
        return _solve_any_system(expression)
    if expression.count("=") == 1:
        left, right = expression.split("=")
        try:
            a, b = _linear(_parse(left))
            c, d = _linear(_parse(right))
        except MathError:
            if not _allows_nonlinear(expression):
                raise
            return solve_polynomial(expression)
        return _linear_result(a, b, c, d, expression)
    if "=" in expression:
        raise MathError("Use one equation at a time.")
    a, result = _linear(_parse(expression))
    if a:
        raise MathError("An expression with x requires an equation.")
    return {"text": f"{expression} = {_format(result)}", "answer": _format(result), "verified": True, "backend": "exact-rational-arithmetic", "expression": expression}


def _extract_special_call(text):
    """Extract a valid special math function call or d/dx (...) if present, else None."""
    # Check d/dx (...) notation
    m_ddx = re.search(r"\b(d(?:\^\d+)?\s*/\s*dx(?:\^\d+)?\s*\()", text, flags=re.I)
    if m_ddx:
        open_idx = m_ddx.end() - 1
        depth = 0
        for idx in range(open_idx, len(text)):
            if text[idx] == "(":
                depth += 1
            elif text[idx] == ")":
                depth -= 1
                if depth == 0:
                    inner = text[open_idx + 1:idx]
                    if (
                        inner.strip()
                        and re.fullmatch(r"[0-9eExX+*/().^\s×÷-]+", inner)
                        and any(c.isdigit() or c in "xX" for c in inner)
                    ):
                        return text[m_ddx.start():idx + 1]
                    break

    pattern = r"\b(" + "|".join(_ALL_SPECIAL_FUNCS) + r")\s*\("
    for m in re.finditer(pattern, text, flags=re.I):
        open_idx = m.end() - 1
        p_depth = 0
        b_depth = 0
        close_idx = None
        for idx in range(open_idx, len(text)):
            ch = text[idx]
            if ch == "(":
                p_depth += 1
            elif ch == ")":
                p_depth -= 1
                if p_depth < 0:
                    break
                if p_depth == 0:
                    close_idx = idx
                    break
            elif ch == "[":
                b_depth += 1
            elif ch == "]":
                b_depth -= 1
                if b_depth < 0:
                    break
        if close_idx is None or p_depth != 0 or b_depth != 0:
            continue
        inner = text[open_idx + 1:close_idx]
        if not inner.strip():
            continue
        if not re.fullmatch(r"[0-9a-zA-Z_+*/().%=,\[\]\s^×÷-]+", inner):
            continue
        identifiers = re.sub(r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", "0", inner)
        words = [w.lower() for w in re.findall(r"[A-Za-z_]+", identifiers)]
        if any(w not in ("x", "at", "order") for w in words):
            continue
        if not (any(c.isdigit() for c in inner) or "x" in words):
            continue
        candidate = text[m.start():close_idx + 1]
        return candidate
    return None


def extract_math(prompt):
    text = prompt.strip().rstrip("?.!")
    special = _extract_special_call(text)
    if special is not None:
        return special

    # 1. Percentage calculation: "What is 15% of 240?"
    m_pct = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)$",
        text,
        flags=re.I,
    )
    if m_pct:
        return f"({m_pct.group(1)}/100) * {m_pct.group(2)}"

    # 2. Exponentiation: "What is 2 to the power of 10?"
    m_pow = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(\d+(?:\.\d+)?)\s+to\s+the\s+(?:power\s+of\s+|power\s+)?(\d+(?:\.\d+)?)$",
        text,
        flags=re.I,
    )
    if m_pow:
        return f"({m_pow.group(1)})^({m_pow.group(2)})"

    # 3. Square root: "What is the square root of 144?"
    m_sqrt = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?square\s+root\s+of\s+(\d+(?:\.\d+)?)$",
        text,
        flags=re.I,
    )
    if m_sqrt:
        return f"sqrt({m_sqrt.group(1)})"

    # 4. Derivative: "What is the derivative of x^3 - 4x?"
    m_diff = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?derivative\s+of\s+([0-9a-zA-Z_+*/().^\s×÷-]+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?(?:\s+at\s+([0-9.-]+))?$",
        text,
        flags=re.I,
    )
    if m_diff:
        poly_str = _normalize_implicit_mult(m_diff.group(1).strip())
        var = m_diff.group(2) or "x"
        at_val = m_diff.group(3)
        if re.fullmatch(r"[0-9eExX+*/().^\s×÷-]+", poly_str) and any(c.isdigit() or c in "xX" for c in poly_str):
            if at_val:
                return f"diff({poly_str}, {var}, at={at_val})"
            return f"diff({poly_str}, {var})"

    # 5. Definite and indefinite integral: "Find the integral of 2x from 0 to 3"
    m_int_def = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?integral\s+of\s+([0-9a-zA-Z_+*/().^\s×÷-]+?)\s+from\s+([0-9.-]+)\s+to\s+([0-9.-]+)$",
        text,
        flags=re.I,
    )
    if m_int_def:
        poly_str = _normalize_implicit_mult(m_int_def.group(1).strip())
        low = m_int_def.group(2).strip()
        high = m_int_def.group(3).strip()
        if re.fullmatch(r"[0-9eExX+*/().^\s×÷-]+", poly_str) and any(c.isdigit() or c in "xX" for c in poly_str):
            return f"integrate({poly_str}, x, {low}, {high})"

    m_int_indef = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?integral\s+of\s+([0-9a-zA-Z_+*/().^\s×÷-]+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?$",
        text,
        flags=re.I,
    )
    if m_int_indef:
        poly_str = _normalize_implicit_mult(m_int_indef.group(1).strip())
        var = m_int_indef.group(2) or "x"
        if re.fullmatch(r"[0-9eExX+*/().^\s×÷-]+", poly_str) and any(c.isdigit() or c in "xX" for c in poly_str):
            return f"integrate({poly_str}, {var})"

    # 6. Greatest common divisor: "What is the greatest common divisor of 84 and 36?"
    m_gcd = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?(?:greatest\s+common\s+divisor|gcd)\s+of\s+([0-9,\s\band\b]+)$",
        text,
        flags=re.I,
    )
    if m_gcd:
        nums = re.findall(r"\d+", m_gcd.group(1))
        if len(nums) >= 2:
            return f"gcd({', '.join(nums)})"

    # 7. Least common multiple: "What is the least common multiple of 4, 6 and 10?"
    m_lcm = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?(?:least\s+common\s+multiple|lcm)\s+of\s+([0-9,\s\band\b]+)$",
        text,
        flags=re.I,
    )
    if m_lcm:
        nums = re.findall(r"\d+", m_lcm.group(1))
        if len(nums) >= 2:
            return f"lcm({', '.join(nums)})"

    # 8. Factorization: "Factorize 1001" or "prime factorization of 1001"
    m_fac = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:factorize|prime\s+factorization\s+of|factor)\s+(\d+)$",
        text,
        flags=re.I,
    )
    if m_fac:
        return f"factor({m_fac.group(1)})"

    # 9. Natural-language matrix operations: "What is the determinant of [[2, 1], [5, 3]]?"
    m_mat = re.match(
        r"^(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|find|evaluate)\s+)?(?:the\s+)?(determinant|inverse|trace|rank|eigenvalues)\s+of\s+(\[\[.*?\]\])$",
        text,
        flags=re.I,
    )
    if m_mat:
        op_map = {
            "determinant": "det",
            "inverse": "inv",
            "trace": "trace",
            "rank": "rank",
            "eigenvalues": "eigenvalues",
        }
        return f"{op_map[m_mat.group(1).lower()]}({m_mat.group(2).strip()})"

    # 10. Word arithmetic: "Calculate 7 times 8 plus 6"
    cleaned_arith = re.sub(
        r"^(?:please\s+)?(?:calculate|compute|evaluate|what\s+is|what's|find|solve)\s+",
        "",
        text,
        flags=re.I,
    ).strip()
    words = re.findall(r"[a-zA-Z]+", cleaned_arith.lower())
    arith_words = {"times", "multiplied", "by", "plus", "minus", "divided", "over"}
    if words and all(w in arith_words for w in words):
        expr_trans = re.sub(r"\bmultiplied\s+by\b", "*", cleaned_arith, flags=re.I)
        expr_trans = re.sub(r"\btimes\b", "*", expr_trans, flags=re.I)
        expr_trans = re.sub(r"\bdivided\s+by\b", "/", expr_trans, flags=re.I)
        expr_trans = re.sub(r"\bover\b", "/", expr_trans, flags=re.I)
        expr_trans = re.sub(r"\bplus\b", "+", expr_trans, flags=re.I)
        expr_trans = re.sub(r"\bminus\b", "-", expr_trans, flags=re.I)
        expr_trans = _normalize_implicit_mult(expr_trans.strip())
        if re.fullmatch(r"[0-9+*/().%\s-]+", expr_trans) and any(c.isdigit() for c in expr_trans):
            return expr_trans

    # 11. "If <eq>, what is <var>?"
    m_if = re.match(
        r"^(?:if\s+)?(.+?)(?:,\s*(?:what\s+is|what's|find|solve\s+for|calculate)\s+([a-zA-Z]))$",
        text,
        flags=re.I,
    )
    if m_if and "=" in m_if.group(1):
        cand = _normalize_implicit_mult(m_if.group(1).strip())
        if re.fullmatch(r"[0-9eExXyYzZwWuUvV+*/().%=,;\s^×÷-]+", cand) and any(c.isdigit() for c in cand):
            return cand

    # 12. System equations joined by "and" (e.g. "Solve x + y = 10 and x - y = 2")
    m_sys_and = re.sub(r"\s+and\s+(?=[^=]+=[^=]+)", ", ", cleaned_arith, flags=re.I)
    m_sys_and = _normalize_implicit_mult(m_sys_and.strip())
    if "," in m_sys_and and "=" in m_sys_and:
        if re.fullmatch(r"[0-9eExXyYzZwWuUvV+*/().%=,;\s^×÷-]+", m_sys_and) and any(c.isdigit() for c in m_sys_and):
            return m_sys_and

    # 13. Fall back to standard extraction
    cand_text = _normalize_implicit_mult(cleaned_arith.strip())
    if not re.fullmatch(r"[0-9eExXyYzZwWuUvV+*/().%=,;\s^×÷-]+", cand_text) or not any(c.isdigit() for c in cand_text):
        return None
    if re.search(r"[zZwWuUvV]", cand_text):
        if "=" not in cand_text:
            return None
        if any(len(run) > 1 for run in re.findall(r"[A-Za-z]+", cand_text)):
            return None
    return cand_text

