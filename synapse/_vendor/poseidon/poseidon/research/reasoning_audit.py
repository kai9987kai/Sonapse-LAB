"""Bounded generated solver audit with independent answer checks.

This is a finite software audit, not a calibrated confidence model. Acceptance
uses the solver's explicit verification flag; correctness is checked separately.
No weights, language generation, or synthetic controller is needed.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable

from ..provenance import assert_source_current

SCHEMA = "poseidon.reasoning-audit.v1"
MAX_SAMPLES = 32
DEFAULT_SEED = 20261030
_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = ("poseidon/research/reasoning_audit.py", "poseidon/reasoning.py",
            "poseidon/problem_solver.py")
_ANSWER_LIMIT = 10**80
_ANSWER_TEXT_LIMIT = 192  # also accommodates normalized 80-digit numerator/denominator pairs


def _hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in _SOURCES}


def _edit_distance(left: str, right: str) -> int:
    # Recursive oracle differs from the shipped solver's table/backtracking path.
    @lru_cache(None)
    def distance(i: int, j: int) -> int:
        if not i:
            return j
        if not j:
            return i
        return min(distance(i - 1, j) + 1, distance(i, j - 1) + 1,
                   distance(i - 1, j - 1) + (left[i - 1] != right[j - 1]))
    return distance(len(left), len(right))


def generate_cases(seed: int = DEFAULT_SEED, samples: int = 8) -> list[dict[str, Any]]:
    """Six parameterized families plus fixed unsupported/invalid controls."""
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    def add(family, prompt, expected, parameters=None, answerable=True):
        cases.append({"id": f"{family}-{len(cases):03d}", "family": family,
                      "prompt": prompt, "expected": expected,
                      "parameters": parameters or {}, "answerable": answerable})

    for _ in range(samples):
        a, b, c = (rng.randint(-90, 90) for _ in range(3))
        add("arithmetic", f"({a} + {b}) * {c}", str((a + b) * c))
        a = rng.choice([x for x in range(-12, 13) if x])
        b, c = rng.randint(-60, 60), rng.randint(-60, 60)
        add("linear", f"{a}*x + {b} = {c}", str(Fraction(c - b, a)))
        a, b, c, d = (rng.randint(-15, 15) for _ in range(4))
        add("determinant", f"det([[{a},{b}],[{c},{d}]])", str(a*d - b*c))
        a, b = rng.randint(2, 900), rng.randint(2, 900)
        # Independent divisor enumeration, rather than the solver's Euclid path.
        common = max(x for x in range(1, min(a, b) + 1) if a % x == b % x == 0)
        add("gcd", f"gcd({a},{b})", str(common))
        n = rng.randint(2, 7)
        add("hanoi", f"Tower of Hanoi with {n} disks", str(2**n - 1), {"n": n})
        left = "".join(rng.choices("abcdef", k=rng.randint(2, 7)))
        right = "".join(rng.choices("abcdef", k=rng.randint(2, 7)))
        add("edit", f"What is the edit distance between '{left}' and '{right}'?",
            _edit_distance(left, right), {"left": left, "right": right})
    for prompt in ("Who painted this picture?", "What is my favourite colour?",
                   "What did I eat yesterday?", "Where did I leave my keys?"):
        add("unsupported", prompt, None, answerable=False)
    for prompt in ("1 / 0", "det([[1,2],[3]])", "modinv(2,4)", "3**9999"):
        add("invalid", prompt, None, answerable=False)
    return cases


def route(prompt: str) -> tuple[str | None, dict | None]:
    """Use chat's exact-math then parsed-problem precedence, without prose."""
    from ..reasoning import extract_math, solve
    from ..problem_solver import detect_problem, solve_problem
    expression = extract_math(prompt)
    if expression:
        return "math", solve(expression)
    domain = detect_problem(prompt)
    if domain:
        return f"problem:{domain}", solve_problem(prompt, domain=domain)
    return None, None


def _bounded_answer_fraction(raw: Any) -> Fraction:
    """Parse bounded exact answers without exponentiating an untrusted exponent."""
    if not isinstance(raw, str) or len(raw) > _ANSWER_TEXT_LIMIT:
        raise ValueError("numeric answer exceeds its text budget")
    text = raw.strip()
    rational = re.fullmatch(r"([+-]?[0-9]+)\s*/\s*([0-9]+)", text)
    if rational:
        numerator, denominator = map(int, rational.groups())
        if denominator == 0:
            raise ValueError("numeric answer has a zero denominator")
    else:
        decimal = re.fullmatch(r"([+-]?)([0-9]*)(?:\.([0-9]*))?(?:[eE]([+-]?[0-9]+))?", text)
        if decimal is None:
            raise ValueError("numeric answer is not a finite rational")
        sign, whole, fractional, exponent = decimal.groups()
        fractional = fractional or ""
        if not whole and not fractional:
            raise ValueError("numeric answer has no digits")
        exponent = int(exponent or "0")
        # The text has at most 192 digits. Larger exponents cannot produce a
        # nonzero value within the 80-digit budget, even after cancellation.
        if abs(exponent) > _ANSWER_TEXT_LIMIT + 80:
            raise ValueError("numeric answer exponent exceeds its budget")
        digits = (whole + fractional).lstrip("0")
        scale = exponent - len(fractional)
        if not digits:
            numerator, denominator = 0, 1
        else:
            if scale > 80 or -scale > 80 + len(digits):
                raise ValueError("numeric answer exceeds its numeric budget")
            numerator = int(digits) * (-1 if sign == "-" else 1)
            numerator, denominator = ((numerator * 10**scale, 1) if scale >= 0
                                      else (numerator, 10**(-scale)))
    divisor = math.gcd(numerator, denominator)
    numerator, denominator = numerator // divisor, denominator // divisor
    if abs(numerator) > _ANSWER_LIMIT or denominator > _ANSWER_LIMIT:
        raise ValueError("numeric answer exceeds its numeric budget")
    return Fraction(numerator, denominator)


def check_answer(case: dict, result: dict | None) -> bool:
    """Never accept a token match in prose or the solver's own verification flag."""
    if not case["answerable"] or not isinstance(result, dict):
        return False
    try:
        if case["family"] == "edit":
            params = case["parameters"]
            displayed = re.fullmatch(r"edit_distance = (\d+), LCS = '([a-f]*)'", str(result.get("answer", "")))
            if displayed is None or int(displayed[1]) != case["expected"]:
                return False
            lcs = displayed[2]
            def subsequence(text):
                cursor = iter(text)
                return all(char in cursor for char in lcs)
            @lru_cache(None)
            def lcs_length(left, right):
                if not left or not right:
                    return 0
                if left[-1] == right[-1]:
                    return 1 + lcs_length(left[:-1], right[:-1])
                return max(lcs_length(left[:-1], right), lcs_length(left, right[:-1]))
            return (result.get("s1") == params["left"]
                    and result.get("s2") == params["right"]
                    and type(result.get("edit_distance")) is int
                    and result["edit_distance"] == case["expected"]
                    and subsequence(params["left"]) and subsequence(params["right"])
                    and len(lcs) == lcs_length(params["left"], params["right"]))
        raw = result.get("answer")
        if not isinstance(raw, str) or len(raw) > 128:
            return False
        if _bounded_answer_fraction(raw) != _bounded_answer_fraction(case["expected"]):
            return False
        if case["family"] != "hanoi":
            return True
        n = case["parameters"]["n"]
        moves = result.get("moves")
        if not isinstance(moves, list) or len(moves) != 2**n - 1:
            return False
        pegs = {"A": list(range(n, 0, -1)), "B": [], "C": []}
        for move in moves:
            match = re.fullmatch(r"([1-7]):([ABC])->([ABC])", move)
            if match is None:
                return False
            disk, start, end = int(match[1]), match[2], match[3]
            if start == end or not pegs[start] or pegs[start][-1] != disk:
                return False
            if pegs[end] and pegs[end][-1] < disk:
                return False
            pegs[end].append(pegs[start].pop())
        return pegs["C"] == list(range(n, 0, -1))
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False


def _metrics(rows: list[dict]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    total = len(rows)
    accepted = sum(row["accepted"] for row in rows)
    wrong = counts["wrong"]
    return {"total": total, "answerable": sum(row["case"]["answerable"] for row in rows),
            "accepted": accepted, "correct": counts["correct"], "wrong": wrong, "errors": counts["error"],
            "correct_refusals": counts["refused"], "missed_answerable": counts["missed"],
            "false_acceptances": sum(row["accepted"] and not row["case"]["answerable"] for row in rows),
            "false_verifications": sum(row["self_verified"] and not row["correct"] for row in rows),
            "coverage": accepted / total if total else 0.0,
            "selective_risk": wrong / accepted if accepted else None,
            "case_accuracy": (counts["correct"] + counts["refused"]) / total if total else None}


def run_audit(seed: int = DEFAULT_SEED, samples: int = 8,
              *, router: Callable | None = None) -> dict[str, Any]:
    assert_source_current()
    cases = generate_cases(seed, samples)
    from ..reasoning import MathError
    from ..problem_solver import ProblemError
    rows = []
    for case in cases:
        error = None
        unexpected_error = False
        try:
            tool, result = (router or route)(case["prompt"])
        except Exception as exc:
            tool, result = "refused", None
            error = f"{type(exc).__name__}: {exc}"[:240]
            unexpected_error = not isinstance(exc, (MathError, ProblemError))
        result = result if isinstance(result, dict) else None
        self_verified = bool(result and result.get("verified") is True)
        accepted = bool(self_verified and result.get("answer") not in (None, ""))
        correct = bool(accepted and check_answer(case, result))
        status = ("correct" if correct else "wrong") if accepted else (
            "missed" if case["answerable"] else "refused")
        if unexpected_error:
            status = "error"
        rows.append({"case": case, "tool": tool, "accepted": accepted,
                     "self_verified": self_verified, "correct": correct, "status": status,
                     "answer": str(result.get("answer", ""))[:240] if result else None,
                     "error": error})
    assert_source_current()
    report = {"schema": SCHEMA, "config": {"seed": seed, "samples": samples},
              "scope": "finite_generated_suite", "acceptance_policy": "solver_verified_answer",
              "calibrated": False, "source_hashes": _source_hashes(),
              "case_manifest_sha256": _hash(cases), "metrics": _metrics(rows),
              "by_family": {family: _metrics([row for row in rows if row["case"]["family"] == family])
                            for family in sorted({case["family"] for case in cases})},
              "rows": rows,
              "limits": ["Generated local checks, not a held-out model benchmark or statistical risk guarantee.",
                         "Acceptance trusts the solver flag; correctness is independently checked for these cases.",
                         "Unsupported controls cover tool refusal, not general language-model abstention.",
                         "A fingerprint detects changes; replay re-executes the finite suite."]}
    report["report_sha256"] = _hash(report)
    return report


def verify_audit(report: dict, *, replay: bool = False) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("unsupported reasoning audit schema")
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    if _hash(payload) != report.get("report_sha256"):
        raise ValueError("reasoning audit fingerprint mismatch")
    cases = generate_cases(**report["config"])
    if (report.get("case_manifest_sha256") != _hash(cases)
            or [row["case"] for row in report["rows"]] != cases):
        raise ValueError("reasoning audit case manifest mismatch")
    if (report["metrics"] != _metrics(report["rows"]) or report["by_family"] != {
            family: _metrics([row for row in report["rows"] if row["case"]["family"] == family])
            for family in sorted({case["family"] for case in cases})}):
        raise ValueError("reasoning audit metrics mismatch")
    source_matches = report.get("source_hashes") == _source_hashes()
    replay_verified = False
    if replay:
        if not source_matches:
            raise ValueError("reasoning audit source changed; replay not verified")
        rerun = run_audit(**report["config"])
        if rerun != report:
            raise ValueError("reasoning audit does not replay")
        replay_verified = True
    return {"integrity_verified": True, "case_manifest_verified": True,
            "source_matches": source_matches, "replay_verified": replay_verified,
            "scope": "finite_generated_suite", "report_sha256": report["report_sha256"]}
