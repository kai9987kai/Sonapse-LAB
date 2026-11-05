"""Finite related-input checks with independent references for every answer.

Consistency alone can reward a solver that always gives the same wrong answer.
This audit therefore reports relation checks, reference correctness, refusals,
and their intersection separately. It checks local tools, not language weights.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Callable

from ..provenance import assert_source_current
from .reasoning_audit import _bounded_answer_fraction, _edit_distance, check_answer, route

SCHEMA = "poseidon.metamorphic-audit.v1"
MAX_SAMPLES = 32
DEFAULT_SEED = 20261031
_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = ("poseidon/research/metamorphic_audit.py",
            "poseidon/research/reasoning_audit.py", "poseidon/reasoning.py",
            "poseidon/problem_solver.py")
_ACCEPTANCE_POLICY = "solver_verified_answer"
_LIMITS = ("Finite local tool checks, not a language-model benchmark or statistical guarantee.",
           "Consistent answers can both be wrong; independent correctness is reported separately.",
           "Refusals and unparseable answers leave a relation unavailable rather than passing it.",
           "Fingerprints detect changes, not fabricated evidence; replay executes the built-in tools.",
           "Custom routers support integrity checking only and cannot replay as the built-in tools.")
_REPORT_KEYS = {"schema", "config", "scope", "calibrated", "router_id", "acceptance_policy",
                "source_hashes", "case_manifest_sha256", "metrics", "by_family", "groups",
                "limits", "report_sha256"}
_GROUP_KEYS = {"id", "family", "relation", "cases", "rows", "relation_status", "joint_correct"}
_ROW_KEYS = {"tool", "result", "error", "error_kind", "accepted", "self_verified", "correct",
             "status", "observed_value"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in _SOURCES}


def generate_groups(seed: int = DEFAULT_SEED, samples: int = 8) -> list[dict[str, Any]]:
    """Generate five pairs per sample, each with independent expected answers."""
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
    rng = random.Random(seed)
    groups = []

    def add(family, prompts, expected, relation="same_value", parameters=None):
        group_id = f"{family}-{len(groups):03d}"
        cases = [{"id": f"{group_id}-{index}", "family": family, "prompt": prompt,
                  "expected": answer, "parameters": (parameters or [{}, {}])[index],
                  "answerable": True}
                 for index, (prompt, answer) in enumerate(zip(prompts, expected))]
        groups.append({"id": group_id, "family": family, "relation": relation, "cases": cases})

    for _ in range(samples):
        integer, offset = rng.randint(10, 900), rng.randint(-30, 30)
        decimal = f"{integer}.00000000000000001"
        expected = str(Fraction(offset) + Fraction(1, 10**17))
        add("arithmetic", [f"({decimal} - {integer}) + {offset}",
                           f"{offset} + ({decimal} - {integer})"], [expected, expected])
        a = rng.choice([value for value in range(-12, 13) if value])
        b, c = rng.randint(-60, 60), rng.randint(-60, 60)
        scale = rng.choice([-7, -3, 2, 5])
        expected = str(Fraction(c - b, a))
        add("linear", [f"{a}*x + {b} = {c}",
                       f"{scale*a}*x + {scale*b} = {scale*c}"], [expected, expected])
        a, b, c, d = (rng.randint(-15, 15) for _ in range(4))
        determinant = a*d - b*c
        add("determinant", [f"det([[{a},{b}],[{c},{d}]])",
                            f"det([[{c},{d}],[{a},{b}]])"],
            [str(determinant), str(-determinant)], "negated_value")
        a, b = rng.randint(2, 900), rng.randint(2, 900)
        common = max(value for value in range(1, min(a, b) + 1) if a % value == b % value == 0)
        add("gcd", [f"gcd({a},{b})", f"gcd({b},{a})"], [str(common), str(common)])
        left = "".join(rng.choices("abcdef", k=rng.randint(2, 7)))
        right = "".join(rng.choices("abcdef", k=rng.randint(2, 7)))
        distance = _edit_distance(left, right)
        add("edit", [f"What is the edit distance between '{left}' and '{right}'?",
                     f"What is the edit distance between '{right}' and '{left}'?"],
            [distance, distance], parameters=[{"left": left, "right": right},
                                            {"left": right, "right": left}])
    return groups


def _project_result(result: Any) -> dict | None:
    if not isinstance(result, dict):
        return None
    # Only checker inputs are retained, keeping arbitrary custom routers bounded.
    projected = {"verified": result.get("verified") is True}
    for key in ("answer", "s1", "s2"):
        value = result.get(key)
        projected[key] = value[:512] if isinstance(value, str) else None
    value = result.get("edit_distance")
    projected["edit_distance"] = value if type(value) is int and abs(value) < 10**10 else None
    return projected


def _numeric_answer(case: dict, result: dict | None) -> str | None:
    if not result or not isinstance(result.get("answer"), str):
        return None
    raw = result["answer"]
    if case["family"] == "edit":
        match = re.fullmatch(r"edit_distance = (\d+), LCS = '[a-f]*'", raw)
        raw = match[1] if match else ""
    if len(raw) > 128:
        return None
    try:
        return str(_bounded_answer_fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None


def _row(case: dict, result: dict | None, tool: str | None,
         error: str | None, error_kind: str | None) -> dict[str, Any]:
    self_verified = bool(result and result.get("verified") is True)
    accepted = bool(self_verified and result.get("answer") not in (None, ""))
    correct = bool(accepted and check_answer(case, result))
    status = "correct" if correct else "wrong" if accepted else "refused"
    if error_kind == "exception":
        status = "error"
    return {"tool": tool, "result": result, "error": error, "error_kind": error_kind,
            "accepted": accepted, "self_verified": self_verified, "correct": correct,
            "status": status, "observed_value": _numeric_answer(case, result) if accepted else None}


def _relation(group: dict, rows: list[dict]) -> tuple[str, bool]:
    values = [row["observed_value"] for row in rows]
    status = "unavailable"
    if all(value is not None for value in values):
        left, right = map(_bounded_answer_fraction, values)
        passed = right == (left if group["relation"] == "same_value" else -left)
        status = "passed" if passed else "violated"
    return status, status == "passed" and all(row["correct"] for row in rows)


def _metrics(groups: list[dict]) -> dict[str, Any]:
    rows = [row for group in groups for row in group["rows"]]
    statuses = Counter(row["status"] for row in rows)
    relations = Counter(group["relation_status"] for group in groups)
    accepted = sum(row["accepted"] for row in rows)
    return {"groups": len(groups), "total": len(rows), "accepted": accepted,
            "correct": statuses["correct"], "wrong": statuses["wrong"],
            "errors": statuses["error"], "refusals": statuses["refused"],
            "false_verifications": sum(row["self_verified"] and not row["correct"] for row in rows),
            "coverage": accepted / len(rows) if rows else 0.0,
            "selective_risk": statuses["wrong"] / accepted if accepted else None,
            "relation_checked": relations["passed"] + relations["violated"],
            "relation_passed": relations["passed"], "relation_violations": relations["violated"],
            "relation_unavailable": relations["unavailable"],
            "joint_correct_groups": sum(group["joint_correct"] for group in groups)}


def _by_family(groups: list[dict]) -> dict[str, Any]:
    return {family: _metrics([group for group in groups if group["family"] == family])
            for family in sorted({group["family"] for group in groups})}


def run_audit(seed: int = DEFAULT_SEED, samples: int = 8,
              *, router: Callable | None = None) -> dict[str, Any]:
    assert_source_current()
    cases = generate_groups(seed, samples)
    from ..reasoning import MathError
    from ..problem_solver import ProblemError
    evaluated = []
    for group in cases:
        rows = []
        for case in group["cases"]:
            error = error_kind = None
            try:
                tool, result = (router or route)(case["prompt"])
            except Exception as exc:
                tool, result = None, None
                error = f"{type(exc).__name__}: {exc}"[:240]
                error_kind = "refusal" if isinstance(exc, (MathError, ProblemError)) else "exception"
            tool = tool[:128] if isinstance(tool, str) else None
            rows.append(_row(case, _project_result(result), tool, error, error_kind))
        relation_status, joint_correct = _relation(group, rows)
        evaluated.append({**group, "rows": rows, "relation_status": relation_status,
                          "joint_correct": joint_correct})
    assert_source_current()
    report = {"schema": SCHEMA, "config": {"seed": seed, "samples": samples},
              "scope": "finite_generated_metamorphic_suite", "calibrated": False,
              "router_id": "builtin_exact_tools" if router is None else "custom_unreplayable",
              "acceptance_policy": _ACCEPTANCE_POLICY, "source_hashes": _source_hashes(),
              "case_manifest_sha256": _hash(cases), "metrics": _metrics(evaluated),
              "by_family": _by_family(evaluated), "groups": evaluated,
              "limits": list(_LIMITS)}
    report["report_sha256"] = _hash(report)
    return report


def verify_audit(report: dict, *, replay: bool = True) -> dict[str, Any]:
    """Recompute stored checks; optionally rerun this exact built-in tool suite."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("unsupported metamorphic audit schema")
    if set(report) != _REPORT_KEYS:
        raise ValueError("metamorphic audit report fields mismatch")
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    if _hash(payload) != report.get("report_sha256"):
        raise ValueError("metamorphic audit fingerprint mismatch")
    try:
        if (report["acceptance_policy"] != _ACCEPTANCE_POLICY
                or report["limits"] != list(_LIMITS)):
            raise ValueError("metamorphic audit metadata mismatch")
        source_hashes = report["source_hashes"]
        if (not isinstance(source_hashes, dict) or set(source_hashes) != set(_SOURCES)
                or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in source_hashes.values())):
            raise ValueError("metamorphic audit source manifest mismatch")
        config = report["config"]
        if not isinstance(config, dict) or set(config) != {"seed", "samples"}:
            raise ValueError("invalid metamorphic audit config")
        cases = generate_groups(**config)
        groups = report["groups"]
        if not isinstance(groups, list) or len(groups) != len(cases):
            raise ValueError("metamorphic audit case manifest mismatch")
        if report.get("case_manifest_sha256") != _hash(cases):
            raise ValueError("metamorphic audit case manifest mismatch")
        for group, case_group in zip(groups, cases):
            if (not isinstance(group, dict) or set(group) != _GROUP_KEYS
                    or _hash({key: group[key] for key in case_group}) != _hash(case_group)
                    or not isinstance(group["rows"], list) or len(group["rows"]) != 2):
                raise ValueError("metamorphic audit case manifest mismatch")
            for case, row in zip(case_group["cases"], group["rows"]):
                if (not isinstance(row, dict) or set(row) != _ROW_KEYS
                        or row.get("error_kind") not in (None, "refusal", "exception")
                        or _hash(row["result"]) != _hash(_project_result(row["result"]))
                        or (row["tool"] is not None and
                            (not isinstance(row["tool"], str) or len(row["tool"]) > 128))
                        or (row["error"] is not None and
                            (not isinstance(row["error"], str) or len(row["error"]) > 240))
                        or ((row["error_kind"] is None) != (row["error"] is None))
                        or (row["error_kind"] is not None and row["result"] is not None)):
                    raise ValueError("metamorphic audit row mismatch")
                expected = _row(case, row["result"], row["tool"], row["error"], row["error_kind"])
                if _hash(row) != _hash(expected):
                    raise ValueError("metamorphic audit row mismatch")
            relation_status, joint_correct = _relation(case_group, group["rows"])
            if (group["relation_status"] != relation_status
                    or type(group["joint_correct"]) is not bool or group["joint_correct"] != joint_correct):
                raise ValueError("metamorphic audit relation mismatch")
        if (_hash(report["metrics"]) != _hash(_metrics(groups))
                or _hash(report["by_family"]) != _hash(_by_family(groups))):
            raise ValueError("metamorphic audit metrics mismatch")
        if report["router_id"] not in ("builtin_exact_tools", "custom_unreplayable"):
            raise ValueError("unsupported metamorphic router identity")
        if report["scope"] != "finite_generated_metamorphic_suite" or report["calibrated"] is not False:
            raise ValueError("unsupported metamorphic evidence scope")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed metamorphic audit report") from exc
    source_matches = report.get("source_hashes") == _source_hashes()
    if replay:
        if report["router_id"] != "builtin_exact_tools":
            raise ValueError("custom router reports cannot replay as the built-in tools")
        if not source_matches:
            raise ValueError("metamorphic audit source changed; replay not verified")
        if run_audit(**config) != report:
            raise ValueError("metamorphic audit does not replay")
    return {"integrity_verified": True, "case_manifest_verified": True,
            "source_matches": source_matches, "replay_verified": bool(replay),
            "scope": "finite_generated_metamorphic_suite", "report_sha256": report["report_sha256"]}
