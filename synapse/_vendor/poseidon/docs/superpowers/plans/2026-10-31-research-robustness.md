# Research Robustness Implementation Plan

> **For agentic workers:** Use regression-first implementation and independent
> review; tasks have separate file ownership and integrate in this checkout.

**Goal:** Repair exact numeric semantics and make related-input robustness and
fixed-policy episode risk inspectable through the workbench.

**Architecture:** Keep arithmetic parsing within the existing solver. Add two
independent research modules with replay/recomputation contracts, then expose
them through thin CLI, runtime, server and browser adapters.

**Tech Stack:** Existing Python, Fraction, SciPy, pytest and browser JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-31-research-robustness-design.md`

## Global constraints

- Reuse the current checkout on `codex/research-robustness-upgrade`.
- Preserve checkpoints and default control. No publication or model promotion.
- No new dependencies. Do not turn episode bounds into per-action probabilities.

## Review focus

- Exact decimals, exponent notation, excessive exponents and algebraic paths.
- Constant-wrong outputs that satisfy a relation but fail independent checks.
- Re-fingerprinted report mutations and custom-router replay identities.
- Reused episode identifiers, missing conditions, nonfinite/boolean parameters.
- Browser/CLI failure states and accurately scoped exports.

## Tasks

### 1. Exact numeric parsing

- [x] Reproduce decimal precision loss in each exact AST solver path.
- [x] Preserve numeric lexemes and reject unreasonable numeric budgets.
- [x] Run reasoning, advanced reasoning and problem solver regressions.

### 2. Metamorphic audit

- [x] Add failing tests for seed reproduction, related-input relations,
  constant-wrong outputs, refusals/errors and report mutations.
- [x] Implement `run_audit(seed=20261031, samples=8, *, router=None)` and
  `verify_audit(report, *, replay=True)` in `research/metamorphic_audit.py`.
- [x] Run generated cases against the actual tools and replay them.

### 3. Episode risk screen

- [x] Test known exact bounds, condition correction, disjoint identifiers,
  strict JSON and mutated/resealed report rejection.
- [x] Implement protocol creation, auditing and report recomputation in
  `research/episode_risk.py`, including raw binary episode outcomes.
- [x] Verify both eligible and insufficient-evidence fixtures without promotion.

### 4. Workbench integration and validation

- [x] Test runtime/API and CLI entry points before adding adapters.
- [x] Add metamorphic and episode-screen reports/export to the browser.
- [x] Document primary research, practical usage and evidence limits.
- [x] Request independent combined review and fix actionable findings.
- [x] Run full Python and Node suites, real HTTP/browser probes, audit replay
  and source-package parity; record actual results in the upgrade document.

## Completion record

All tasks completed locally. Final v3.14.0 Python suite: 1,451 passed; Node harness:
50 passed; real browser exports replayed/recomputed; all 193 wheel source/asset
files matched; isolated wheel audit replayed. See the upgrade document for the
scope and generated evidence paths. The user subsequently requested GitHub and
Hugging Face publication; that follow-up prepares v3.14.0 without promotion.
