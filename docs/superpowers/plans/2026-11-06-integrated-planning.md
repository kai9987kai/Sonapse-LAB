# Integrated Planning Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Track the owned tasks below.

**Goal:** Repair live learning boundaries and add bounded multi-step planning, honest experiments, and an interactive inspector.

**Architecture:** Executed transitions remain the only training source. A pure planner uses persistent bootstrap dynamics; runtime owns real transitions and final action bounds. Workbench previews and paired studies consume those interfaces.

**Tech Stack:** Existing Python, NumPy, PyTorch, pytest, vanilla JavaScript; CPU only.

**Spec:** `docs/superpowers/specs/2026-11-06-integrated-planning-design.md`

## Global Constraints

No vendor modifications, dependency changes, paid compute, publication, or push.
Default planner is off; default horizon 1. Uncertainty is heuristic.
Preserve prior checkpoints with unambiguous defaults; retain negative outcomes.

## Review Focus

- Invalid actions must fail before actor/world/physiology mutation.
- Manual and autonomous learning must consume the actual action exactly once.
- Frozen inference must preserve learned parameters, structural statistics, and topology.
- Topology restore must update executing buffers, graph metadata, and live optimizer parameters.
- Planner and preview must have bounded work, pure reads, finite outputs, and a baseline fallback.

### Task 1: Runtime and learning contracts

Files: `synapse/actions.py`, `agent.py`, `runtime.py`, `config.py`, `complex_world.py`, `homeostasis.py`, topology helpers, behavioral tests.

- [x] Write and run failing regressions for manual learning, duplicate physiology, drive/turn convention, fatigue after planning, frozen topology and changed-endpoint checkpoint replay.
- [x] Implement validated action boundary, consistent transition attribution, authoritative topology updates and complete restore.
- [x] Add `experience_horizon: int = 1` in [1,5]; dispatch horizon >1 through `ExperienceMPC(model, horizon).plan(state, baseline, memory)`.
- [x] Run targeted runtime/agent/experience/topology tests.

### Task 2: Pure experience MPC

Files: `synapse/experience_planner.py`, `synapse/experience.py`, `tests/test_experience_planner.py`.

Interface: `ExperienceMPC(model, horizon=3, beam_width=6, discount=.9, max_evaluations=256).plan(state, baseline, memory=None)` returns `(action, diagnostics)`. Diagnostics include candidate paths and bounded evaluation counts. `ExperienceWorldModel.predict_members(state, actions)` is a pure read. Dynamics gate uses pre-update delta error versus no-change baseline.

- [x] Write failing delayed-credit, member propagation, cold-start, invalid-input, work-budget, pure-read and restore tests.
- [x] Implement deterministic local action-sequence search with baseline retention, support/uncertainty gates, per-member trajectory propagation, and context held fixed.
- [x] Preserve horizon-one behavior and old RLS checkpoint import.
- [x] Run planner and existing experience tests.

### Task 3: Earlier-spec evidence restoration

Files: `synapse/capabilities.py`, relevant benchmark/evidence modules, tests, `docs/design-audit.md`.

- [x] Record current-source evidence for missing/unwired original commitments.
- [x] Reproduce literal RSI/evaluation responses, replace claimed measurements with actual bounded evaluations or explicit unmeasured status.
- [x] Run relevant capability and evidence tests; document scope remaining for standalone research demos.

### Task 4: Workbench and measured planning study

Files: `synapse/server.py`, `web/index.html`, `web/app.js`, `web/style.css`, `cli.py`, `planning_study.py`, tests.

- [x] Add failing tests for pure `/api/experience/preview`, horizon mode validation and atomic invalid requests.
- [x] Implement preview using current observation, previous executed action, and persistent experience models without acting or learning.
- [x] Add horizon controls, forecast visualization, learning-contract telemetry, correct teleop coordinates, and receipt-backed CLI study/verify commands.
- [x] Test shared-support paired queries, learned-state freeze, reproducibility and tamper detection.
- [x] Run actual-world development study, full tests, JS syntax and browser checks; write measured results and update README.

## Execution notes

Initial checkout: clean `main` at `d76b72e`. User explicitly requested all three
areas. Work uses supplied checkout with independent file ownership; no commit or
push requested. Parallel audit found actionable bugs before implementation.

The independent review found four additional regressions: shape-changing legacy
rollback, dormant-slot birth telemetry, nonfinite fallback diagnostics, and invalid
receipt step counts. Each was reproduced RED and fixed with behavioral regressions.
An additional root regression corrected missing horizon-one model-work telemetry.
Final consolidated suite: 322 passed in 117.12 seconds. Python compilation,
JavaScript syntax and diff whitespace checks passed. Actual-world pilot and browser
checks are recorded in the results report. No source dependency or vendor edits.
