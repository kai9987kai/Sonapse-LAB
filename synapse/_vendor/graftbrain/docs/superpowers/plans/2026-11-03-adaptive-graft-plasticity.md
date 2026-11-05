# Adaptive Graft Plasticity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Track work with the checkboxes below.

**Goal:** Add observable, checkpointed adaptive graft routing and bounded support adaptation.

**Architecture:** Independent plasticity module computes effective graft weights and changes active support only after learner updates. Agent owns its optimizer; runtime, CLI and viewer expose the same saved configuration. Paired experiments distinguish routing, rewiring and static wiring.

**Tech Stack:** Existing Python/PyTorch/NumPy/aiohttp; pytest for regression checks.

**Spec:** `docs/superpowers/specs/2026-11-03-adaptive-graft-plasticity-design.md`

## Global Constraints

- Base anatomy remains frozen; candidate endpoint pool stays fixed.
- Preserve existing modified checkpoint/journal; run tests in temporary paths.
- Compatibility default is `legacy`; existing optimizer layout stays unchanged.
- No superiority or biological-fidelity claims without supporting evidence.
- Reuse the current checkout on a feature branch; no worktree is needed.

## Review Focus

- Zero-site graphs and tiny directional buckets must remain finite/no-op.
- Dormant edges must have observable coactivity but no signal contribution.
- Snapshot flow must use effective masked/gated weights.
- Legacy loads and conflicting adaptive resume settings must be explicit.
- Frozen evaluation must leave all learned/plastic state unchanged.

### Task 1: Plasticity engine

Files: `graftbrain/plasticity.py`, `tests/test_plasticity.py`.
Interfaces: `PlasticityConfig(mode='legacy', density=.6, interval=10, warmup=2, turnover=.1, router_lr=1e-4)`; `GraftPlasticity(graph, obs_dim, config=None, seed=0)` exposes `gates(obs)`, `effective_weights(weights, gates=None)`, `observe(rates)`, `after_update(weights, optimizer, elig, heb)`, `regularization(gates)`, `summary()`, `site_stats()` and validated `state_dict/load_state_dict`.

- [x] Write and observe failing behavioral tests for neutral parity, context gradients, conserved directional budgets, dormant isolation, tiny/no-site handling, trace/moment resets, graph identity, deterministic restore.
- [x] Implement isolated RNG, deterministic support/score ordering, checkpointed masks/schedules/router, and immutable candidate identity.
- [x] Run `python -m pytest tests/test_plasticity.py -q`; require passing.

### Task 2: Neural learning and persistence integration

Files: `graftbrain/brain.py`, `graftbrain/runtime.py`, `tests/test_agent_plasticity.py`.
Interfaces: `Agent(..., plasticity_config=None)` owns `plasticity` and optional `router_opt`; `act(obs, cam, training=True)` supports frozen mean-action evaluation; `feedback(..., next_obs=None, next_cam=None)` accounts every transition; `reset_episode()` clears episodic state. Brain `forward(i_sens, graft_weights=None, adapt=True)` centralizes effective current.

- [x] Write failing tests for active masks in dynamics, router learning, transition accounting, terminal modulation, legacy load, adaptive restore and frozen evaluation.
- [x] Wire effective weights, router gradients, post-update structure, complete transition accounting and module persistence.
- [x] Extend Simulation configuration/resume validation and effective-flow snapshots; verify in temporary state paths.
- [x] Run `python -m pytest tests/test_agent_plasticity.py -q`; require passing.

### Task 3: CLI, experiment controls and evidence

Files: `graftbrain/__main__.py`, `graftbrain/experiments.py`, `graftbrain/web/index.html`, `tests/test_experiments.py`, docs and README/PAPER.
Interfaces: `benchmark_plasticity(out, specs, max_neurons, train_steps, tasks, seeds, eval_seeds, microsteps)` writes manifest/episodes/summary JSON; five modes use paired seeds and matched sparse masks.

- [x] Write failing tests for invalid CLI bounds and benchmark frozen-evaluation/artifact behavior.
- [x] Add CLI modes, config overrides, topology API support and viewer routing/structure telemetry.
- [x] Implement bounded paired experiment runner, explicit development/evaluation seeds and truthful evidence labels.
- [x] Run focused tests, full pytest suite, bounded CLI training/resume and actual CPU pilot; inspect output artifacts and viewer.
- [x] Update documentation with measured results and limitations; record verification.

### Task 4: Review and verification

- [x] Request independent final code review; resolve important findings with regression tests.
- [x] Run full pytest, oracle script, compile checks and diff checks; compare protected state hashes.
- [x] Leave reviewable changes on the feature branch. No publication or push is requested.

## Progress and rulings

- Baseline: oracle script prints 90 positive checks; pytest collects no tests. This is not evidence of learned task success.
- User approved design and implementation; proceed without another execution-method confirmation.
- Existing modified saved state stays untouched. Work remains uncommitted for user review unless subsequently requested.
- Completed engine, learner/runtime persistence, CLI, viewer, paired evidence tooling, and documentation.
- Independent review found partial-override resume rejection and zero-site NaN diagnostics; both reproduced and fixed with regression tests.
- Added legacy port-shape compatibility, final CLI rollout flushing, and pre-save journal-directory regression coverage.
- Final verification: 59 tests passed; compile, viewer syntax, oracle printout, and diff checks completed. See `docs/verification-adaptive-grafts.md` for the pilot and its negative/inconclusive task evidence.
- Original saved state hashes unchanged. Browser webview attachment failed; live HTTP/WebSocket and isolated DOM checks passed, visual verification remains unavailable.
