# Graft Credit and Maturation Implementation Plan

> **For agentic workers:** Use test-first implementation with independent focused workers and a final independent review. Checkboxes track completion.

**Goal:** Recruit dormant grafts using learning-gradient evidence and protect their integration time, with matched evaluations.

**Architecture:** Extend the existing plasticity module without changing its compatibility defaults. Capture detached upstream effective-weight derivatives during backward; finalize scores once per learner update. Separate empirical tools compare policies and controlled support disruption.

**Tech stack:** Existing CPU Python/PyTorch/NumPy/aiohttp; pytest; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-11-03-graft-credit-design.md` (approved).

## Global constraints

- Preserve current process 4272/port 8770 and all live checkpoint writes; no pause, restart, or live control commands.
- Keep previous uncommitted work and original checkpoint/journal intact; use the current feature branch.
- Defaults: growth_rule=coactivity, growth_init=original, min_age=0.
- Preserve candidate identity and exact directional budgets; no base-anatomy or learner algorithm change.
- Do not claim general task benefit, novelty, biological fidelity, or exact mid-rollout replay.

## Review focus

- Dormant upstream derivatives must bypass the mask in scoring but never in signal or raw-weight gradient.
- Hook callbacks must accumulate signed detached derivatives; EMA updates occur once after backward.
- Age protection, missing score evidence, and empty buckets must reduce/no-op swaps without budget drift.
- Old checkpoints and explicit CLI overrides must preserve old behavior and reject real conflicts.
- Balanced pre/post evaluation and support interventions must not change the source agent or share action RNG.

### Task 1: Gradient scoring, maturation and migration

Files: `graftbrain/plasticity.py`, `tests/test_graft_credit.py`.

Interfaces: extend `PlasticityConfig` with `growth_rule`, `growth_init`, `min_age`. `GraftPlasticity` adds `capture_gradient(effective, gates)`, `begin_gradient_update()`, `finish_gradient_update()`, `after_update(weights, optimizer, elig, heb, allow_rewire=True)`, `perturb_support(weights, optimizer, elig, heb, fraction, seed)` and persistent `edge_age`, `gradient_ema`, `gradient_update_count`. Existing methods remain compatible. Intervention counters stay separate from learned swaps. `summary()`/`site_stats()` expose scores, ages, protection and cohorts.

- [x] Write failing finite-difference/dormant-gradient, loss-ranking, age/budget, zero-init, private-RNG, checkpoint migration and next-decision tests.
- [x] Implement gradient collection/EMA, optional selection rules, age protection, intervention and validated migration.
- [x] Run focused tests; require all existing plasticity tests still pass.

### Task 2: Agent, CLI and viewer integration

Files: `graftbrain/brain.py`, `graftbrain/__main__.py`, `graftbrain/web/index.html`, integration tests and docs.

Interfaces: `_forward` captures effective-weight hook only in training/grad scope; `_update` brackets backward with begin/finish and passes `rewiring_enabled` to after_update. Agent saves/restores `rewiring_enabled` (default true). CLI `--graft-growth`, `--graft-growth-init`, `--graft-min-age` map to config fields, preserving only-explicit conflict checks.

- [x] Write failing integrated score/frozen-state, legacy migration, CLI bounds/partial-resume and control-flag tests.
- [x] Implement hook lifecycle and controls; add graceful unavailable diagnostics for the still-running old backend.
- [x] Run integration tests and actual viewer/API verification using an isolated new server only.

### Task 3: Balanced matched credit benchmark

Files: `graftbrain/credit_experiments.py`, `tests/test_credit_experiments.py`.

Interface: `benchmark_graft_credit(out, specs=None, max_neurons=400, train_steps=384, tasks=None, seeds=None, eval_seeds=None, microsteps=1)` returns output directory; `validate_credit_benchmark(...)` validates without writes. `evaluate_credit(agent,graph,scenarios,train_seed,microsteps,phase)` evaluates frozen independent clones and returns rows with scenario/condition. Seven controls: coactivity/gradient/random × age0/20, plus routing-only age0, all zero-init with matched initial mask, 16-step learner rollouts.

- [x] Write failing balanced scenario, unchanged-policy zero delta, source-state preservation, seed separation, matched config/support, source-hash and overwrite refusal tests.
- [x] Implement predeclared balanced scenarios, zero-action references, pre/post measurements, per-seed paired deltas, event traces and exact source/library fingerprints.
- [x] Run focused tests and bounded real CPU pilot; preserve all outcomes.

### Task 4: Controlled recovery probe

Files: `graftbrain/graft_probe.py`, `tests/test_graft_probe.py`.

Interface: `run_graft_probe(out, seeds=(0,1,2), train_steps=80, recovery_steps=80)` writes JSON artifacts and returns directory. Use the real GraftPlasticity engine in a small supervised candidate-current problem with a declared target switch. Clone source into sham, disturbed+rewiring and disturbed+no-rewire; disruption preserves directional support budgets and independently resets changed traces/moments. Record before/immediate/after error, complete learning curves, support events and source-state digests. This is a synthetic supervised probe, never an RL task-success claim.

- [x] Write failing real-gradient, intervention-budget/source-preservation, deterministic replay and complete artifact tests.
- [x] Implement the probe; run tests and a real bounded multi-seed probe.

### Task 5: Review and evidence

- [x] Run full pytest, compile and JS syntax checks; request fresh independent final review and repair important findings with regressions.
- [x] Record source hashes, measured outcomes and limitations in a verification report; update README/PAPER with prior-art links and exact commands.
- [x] Confirm original hashes and live-process continuity. Leave work uncommitted; no publication requested.

## Progress

- Prior baseline: 59 tests pass. Live backend and browser site table verified; no errors observed.
- User explicitly approved design and implementation; no further execution-method approval is needed.
- Completed engine, learner/CLI/viewer integration, balanced benchmark and recovery probe.
- Full verification: 129 tests passed; Node known-event/legacy-snapshot renderer checks, JS syntax and Python compilation passed.
- Independent review: fixed oversized probe seeds before directory creation and added actual last-event selection telemetry. No unresolved important finding remains; numerical dormant credit independently checked against a three-microstep brain finite difference.
- Recorded balanced seven-arm pilot: zero paired RL return improvement. Recorded three-seed supervised probe: lower adaptive final current MSE versus matched fixed support in all three seeds, with full curves and explicit target-switch limits.
- Computational fingerprints stable; original saved-state hashes unchanged; process4272/port8770 continued. Isolated test server stopped. New-panel browser attachment unavailable, backend and text-render checks passed.
- Ruling: restrict next-growth replay claims to complete update boundaries under identical subsequent inputs/RNG; mid-rollout world recovery is outside scope, as clarified in the approved spec.
