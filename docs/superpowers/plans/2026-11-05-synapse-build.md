# SYNAPSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Track checkboxes and report real tests.

**Goal:** Build the approved integrated connectome/Poseidon/Expanse workbench with bounded weight edits, inherited RSI, source-backed capabilities, replay and usable local UI.

**Architecture:** A standalone Python package vendors immutable upstream source snapshots and imports an actual Tidal checkpoint. A shared actor connects recurrent graft activity and trained Tidal features bidirectionally. A transaction layer, finite operator language and experimental meta-controller evaluate object and improver changes separately. Local HTTP services expose live execution and bounded jobs.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, SciPy, standard-library HTTP, vanilla HTML/CSS/JavaScript, pytest.

**Spec:** ../specs/2026-11-05-connectome-weightspace-design.md

## Global Constraints

- Preserve original source projects and checkpoints; work in this new checkout.
- Default graph is synthetic, CPU-first and provenance-labeled; no biological/general intelligence claims.
- Supported architecture families execute through typed services with actual result consumers; missing artifacts remain explicitly unavailable.
- Fixed evaluator and sealed query data are outside the improver mutation surface.
- Exact replay claims require complete state, model and source identity. All timings and negative results remain recorded.
- No paid compute, publication or live camera activation.

## Review Focus

- Stale/nonfinite/oversized edits must fail without partial mutation (Task 3).
- Failed/cancelled jobs must restore a consistent checkpoint and expose errors (Task 5).
- Shared model sessions must not leak state across tests/requests (Tasks 2, 5).
- Query labels and final evaluations must not enter proposal selection (Task 4).
- Missing upstream artifacts and incompatible schemas must be visible rather than substituted silently (Tasks 1, 6).

## File structure and interfaces

`synapse/config.py`: frozen `SynapseConfig(seed=7,nodes_per_source=100,microsteps=2,cognition_interval=8,readout='absolute',task='forage')`, `ROOT`, `VENDOR`, `ARTIFACTS`.
`synapse/sources.py`: activate pinned bundled upstream imports, inventory/hashes.
`synapse/agent.py`: `IntegratedAgent(config)`, `act(obs, camera=None, learn=False)->dict`, `reset_episode()`, `export_state()->dict`, `import_state(dict)`, `editable_tensors()->dict[str,Tensor]`, `status()->dict`. Action in result is 5-element NumPy array; result contains JSON-compatible telemetry and `features` NumPy array.
`synapse/runtime.py`: `LabRuntime(config, output_root=None)`, `step(steps=1, learn=False)->dict`, `snapshot()->dict`, `save(name)->dict`, `load(name)->dict`, `reset(task=None,seed=None)->dict`. Owns a GraftBrain World, agent, bounded histories; exposes `agent` for transactional edits. Snapshot is JSON-compatible.
`synapse/stateio.py`: `digest(value)->str`, `save_state(path,payload)->str`, `load_state(path)->dict`; tensor/primitive-only archives, typed NumPy round-trip and atomically written manifests.
`synapse/patches.py`: `make_patch(agent, edits, description='')->dict`, `apply_patch(agent,packet)->dict`, `rollback(agent,receipt)->None`, `evaluate_patch(agent,packet,evaluator)->dict`; validated tensor edits with exact parent and bound budgets. Keep checkpoint sidecars outside UI payloads.
`synapse/learning.py`, `synapse/rsi.py`, `synapse/experiments.py`: standalone bounded training/edit-policy study API `run_study(output_dir, config:dict, progress=None, cancelled=None)->dict`, `verify_study(path)->dict`; policy inheritance, support/query separation, replayable receipts.
`synapse/capabilities.py`: `catalog()->dict`, `invoke(capability_id,payload)->dict`, returns evidence scope, structured result and downstream consumption trace.
`synapse/server.py`: loopback API/UI, one serialized runtime; persistent bounded job records with cancellation.
`synapse/cli.py`, `synapse/web/*`: status/run/train/serve/study/verify/capabilities/patch commands and local dashboard.

### Task 1: Reproducible foundation
- [x] Snapshot local GraftBrain/Poseidon and pinned Expanse source; copy selected Tidal artifact, licenses, manifests. Inventory every upstream Python file and entry point.
- [x] Add package/config/state archive contracts and validate safe round-trip, tamper rejection and installation.
- [x] Commit foundation.

### Task 2: Integrated acting network
- [x] Write failing tests for bidirectional coupling, finite motors, memory acquisition/reversal, exact restore and no updates during evaluation.
- [x] Implement actual GraftBrain rate graph + active plasticity and Tidal checkpoint with trained semantic/intent bridge interfaces, persistent memory, compass, curiosity and predictor traces.
- [x] Repair fresh MushroomBody eligibility in isolated code; add Expanse-style dormant experts, native consolidation and absolute/evoked/persistent-evoked branches with measurable participation.
- [x] Implement world runtime and preparation training, explicit resource semantics, all nine native task backends and state restart semantics.
- [x] Run targeted tests and bounded headless episodes. Record behavior limitations.

### Task 3: Weight programs and transactions
- [x] Test valid/stale/nonfinite/out-of-range patches, rollback, evaluator rejection and shape contracts.
- [x] Implement bounded typed edits, named surfaces, exact fingerprints and inactive candidate evaluation. Add packet export/import and seeded local numeric operators with author provenance.
- [x] Test all changed-state restoration and write actual example packet/receipt.

### Task 4: Learning, RSI and scientific experiments
- [x] Test support/query isolation, inherited successor proposals, actor reset, shared archive controls and receipt tampering.
- [x] Build preparation/retention benchmarks using actual integrated graph/Tidal feature extraction, gradient and direct numeric edit arms, donor/graft lesions and bounded self-modifying improvement policies.
- [x] Provide fixed, nonrecursive, recursive and random meta arms; record every proposal, rejection, parent, cost and generation. Frozen query scoring does not select successors.
- [x] Execute a bounded paired pilot and strict replay; report negative outcomes without promotion.

### Task 5: Local workbench and CLI
- [x] Build live graph/world view, neural routing, memory, capabilities, patch editor/preview, studies, lineage, checkpoints and job progress/cancel/error handling.
- [x] Provide loopback-only JSON endpoints with bounded inputs, structured errors, no arbitrary path access and a serialized runtime.
- [x] Test APIs, restart/checkpoint behavior, CLI help, browser interactions and meaningful empty/failure states.

### Task 6: Complete upstream capability integration
- [x] Expose audited typed service operations for every supported architecture family plus full source inventory; generate runtime invocation schemas from actual signatures.
- [x] Wire capability results into shared episodic/tool memory and explicitly selected planning profiles. Do not claim every historical module executes in each action.
- [x] Execute coverage sweep with actual outputs, artifact-unavailable outcomes and per-entry consumers; resolve unsupported imports/dependencies where feasible.

### Task 7: Validation and delivery
- [x] Run full new suite, source integrity checks, package/wheel validation, CLI/API/browser smoke and experiment replay.
- [x] Independent architecture/code review, fix material findings, rerun affected checks.
- [x] Write README, model/data cards, capability coverage, experiment results and limitations; commit completed code and launch the loopback workbench.

## Execution rulings

The user explicitly said “build it now full featured and complete”; this authorizes implementation without another plan confirmation. Independent files may be delegated in parallel, with fixed interfaces and final integration review. The existing empty workspace is the standalone project; no new worktree is necessary. Unavailable upstream pretrained artifacts are reported honestly and are not substituted with fabricated trained output.
