# Compute, Training and Connectome Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Tasks have separate file ownership.

**Goal:** Preserve Antigravity's additions while connecting genuine training, verified connectomes and useful compute to the live workbench, then publish.

**Architecture:** An independently checkpointed residual neural graft and fixed sparse connectome prior supplement the existing actor. CPU batch training, source inventories and honest evaluation remain explicit.

**Tech Stack:** Existing Python/NumPy/SciPy/PyTorch; optional installed PyArrow for Feather assets; vanilla JavaScript.

**Spec:** `docs/superpowers/specs/2026-11-06-compute-training-connectome-design.md`.

## Global Constraints

Preserve Antigravity HEAD `ddf7406` and all current features. CPU-only installed runtime; no GPU-training claims. Original external data is read-only, raw datasets stay local, and at least 2 GB disk remains free. No force push.

## Review Focus

- Manual/governed actions must not be assigned invalid policy gradients.
- Frozen inference and preview reads preserve all learned state and RNG.
- Optional-module optimizers, buffers and RNG replay after checkpoint restoration.
- Dataset availability, type aggregation, neuron subsets and source identity are distinct.
- Long training tasks expose progress and bounded work without corrupting a live learner.

### Task 1: Data and connectome graft

Owner: connectome worker. Files: `connectome_assets.py`, `connectome_datasets.py`, `connectome_compiler.py`, new data tests/docs.

- [x] Verify real external MaleCNS assets and acquire public FlyWire bulk sources if feasible.
- [x] Add source inventory, portable verified ingestion and sparse neuron-level sampling.
- [x] Expose `load_programmed_graft(max_nodes=128,seed=7)` with edges, weights and provenance; never return an authentic proxy silently.
- [x] Test source absence, classified untyped records, sparse weights and truthful labels.

### Task 2: Neural graft and compute

Owner: training worker. Files: `neural_graft.py`, `active_training.py`, `compute.py`, `fast_weight_memory.py`, new tests/docs.

- [x] RED/GREEN zero-head identity, real updates, eligible actual rewards, pure inference and exact continuation.
- [x] Implement `propose(obs,features,baseline,learn=False)`, `observe(next_obs,reward,done,executed_action,receipt=None,manual=False)`, `train_replay(updates,batch_size,actor=...)`, export/import/summary and fixed prior binding.
- [x] Add actual cloned-transition pretraining and configurable CPU budgets with measured work.
- [x] Detach analytic fast-weight updates so trained agents remain cloneable.

### Task 3: Preserve Antigravity feature contracts

Owner: catchup worker. Files: dormant lifecycle, self model, Omnibus, research fabric, reusable skills/plasticity monitor, regressions/audit.

- [x] Repair frozen dormant mutation, optional checkpoint replay and skill summary.
- [x] Replace candidate-independent and mismatched evaluation with matched cloned-world screens; bind receipt integrity.
- [x] Make warmup/untrained/proxy evidence explicit and preserve callable feature APIs.

### Task 4: Live integration and product controls

Owner: root. Files: agent/runtime/config/server/CLI, visualizer/environment UI, integration tests/docs.

- [x] RED/GREEN configurable compute and graft integration with old checkpoint defaults.
- [x] Integrate actual transition receipts, bounded live fitting and explicit batched pretraining jobs.
- [x] Bind the verified fixed prior and source-aware live graph selection, preserving legacy modes.
- [x] Improve 3D source/role/activity/graft inspection, environments and current-state WSP/RSI controls.
- [x] Run full tests, browser flows, real training/paired pilots and independent review; commit and publish both remotes preserving histories.
