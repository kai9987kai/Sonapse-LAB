# Compute, Neural Graft Training, and Verified Connectome Pilot Report

**Date:** 7 November 2026  
**Raw Summary Receipt:** [`docs/results/2026-11-07-compute-training-connectome-summary.json`](2026-11-07-compute-training-connectome-summary.json)  
**Related Specifications:** [`docs/training-graft.md`](../training-graft.md) · [`docs/connectome-data.md`](../connectome-data.md) · [`docs/antigravity-audit.md`](../antigravity-audit.md)

---

## 1. Parallel Pretraining Pilot (Cloned-World Rollout Workers)

Executed `pretrain_runtime` with `steps=128`, `updates=32`, `batch_size=64`, `workers=2`, and `compute_threads=4` on `forage` (`seed=7`) with the 128-neuron MaleCNS sparse prior bound:

| Metric | Value |
|---|---|
| Rollout workers (`spawn`) | `2` (`PID 19092`, `PID 5860`) |
| Worker world seeds | `100010`, `200013` |
| Recorded executed transitions | `128` |
| Optimizer updates | `32` |
| Policy / actor updates | `0` (`policy_frozen: true`) |
| Initial sensory + reward prediction loss | `0.073937` |
| Final sensory + reward prediction loss | `0.005502` (**−92.56%**) |
| Final TD value loss | `0.001434` |
| Worker collection CPU time | `2.8125 s` |
| Session CPU time / wall time | `3.3438 s` / `19.9181 s` |

Auxiliary pretraining substantially reduces sensory-summary and reward prediction error on real executed transitions while keeping the residual actor output head at zero (`policy_frozen: true`).

---

## 2. Paired 4-Arm Frozen-Query Evaluation

Compared four matched controller configurations across `seeds = [7, 11, 19]` and `tasks = ['forage', 'push']` (`64` support steps with `learn=True`, followed by `48` frozen query steps with `learn=False` on held-out query seed `seed + 5003`):

| Arm | Description | Mean Query Return | Mean Diff vs Baseline |
|---|---|---:|---:|
| `baseline_actor` | Existing coupled actor (`neural_graft=False`) | `-0.299415` | — |
| `graft_pretrained_frozen_policy` | Pretrained auxiliary/value heads (`64` steps, `16` updates); `active_training=False` | `-0.305174` | `-0.005759` |
| `graft_online_training` | Pretrained auxiliary + online clipped policy & TD critic updates (`24` opt / `8` actor updates) | `-0.413968` | `-0.114552` |
| `graft_online_malecns_prior` | Online graft + 128-neuron MaleCNS sparse prior (`1,442–1,758` edges) + source-backed substrates | `-0.375928` | `-0.076513` |

### Per-Seed / Per-Task Breakdown

| Task | Seed | Query Seed | `baseline_actor` | `graft_pretrained_frozen_policy` | `graft_online_training` | `graft_online_malecns_prior` | Prior Edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| `forage` | 7 | 5010 | `+0.243555` | `+0.217548` | `+0.234702` | **`+0.497463`** (`+0.253908`) | 1,585 |
| `forage` | 11 | 5014 | `-0.226643` | `-0.226572` | `-0.457793` | `-0.557301` (`-0.330658`) | 1,442 |
| `forage` | 19 | 5022 | `-0.601925` | `-0.601927` | `-1.145564` | `-0.967548` (`-0.365624`) | 1,758 |
| `push` | 7 | 5010 | `-0.499550` | `-0.507890` | `-0.526655` | `-0.510833` (`-0.011283`) | 1,585 |
| `push` | 11 | 5014 | `-0.262023` | `-0.262307` | **`-0.192045`** (`+0.069978`) | `-0.240734` (`+0.021289`) | 1,442 |
| `push` | 19 | 5022 | `-0.449906` | `-0.449896` | **`-0.396451`** (`+0.053456`) | `-0.476615` (`-0.026708`) | 1,758 |

**Scientific Interpretation:**
- Individual seed/task pairs show localized improvements (e.g., `graft_online_malecns_prior` improves `forage` seed 7 by `+0.2539`, and `graft_online_training` improves `push` seeds 11 and 19 by `+0.0700` and `+0.0535`), and adding the MaleCNS prior improves mean return over the unconditioned online graft (`-0.3759` vs `-0.4140`).
- However, across this short 64-step support budget, neither online graft variant outperforms `baseline_actor` on average (`-0.1146` and `-0.0765`). Short on-policy exploration and 8 clipped policy updates are insufficient to establish a reliable cross-task control advantage. Negative and neutral differences are retained in full.

---

## 3. Bounded Live Recursive Self-Improvement (`run_live_rsi`)

Screened `2` stochastic motor-weight candidates on `forage` (`24` steps per world across query and retention seeds, `96` world steps per screen):
- Candidate 1: `gain = 0.0000`, `retention = -1.4965` vs `-1.4965` $\to$ rejected (`gain_or_retention_screen_failed`).
- Candidate 2: `gain = 0.0000`, `retention = -1.4970` vs `-1.4965` $\to$ rejected (`gain_or_retention_screen_failed`).
- Parent digest `15ee30eb5971e63b...` remained bit-for-bit identical after rejecting non-improving candidates.
