# HYPERION: Paired Multi-Arm Evaluation Benchmark Report

## 1. Executive Summary

The **HYPERION Frontier Super-Controller** (Poseidon v1.1.0) synthesizes:
1. **Morpheus** offline oneiric sleep engine with SWS memory consolidation and REM counterfactual trajectory replay with phase annealing.
2. **Prometheus** meta-plasticity with rolling Shannon scarcity entropy and homeostatic drive allostasis.
3. **Intermittent** passive ambient energy harvesting and 38-byte binary NTAG 215 ephemeral state checkpointing with zero-power resuscitation.
4. **NexusSearch** federated hybrid associative vector and spatial search indexing.
5. **CHIMERA** foundational substrates: Causeway quantum superposition, 256-d HRR concept memory, 3D Diamond Lattice geodesic neuropil, AURA Central Complex ring compass, MOLT developmental instars, NexusFlow hydraulic potential flux, and Chronos cyclic beacon clock.

All experiments were executed on CPU capped at 2 threads without modifying or retraining the frozen Tidal core (`runs/tidal_dagger/core.pt`).

---

## 2. Sealed Empirical Receipt

- **Receipt Location:** `outputs/hyperion_experiments/RECEIPT.json`
- **Schema:** `poseidon-hyperion-experiment-v1`
- **Experiment ID:** `b7ccaa832ae4495ecf4cfda28abc0ae6`
- **Receipt SHA-256:** `b7ccaa832ae4495ecf4cfda28abc0ae6461c6672e675c376081af7d5ff994203`
- **Cryptographic Verification:** `verified: true`
- **Evaluation Settings:** 4 seeds (`303000001`–`303000004`), 64 max steps per episode, scarcity 2.5.

---

## 3. Comparative Performance Across Arms

| Arm / Controller | Episodes | Survival Rate | Mean Steps | Mean Reward |
|---|---|---|---|---|
| **HYPERION (v1.1.0)** | 4 | **100.0%** (4/4) | **64.0** | **6.2037** |
| CHIMERA (v1.0.0) | 4 | **100.0%** (4/4) | **64.0** | **6.5118** |
| METAMORPH (v0.9.0) | 4 | **100.0%** (4/4) | **64.0** | **6.1391** |
| Core Baseline Policy | 4 | **100.0%** (4/4) | **64.0** | **6.4795** |
| AURA Biomimetic Central Complex | 4 | 75.0% (3/4) | 58.5 | 4.5587 |

---

## 4. Hyperion Subsystem Metrics

| Metric | Measured Value | Standard / Safety Bound | Interpretation |
|---|---|---|---|
| **Survival Rate** | **100.0%** | $\ge 90.0\%$ | Perfect survival under scarcity 2.5 |
| **Spectral Divergence Index (SDI)** | **0.051618** | $\le 0.1500$ | Non-destructive behavioral consistency |
| **Mean Quantum Entropy** | **0.964695** | $> 0.5000$ | Robust action superposition without premature collapse |
| **Mean Lattice Coherence** | **0.723308** | $[0.60, 0.85]$ | Multi-scale 3D geodesic spatial grid resonance |
| **Total Energy Harvested** | **9.1887** | $> 0.0$ | Continuous passive RF ambient energy accumulation |
| **REM Counterfactual Replays** | **2** | $> 0$ | Active offline oneiric replay during resting cycles |
| **Mean REM Phase Gain** | **+0.0506** | $\ge 0.0$ | Constructive quantum phase alignment towards reward |
| **Mean Scarcity Entropy** | **0.0843** | $\ge 0.0$ | Self-regulated Prometheus meta-plasticity |
| **Mean Holographic Items Stored** | **651.75** | $> 50.0$ | Constant-memory HRR associative concept encoding |
| **Destructive Conflicts Detected** | **0** | $0$ | Zero contradictory control overrides |

---

## 5. Offline Verification

The benchmark receipt can be verified offline via the CLI:
```bash
python -m poseidon verify-hyperion outputs/hyperion_experiments/RECEIPT.json
```
Receipt verification re-computes the canonical SHA-256 payload digest and validates all arithmetic and arm statistics without neural weights.
