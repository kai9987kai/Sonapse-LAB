# AURA Biomimetic Evaluation Results v0.8

This document records the empirical results of the multi-arm paired AURA evaluation study across identical seeds, horizons, and TidePool environmental dynamics without altering the frozen Tidal DAgger policy weights.

---

## 1. Study Specification & Artifacts

- **Experiment Schema**: `poseidon-aura-experiment-v1`
- **Experiment ID**: `d9e0d76bd4812c037152636768f0bf8d`
- **Receipt SHA-256**: `b489e9702cff856c1a25535e4845a82e5fc1f6b17b8f248f4801a691981f3995`
- **Artifact Path**: `outputs/aura_experiments/d9e0d76bd4812c03-b489e9702cff.json`
- **Seeds Evaluated**: `[150000001, 150000002, 150000003, 150000004]` (4 paired seeds)
- **Episodes per Arm**: 4 (20 total episodes across 5 arms)
- **Max Horizon**: 64 steps
- **Environmental Scarcity**: 2.5

---

## 2. Summary Outcomes by Arm

| Controller Arm | Episodes | Survival Rate | Mean Reward | Mean Steps | Mean PVA Coherence ($R$) | Optomotor Triggers | Description |
|---|---|---|---|---|---|---|---|
| `policy` | 4 | **100.0%** | **6.5569** | 64.0 | 1.0000 | 0 | Frozen selected Tidal DAgger core |
| `odysseus` | 4 | **100.0%** | **6.5569** | 64.0 | 1.0000 | 0 | Empirical Bayesian replenishment navigator |
| `aura` | 4 | 25.0% | -0.0905 | 46.75 | 0.8374 | 0 | Full AURA (CX-ring + Neuropil + Tessera) |
| `aura_no_ring` | 4 | **100.0%** | **6.5569** | 64.0 | 1.0000 | 0 | Ring attractor disabled |
| `aura_no_neuropil` | 4 | **100.0%** | **6.5569** | 64.0 | 0.8408 | 0 | Sparse neuropil arbiter disabled |

---

## 3. Findings & Evidence Disclosures

1. **Ablation Separation**:
   - Disabling the ring attractor (`aura_no_ring`) or disabling the neuropil arbiter (`aura_no_neuropil`) restores survival to 100% and reward to 6.5569, demonstrating that the behavioral change stems directly from descending homeostatic drive arbitration interactions.
2. **Optomotor Heading Stability**:
   - Across all 4 seeds, ring attractor bump coherence remained stable ($R = 0.8374 \ge 0.35$), requiring 0 emergency optomotor stabilization triggers.
3. **Safety & Default Policy**:
   - As required by the project's empirical principles, availability does not imply superiority. The frozen Tidal DAgger core policy remains the default selected controller. AURA is strictly an opt-in experimental biological research controller.
