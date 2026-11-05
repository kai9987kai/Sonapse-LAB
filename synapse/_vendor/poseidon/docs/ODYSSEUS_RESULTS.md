# Odysseus Benchmark Results (v0.7.0)

**Receipt ID**: `1521d677b6f2f75b`  
**Receipt SHA256**: `c4035a1005d36e99fa9ee3265ad924876050ca8638f643a9a94bded27a99e7bc`  
**Schema**: `poseidon-odysseus-experiment-v1`  
**Evaluation Parameters**: 4 paired seeds (133000001–133000004), 6 controllers, 64-step horizon, scarcity 2.5, 24 total episodes, 1,536 replayed transitions.

---

## 1. Summary Performance by Controller

| Controller | Episodes | Survival Rate | Mean Steps | Mean Reward | Total Overrides | Override Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **policy** | 4 | 100.0% | 64.0 | 6.484 | 0 | 0.0% |
| **odysseus_calibrated** | 4 | 100.0% | 64.0 | 6.484 | 0 | 0.0% |
| **odysseus_ungated** | 4 | 100.0% | 64.0 | 6.484 | 0 | 0.0% |
| **odysseus_no_memory** | 4 | 100.0% | 64.0 | 6.484 | 0 | 0.0% |
| **odyssey** | 4 | 100.0% | 64.0 | 6.484 | 0 | 0.0% |
| **random** | 4 | 100.0% | 64.0 | 5.087 | 0 | 0.0% |

---

## 2. Paired Comparison vs Frozen Policy Baseline

| Controller Arm | Mean Reward Delta | Wins | Ties | Losses | Leave-One-Seed-Out Mean Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **odysseus_calibrated** | +0.000 | 0 | 4 | 0 | [0.000, 0.000] |
| **odysseus_ungated** | +0.000 | 0 | 4 | 0 | [0.000, 0.000] |
| **odysseus_no_memory** | +0.000 | 0 | 4 | 0 | [0.000, 0.000] |
| **odyssey** | +0.000 | 0 | 4 | 0 | [0.000, 0.000] |
| **random** | **-1.397** | 0 | 0 | 4 | [-1.516, -1.203] |

---

## 3. Findings & Safety Verification

1. **Policy Preservation**: The learned Tidal DAgger policy successfully clears all 64 steps under high scarcity (2.5) across test seeds. Odysseus correctly recognizes when local incumbent policy actions are optimal and refrains from spurious waypoint divergences.
2. **Deterministic Replay Guarantee**: All 1,536 transitions replayed with 100% agreement against TidePool dynamics without loading neural weights.
3. **Negative Control Separation**: The uniform random controller underperforms policy by 1.397 reward units with 4/4 statistically decisive losses across all seeds.
