# Residual State Prediction Pilot Results — 12 October 2026

## 1. Executive Summary

This report documents the completed, matched CPU prediction experiment investigating whether a learned residual neural correction improves upon the action-conditioned affine linear state predictor in synthetic TidePool dynamics.

The experiment matched data budgets, update counts (1,024 updates), batch size (128), and model dimensions (five 128-unit ensemble members) between the unanchored baseline (`mlp-v1`) and the anchored candidate (`linear-residual-v1`). Both architectures were fitted and frozen on disjoint partitions before fresh test evaluations were conducted.

**Findings:**
1. **Unanchored MLP severely degrades state prediction:** Relative to the training-only action-conditioned linear baseline, `mlp-v1` suffers a large, statistically significant loss in state-delta MSE: paired difference of $-0.00729$ (95% CI $[-0.00848, -0.00613]$) on in-distribution (ID) test and $-0.00824$ (95% CI $[-0.00939, -0.00714]$) on out-of-distribution (OOD) scarcity-shifted test.
2. **Linear anchor prevents degradation:** The `linear-residual-v1` architecture completely prevents this collapse by anchoring onto the frozen affine predictor and learning a residual initialized at zero. On the selection split, shrinkage selected a residual multiplier of $\alpha = 0.5$ (MSE $0.0044396$ vs $\alpha=0$ MSE $0.0044419$).
3. **Residual gain is statistically indistinguishable from zero:** On held-out test splits, the learned residual produces an empirical gain of $+3.72 \times 10^{-7}$ (ID) and $+1.19 \times 10^{-6}$ (OOD), but 95% bootstrap confidence intervals span zero: $[-3.72 \times 10^{-6}, +4.59 \times 10^{-6}]$ and $[-1.17 \times 10^{-6}, +3.68 \times 10^{-6}]$.
4. **Reward prediction shows reliable positive gain:** Both architectures achieve statistically significant reward prediction gains over the training action-mean baseline: $+0.00752$ MAE gain ($[0.00453, 0.01093]$) for linear-residual, and $+0.00716$ ($[0.00405, 0.01071]$) for MLP.
5. **Decision:** Under the preregistered acceptance criterion requiring a strictly positive 95% lower bound for state gain over the linear baseline (`lower_bound > 0`), both candidates are **rejected**. Neither model is promoted. The frozen Tidal core policy remains the default runtime controller.

---

## 2. Experimental Partitions & Protocol

The run protocol was archived prior to training in `outputs/residual-pilot-652000000/protocol.json` (SHA-256 bound). Seeds were partitioned into disjoint million-seed namespaces:

| Partition | Inclusive Seed Range | Episodes | Purpose |
|---|---|---:|---|
| Train | 652000000–652000127 | 128 | Training ensemble members & affine baseline |
| Selection | 653000000–653000031 | 32 | Checkpoint selection & residual shrinkage grid search |
| Calibration | 654000000–654000063 | 64 | Split-conformal calibration quantiles |
| ID Test | 655000000–655000063 | 64 | Fresh in-distribution prediction evaluation |
| OOD Scarcity Test | 656000000–656000063 | 64 | Fresh scarcity 2.5 prediction evaluation |

Both architectures used identical seeds and episode trajectories across all five partitions.

---

## 3. Training & Model Selection

Both models completed 1,024 AdamW updates using batch size 128 across 5 ensemble members. Training elapsed times were matched:
- `linear-residual-v1`: 34.17 seconds
- `mlp-v1`: 34.36 seconds

### Residual Shrinkage Selection (`linear-residual-v1`)
On the 32 selection episodes, state-delta MSE across the residual scale grid was:
- $\alpha = 0.00$ (pure linear anchor): $0.0044419$
- $\alpha = 0.25$: $0.0044404$
- $\alpha = 0.50$: **$0.0044396$** (selected)
- $\alpha = 1.00$: $0.0044403$

The grid search confirmed a shallow minimum at $\alpha = 0.50$, selecting that scaling factor for test evaluation.

---

## 4. Test Evaluation Results

All metrics below represent paired comparisons on 64 fresh test episodes per condition, with 2,000 bootstrap replicates for 95% confidence intervals.

### In-Distribution (ID) Test (Seeds 655000000–655000063)

| Metric | `linear-residual-v1` | `mlp-v1` | Comparison Baseline |
|---|---:|---:|---|
| Candidate State MSE | 0.0049038 | 0.0121938 | Linear baseline: 0.0049042 |
| Paired State Gain over Linear | **+3.72e-07** | **-0.00729** | Positive is better |
| 95% Bootstrap CI (State Gain) | [-3.72e-06, +4.59e-06] | [-0.00848, -0.00613] | Spans zero for residual |
| Candidate Reward MAE | 0.05779 | 0.05815 | Action-mean baseline: 0.06531 |
| Paired Reward Gain over Baseline | **+0.00752** | **+0.00716** | Positive is better |
| 95% Bootstrap CI (Reward Gain) | [+0.00453, +0.01093] | [+0.00405, +0.01071] | Strictly positive |
| Candidate Death Brier Score | 0.01025 | 0.01034 | Prior prevalence: 0.00892 |
| State Episode Coverage (90% target) | 78.1% (half-width 0.548) | 75.0% (half-width 0.542) | Nominal coverage 90% |

### Out-of-Distribution (OOD Scarcity 2.5) Test (Seeds 656000000–656000063)

| Metric | `linear-residual-v1` | `mlp-v1` | Comparison Baseline |
|---|---:|---:|---|
| Candidate State MSE | 0.0051216 | 0.0133579 | Linear baseline: 0.0051228 |
| Paired State Gain over Linear | **+1.19e-06** | **-0.00824** | Positive is better |
| 95% Bootstrap CI (State Gain) | [-1.17e-06, +3.68e-06] | [-0.00939, -0.00714] | Spans zero for residual |
| Candidate Reward MAE | 0.04934 | 0.04994 | Action-mean baseline: 0.05624 |
| Paired Reward Gain over Baseline | **+0.00691** | **+0.00631** | Positive is better |
| 95% Bootstrap CI (Reward Gain) | [+0.00409, +0.01021] | [+0.00334, +0.00974] | Strictly positive |
| Candidate Death Brier Score | 0.01258 | 0.01269 | Prior prevalence: 0.01014 |
| State Episode Coverage (90% target) | 90.6% (half-width 0.638) | 87.5% (half-width 0.632) | Nominal coverage 90% |

---

## 5. Formal Gate Assessment

The automated `assess_prediction_gate()` evaluation produced:

```json
{
  "state": "rejected",
  "promoted": false,
  "reasons": [
    "id_test.state_gain_over_linear: lower_bound_does_not_meet_requirement",
    "ood_scarcity_test.state_gain_over_linear: lower_bound_does_not_meet_requirement"
  ]
}
```

Both models failed the mandatory gate requirement that the lower 95% bootstrap bound on state prediction gain over the linear baseline must be strictly greater than zero (`lower_bound > 0`).

---

## 6. Scientific Interpretation & Implications

1. **Affine Dominance in TidePool Physics:** The 16-D normalized TidePool observation dynamics are overwhelmingly affine with respect to actions and state deltas. An unregularized or lightly regularized ridge model captures almost all predictable variation. Deep neural networks without strong inductive bias overfit local variations and perform significantly worse on fresh held-out trajectories.
2. **Residual Head Efficacy & Limits:** The residual head architecture successfully prevents the severe $-0.007$ MSE degradation seen in pure MLPs by anchoring to the linear predictor. However, the residual network learns negligible true non-linear signal: the selected $\alpha=0.5$ scaling produces an empirical gain of only $+3.7 \times 10^{-7}$ in ID and $+1.2 \times 10^{-6}$ in OOD, indistinguishable from noise.
3. **Reward Signal is Learnable:** In contrast to state transitions, reward functions in TidePool exhibit non-linear patterns that the neural ensemble reliably captures, providing a statistically significant $+0.007$ MAE improvement over action-mean prediction.
4. **Safety & Invariance:** By enforcing strict preregistered acceptance gates and full cryptographic replay, the system prevents false promotion of underperforming models. The active control policy remains the tested and verified frozen Tidal DAgger checkpoint.
