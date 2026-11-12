# Prequential Residual Calibration for Experience MPC

**Date:** 11 November 2026  
**Status:** Implemented as an empirical one-step uncertainty-scaling mechanism; task-level benefit is untested.

## Why this change

The experience model already records reward and state-delta errors before it
fits each completed transition. Its planner used recent RMSE, ensemble spread,
and covariance leverage as heuristics, without measuring whether those scales
were wide enough after a change in observed dynamics. This addition uses that
same pre-update feedback to adapt the scale applied to one-step planning.

Adaptive conformal inference (ACI) is a published online method that adjusts a
miscoverage level from whether each prior interval missed; Gibbs and Candès
describe its long-run coverage objective under changing data distributions
([NeurIPS 2021 paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)).
Their later work studies a dynamically tuned step size for local-interval
regret under arbitrary distribution shifts
([JMLR 2024 paper](https://www.jmlr.org/beta/papers/v25/22-1218.html)). Sonapse
uses the ACI update as an engineering design pattern; this implementation does
not inherit either paper's theorem.

## Mechanism

For each completed transition, before any RLS fitting, compute the joint
normalized residual

```text
score = max(
    abs(reward - predicted_reward) / max(previous_reward_RMSE, 0.001),
    RMS(actual_delta - predicted_delta) / max(previous_delta_RMSE, 0.001)
)
```

The prior errors are actual executed outcomes. Imagined planner rollouts do not
enter the calibration stream. Scores are capped at 1,000 and the controller
keeps at most 256. Once it has 24 scores, the next forecast uses a finite-sample
upper order statistic at the current miscoverage level. Following an observed
miss `e_t` (1 for a miss, 0 for a hit), the level updates as

```text
alpha[t+1] = clip(alpha[t] + 0.02 * (0.10 - e[t]), 0.001, 0.999)
```

The resulting empirical threshold multiplies the model's one-step reward and
delta RMSE terms. The multiplier is bounded to `[0.25, 100]`; before the 24
calibration observations it is exactly 1, preserving the prior uncertainty
scale. The 1-step planner and multi-step MPC both consume this scale. Their
ensemble disagreement, covariance support, and multi-step forecasts remain
heuristic. New counters, scores, outcomes, and adaptive level are checkpointed;
legacy checkpoints without the calibrator restore into a cold-start instance.

## Controlled mechanism check

Run from the repository root:

```powershell
python scripts/audit_experience_calibration.py --output docs/results/2026-11-11-prequential-calibration.json
```

The audit sends one identical seeded score stream to a fixed 90% normal
threshold and to the adaptive calibrator. It uses 800 absolute standard-normal
scores, followed by 800 absolute normal scores with twice the scale. The
receipt is [2026-11-11-prequential-calibration.json](../results/2026-11-11-prequential-calibration.json).

| Phase | Fixed threshold miss rate | Adaptive threshold miss rate | Adaptive threshold, last 100 mean |
|---|---:|---:|---:|
| Stationary scale | 8.89% (69 / 776) | 10.05% (78 / 776) | 1.662 |
| Scale shift | 42.00% (336 / 800) | 9.88% (79 / 800) | 3.339 |

This deterministic result shows the controller widened its empirical interval
on this scalar scale-shift stream. It does not show an agent learned better,
that predictions are calibrated conditionally by state/action, or that task
returns or safety improved. The next empirical step is a matched-seed planner
evaluation that reports baseline returns, action changes, abstentions, runtime,
and one-step joint miss rates, with the old fixed-scale planner as a control.

The exact script digest, seed, runtime versions, thresholds, and phase counts
are recorded in the JSON receipt. The controller tests also cover shift
response, deterministic checkpoint continuation, legacy cold-start loading,
and invalid-checkpoint atomicity.
