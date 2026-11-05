# Homeostatic guidance: held-out one-step forecast assay

**Date:** 13 October 2026  
**Status:** Completed synthetic benchmark; guidance failed the preregistered forecast-improvement criterion  
**Receipt:** [`runs/nexus_omega/guidance_forecast_assay.json`](../../runs/nexus_omega/guidance_forecast_assay.json)

## Question and design

Does the Stage 10 homeostatic state adjustment improve the factored model's one-step predictions against realized transitions? This is a forecast-accuracy question, not a policy or survival-benefit question.

The assay ran 32 fixed `POMDPPhaseWorld` seeds and eight fixed, randomly initialized predictor ensembles. Each environment/predictor pair followed the same 32-step open-loop action schedule for every guidance arm. At each transition, the model predicted from the current observation and action; each adjusted prediction was scored against the next observation actually returned by the environment. The arms used guidance scales -1.25, 0, 0.625, 1.25, and 2.5. Zero is the sham control; -1.25 is the opposite-direction control. The primary comparison was vital-channel (observation indices 0-2) mean absolute error at scale 1.25 versus zero. Predictor initializations were averaged within environment seed; uncertainty intervals and the exact sign test use the 32 environment seeds as clusters. Other scales and channel summaries are descriptive.

The policy and environment were not changed between arms. Guidance adjusted only the three channels explicitly identified as health, energy, and hydration. The earlier index-4 “exposure” adjustment was removed: the canonical POMDP defines index 4 as episode progress, so applying an exposure penalty there was a semantic error. Unnamed observation channels now remain untouched unless a future schema explicitly identifies them.

## Results

| Guidance scale | Vital MAE | All-channel MAE | Change in vital MAE vs sham |
| ---: | ---: | ---: | ---: |
| -1.25 (opposite) | 0.05138 | 0.09127 | +0.02240 |
| 0 (sham) | **0.02898** | **0.08287** | reference |
| 0.625 | 0.03941 | 0.08679 | +0.01043 |
| 1.25 (primary) | 0.05421 | 0.09234 | +0.02523 |
| 2.5 | 0.08546 | 0.10405 | +0.05647 |

At the primary scale, mean absolute error increased by 0.02523 (95% environment-seed cluster-bootstrap interval for sham minus guided: [-0.02688, -0.02355]). All 32 environment-seed averages favored sham; the two-sided exact paired sign test gives p = 4.66e-10. These numbers support a narrow conclusion: on this synthetic transition distribution and these untrained predictor initializations, the fixed guidance adjustment made one-step vital-state forecasts less accurate. Larger positive guidance produced larger error. No guidance scale in this small sweep improved the measured forecast metric.

## Interpretation and limits

The predictor is a randomly initialized ensemble, not a trained or artifact-bound model. The interval and sign test generalize over the declared environment-seed sampling while conditioning on these eight fixed predictor initializations; they do not estimate variation over all possible initializations. The result does not establish that guidance harms a trained model, that no other guidance rule can help, or that forecast accuracy alone determines control quality. The open-loop schedule and the POMDP are synthetic. The primary test was selected before execution; the neighboring-scale comparisons are descriptive and should not be treated as additional confirmatory tests. The receipt preserves per-seed and per-initialization cells, protocol, environment/model seeds, source hashes, and a SHA-256 digest for auditability; it cannot prove that execution occurred as recorded.

This result is a reason to keep guidance out of macro selection and out of any safety claim. A next-stage control study would need a trained, frozen predictor and a separate untouched seed set, comparing raw, guided, sham, and a reactive policy on realized reward, survival, and forecast error. Guidance parameters must be fixed before that holdout is opened.
