# NEXUS-Ω residual adapter: paired closed-loop control assay

**Date:** 13 October 2026  
**Status:** Historical run; channel-4 semantics were incorrect. See the corrected-semantics rerun.  
**Receipt:** [`runs/nexus_omega/adapter_control_assay.json`](../../runs/nexus_omega/adapter_control_assay.json)

## Question and protocol

Does the contextual residual adapter that improved one-step forecasts improve the decisions made by the NEXUS-Ω agent? The candidate and baseline used the same agent initialization (`base_seed=44001`) and ran on the same 64 `POMDPPhaseWorld` seeds, with a 32-step horizon. The baseline used no adapter; the candidate loaded the receipt-validated adapter. The primary outcome was paired total episode reward. Survival was a guardrail. Per-episode bootstrap intervals resampled seed pairs; the exact sign test used non-tied seed pairs.

Before execution, the control gate required the lower bound of the paired 95% reward-difference interval to exceed +0.05 and the lower bound of the survival difference to exceed -0.02. A forecast-only gain was not sufficient for default promotion.

## Results

| Measure | Baseline | Adapter | Paired difference |
| --- | ---: | ---: | ---: |
| Mean episode reward | 5.1617 | 4.7938 | **-0.3680** (95% CI [-0.4344, -0.3016]) |
| Survival rate | 0/64 | 0/64 | 0.000 |
| Action-switch rate | 0.2105 | 0.3136 | +0.1031 |
| Mean online prediction MAE | 0.08366 | 0.08372 | descriptive, on-policy states differ |

All 47 non-tied reward pairs favored the baseline (two-sided exact sign-test *p* = 1.42 × 10⁻¹⁴); 17 pairs tied. Neither arm survived to the horizon in any episode. The survival guardrail alone passed because there was no difference, but the reward criterion failed substantially. The action-switch rate also increased. The contextual adapter therefore **did not pass the closed-loop control gate** and must not become the default planner path.

## Interpretation

This original run is retained for provenance, but it incorrectly let the safety planner interpret canonical observation channel 4 (episode progress) as exposure. Do not use it as the current corrected-semantics estimate. The fresh rerun on seeds 74501–74564 and agent initialization 44002 still finds a substantial reward loss and increased action switching; see the [corrected-semantics report](nexus-omega-adapter-schema-control-assay-2026-10-13.md) and [receipt](../../runs/nexus_omega/adapter_schema_control_assay.json).

The adapter remains available only as an explicit research option. A follow-up should diagnose action churn and require multiple fresh agent initializations, new environment seeds, a competitive survival baseline, and a predeclared noninferiority or superiority criterion before any default change. This historical receipt records the completed run; its digest does not prove execution by itself.
