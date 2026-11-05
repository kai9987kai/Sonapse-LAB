# NEXUS-Ω adapter control assay with corrected observation semantics

**Date:** 13 October 2026  
**Status:** Closed-loop gate failed; adapter remains opt-in  
**Receipt:** [`runs/nexus_omega/adapter_schema_control_assay.json`](../../runs/nexus_omega/adapter_schema_control_assay.json)

## Why this rerun was needed

The first paired control assay ran before the safety planner's observation mapping was audited. The canonical `POMDPPhaseWorld` observation channel 4 is episode progress, not exposure. The planner had implicitly treated it as an exposure signal, creating a false late-episode threat trigger. `CausalEFEPlanner` now defaults `exposure_index` to `None`; callers with a real exposure channel can map it explicitly. This receipt is a fresh paired comparison with that corrected default and a new agent initialization.

## Protocol

The baseline and adapter candidate use the same agent initialization (`base_seed=44002`) and are paired on 64 fresh `POMDPPhaseWorld` seeds (74501–74564), with a 32-step horizon. The candidate loads the receipt-validated contextual residual adapter. The primary outcome is paired total reward; a 20,000-replicate seed-pair bootstrap estimates the interval. The exact sign test uses non-tied seed pairs. The predeclared gate requires the reward-delta 95% lower bound to exceed +0.05 and the survival-delta lower bound to exceed -0.02.

## Results

| Measure | Baseline | Adapter | Paired difference |
| --- | ---: | ---: | ---: |
| Mean episode reward | 6.2594 | 5.1883 | **-1.0711** (95% CI [-1.1633, -0.9688]) |
| Survival | 0/64 | 0/64 | 0.000 (95% CI [0, 0]) |
| Action-switch rate | 0.2789 | 0.3351 | +0.0562 |
| Online prediction MAE | 0.08534 | 0.08544 | descriptive; on-policy states differ |

There were six identical action paths. All 58 non-tied reward pairs favored the baseline (two-sided exact sign-test *p* = 6.94 × 10⁻¹⁸). Neither arm survived any episode, so equal survival is not evidence of a useful or safe policy. The closed-loop gate failed decisively; do not promote this adapter to the default planner.

## Interpretation and limits

Correcting the false exposure mapping does not rescue the adapter. It still worsens reward and increases action switching. This strengthens the rejection for this initialization and seed set, while the all-death outcome shows that the baseline itself is not a validated safe controller. The experiment covers one synthetic environment, one agent initialization, and 64 paired seeds; it does not establish generality across initializations or real systems. The adapter's separate one-step forecast admission result remains valid as a forecast-only result, but it cannot justify a control claim.

The earlier [control assay report](nexus-omega-adapter-control-assay-2026-10-13.md) and receipt preserve the original run. That run used the mistaken channel-4 exposure interpretation and is superseded for current control conclusions by this corrected-semantics assay. The adapter stays available only through explicit opt-in for research.
