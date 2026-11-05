# NEXUS-Ω action-conditioned residual calibration gate

**Date:** 13 October 2026  
**Status:** Holdout showed a small vital-channel gain; admission gate rejected it; runtime inactive  
**Receipt:** [`runs/nexus_omega/residual_calibration_gate.json`](../../runs/nexus_omega/residual_calibration_gate.json)

## Experiment

The prior held-out assay found that fixed homeostatic nudges increased forecast error. This follow-up asked whether a simpler, data-derived correction could remove systematic bias without changing the predictor weights. For each of eight fixed NEXUS factored-ensemble initializations, the candidate stores a per-action mean residual vector (`observed next state - predicted next state`). A single shrinkage factor multiplies that vector at prediction time.

The evidence partitions were disjoint: 24 environment seeds fitted the residual vectors, 16 selected shrinkage from `{0, 0.25, 0.5, 0.75, 1}`, and 64 untouched environment seeds evaluated the frozen choice. Each episode used the same predetermined open-loop action schedule. Predictor initializations were averaged within environment seed. The primary metric was one-step MAE over health, energy, and hydration. The all-channel MAE was a guardrail. Bootstrap intervals resampled environment seeds, and a paired exact sign test counted environment-level primary outcomes.

Before opening the holdout, admission required all of the following: selected shrinkage above zero; lower bound of the 95% cluster-bootstrap interval for vital-MAE reduction greater than 0.001; exact sign-test *p* below 0.05; and lower bound of the all-channel-MAE reduction interval above -0.002. The script binds every correction to the exact predictor initialization weight digest. It does not alter predictor parameters or activate a runtime path.

## Results

The selection partition chose shrinkage **0.5** (mean vital MAE 0.03153; the zero-correction arm scored 0.03276). On untouched seeds, raw vital MAE was **0.03164** and corrected vital MAE was **0.03087**, a reduction of **0.000768**. The 95% environment-seed bootstrap interval was **[0.000277, 0.001227]**; 49 of 64 environment-seed averages favored correction, with two-sided exact sign-test *p* = **2.44 × 10⁻⁵**.

The all-channel MAE changed from **0.08324** to **0.08316**, a reduction of **0.000080** with 95% interval **[-0.000359, 0.000531]**. This is practically neutral at the declared 0.002 guardrail.

The result **did not pass admission**: although the primary interval excludes zero and the sign test is significant, its lower bound (0.000277) is below the preregistered minimum useful reduction (0.001). The gate therefore rejects the correction. Its residual vectors are retained in the receipt for inspection, but nothing is wired into planning or forecasts.

## Context-conditioned follow-up and opt-in runtime path

A second experiment replaced the per-action constant with a small ridge map from the current eight-channel observation and action to the next-state residual. It trained on 24 environment seeds and eight predictor initializations, selected ridge/shrinkage on 16 disjoint seeds and eight different initializations, then evaluated once on 64 new environment seeds and eight unseen initializations. The selected setting was ridge 10 with shrinkage 0.75. On holdout, vital MAE fell from 0.03174 to 0.02939 (reduction 0.00235; 95% environment-seed interval [0.00164, 0.00302]); all-channel MAE fell from 0.08297 to 0.06701 (reduction 0.01597; 95% interval [0.01430, 0.01761]). The admission rule passed on this synthetic forecast assay. Full selection and holdout rows are in [`contextual_residual_assay.json`](../../runs/nexus_omega/contextual_residual_assay.json), with the protocol in [its report](nexus-omega-contextual-residual-assay-2026-10-13.md).

This admitted, model-family-level adapter is now available as an explicit opt-in through `NexusOmegaAgent(residual_adapter_path=...)`. Loading checks the receipt digest, schema, admission result, and source hashes; the default agent remains unmodified. The adapter updates the factual prediction cache so subsequent comparator residuals correspond to the corrected forecast. Its ensemble uncertainty values remain uncalibrated. The holdout establishes only a one-step forecast result on the named synthetic distribution, not control benefit; closed-loop evaluation on fresh seeds is still required before making it a planner default.

## Limits and next step

This is evidence about one synthetic POMDP, these eight fixed randomly initialized predictors, and one open-loop action schedule. It is not a result for trained models or realized policy outcomes. A future candidate would need a materially larger predeclared improvement, independent model initializations or a frozen trained artifact, and then a separate control evaluation on fresh seeds. The holdout should not be reused to tune this correction. Receipt hashing and source/model identities support replay audits but cannot prove the recorded run occurred.
