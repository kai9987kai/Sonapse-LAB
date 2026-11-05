# NEXUS-Ω context-conditioned residual adapter

**Date:** 13 October 2026  
**Status:** Synthetic one-step forecast gate passed; separate closed-loop control gate failed  
**Receipt:** [`runs/nexus_omega/contextual_residual_assay.json`](../../runs/nexus_omega/contextual_residual_assay.json)

## Research question

Can a small observation-and-action-conditioned residual map generalize beyond the predictor initializations and environment seeds used to fit it, improving one-step next-observation forecasts over the raw factored ensemble?

## Protocol

The candidate is a ridge regression from the current 8-channel observation and 4-way action to the 8-channel next-observation residual. Each action has a separate intercept and set of observation coefficients. Observation normalization is fit only on the training partition. Data were gathered under a fixed open-loop action schedule in `POMDPPhaseWorld`; each model family saw the same environment seed and action sequence.

The train partition used 24 environment seeds and eight predictor initializations. Candidate family selection used 16 disjoint environment seeds and eight different initializations. The untouched holdout used 64 further environment seeds and eight additional predictor initializations not used in either fitting or selection. Selection compared no correction, action-only mean residual corrections, and a predeclared ridge/shrinkage grid. The primary measure was vital-channel MAE over health, energy, and hydration. The admission gate required the 95% environment-seed cluster-bootstrap lower bound on vital-MAE reduction to exceed 0.001, an exact paired sign-test *p* below 0.05, and the all-channel-MAE reduction lower bound above -0.002.

## Result

Selection chose **ridge λ = 10** and **shrinkage = 0.75**. On the untouched holdout:

- Vital MAE fell from **0.03174** to **0.02939**, a reduction of **0.00235** (95% environment-seed interval **[0.00164, 0.00302]**). The exact two-sided sign test was *p* = **1.01 × 10⁻⁷**, with improvement on 53 of 64 environment-seed averages.
- All-channel MAE fell from **0.08297** to **0.06701**, a reduction of **0.01597** (95% interval **[0.01430, 0.01761]**).
- The predeclared forecast gate passed. The candidate is retained in the receipt and can be loaded explicitly by `NexusOmegaAgent(residual_adapter_path=...)`.

## Runtime contract and limits

The optional loader verifies receipt SHA-256, schema, admission state, and source hashes. It rejects other observation/action/model dimensions. By default no adapter is loaded. When loaded, it corrects predicted next observations and updates the factual prediction cache so comparator residuals are computed against the forecast actually used. The ensemble's disagreement values are unchanged and remain uncalibrated.

The result is limited to one synthetic POMDP, one open-loop action schedule, and the tested predictor family. The model initializations were unseen during fitting/selection, but they share the same source architecture and random initialization procedure. It does not establish improved control, survival, safety, or real-world accuracy. A fresh 64-seed paired closed-loop assay with corrected observation semantics found lower reward with the adapter (-1.071 mean paired delta; 95% interval [-1.163, -0.969]), no survival change (0/64 survived in either arm), and more action switching. A subsequent exact same-state action-ranking audit on two fresh seed/initialization blocks found higher immediate regret and lower EFE-to-reward rank agreement with the adapter; vital-state error also increased on the random behavior anchor stream despite improved all-channel MAE. See the [corrected-semantics control report](nexus-omega-adapter-schema-control-assay-2026-10-13.md), [decision-rank audit](nexus-omega-decision-rank-audit-2026-10-13.md), and their receipts. The earlier control report preserves the superseded run. The adapter therefore remains opt-in, and forecast admission must not be treated as control evidence.
