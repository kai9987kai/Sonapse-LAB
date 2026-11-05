# NEXUS-Ω decision-rank audit

**Date:** 13 October 2026  
**Status:** Diagnostic replicated across two fresh seed/initialization blocks; no controller promotion  
**Receipts:** [`decision_rank_audit.json`](../../runs/nexus_omega/decision_rank_audit.json), [`decision_rank_audit_replication.json`](../../runs/nexus_omega/decision_rank_audit_replication.json)

## Why this experiment

The contextual residual adapter passed a one-step forecast gate on its declared holdout but lost reward in closed-loop control. This audit asks a narrower question: on identical agent-observed states, does the adapter improve the planner's action ordering against outcomes the synthetic environment can reveal by exact counterfactual branching?

This follows the portfolio's useful separation of retrieval/ranking from outcome evidence, paired interventions, and admission from performance. The implementation is original to Poseidon: [NexusSearch](https://github.com/kai9987kai/NexusSearch), [Causeway](https://github.com/kai9987kai/Causeway), and [MOLT](https://github.com/kai9987kai/MOLT) inform filtered ranking and controlled branches; [TESSERA](https://github.com/kai9987kai/TESSERA) informs keeping evidence admission distinct from promotion. Their reported results are not Poseidon evidence.

## Protocol

For each of 64 synthetic `POMDPPhaseWorld` episodes, a seeded uniform-random behavior stream generated anchor states. The baseline and adapter agents had the same initialization within each seed and received the same observations, behavior actions, and realized transitions. At every anchor, the evaluator deep-copied the complete environment and executed all four actions from the identical state and random-generator position. The copy corresponding to the executed behavior action was checked against the real transition, including reward, terminal state, death, and next observation.

The agents saw only observations and completed behavior transitions; hidden simulator fields were used only by the offline branch evaluator. The primary diagnostic is **immediate reward regret**: the best reward among four one-step branches minus the reward of the agent's proposed action. Other diagnostics compare each action's EFE score and predicted reward with the four actual immediate rewards using Kendall's tau-b. This is not a multi-step value estimate or a closed-loop performance assay.

Two fresh blocks used different environment seeds and agent initializations. Both yielded complete branches with zero lockstep mismatches; all 64 behavior episodes in each block ended in death before the 32-step horizon. The random behavior stream supplies an analysis distribution and is not a survival baseline.

## Results

| Block | Anchors | Mean regret: baseline → adapter | Adapter − baseline (95% seed-bootstrap CI) | EFE/reward tau-b: baseline → adapter |
| --- | ---: | ---: | ---: | ---: |
| Seeds 88101–88164, agent 49001 | 1,104 | 0.1001 → 0.1107 | **+0.01061** [0.00610, 0.01565] | 0.4907 → 0.4391 |
| Seeds 88201–88264, agent 49002 | 1,089 | 0.0821 → 0.0900 | **+0.00785** [0.00347, 0.01276] | 0.4950 → 0.4571 |

The adapter increased immediate regret in both blocks. EFE score agreement with branch reward also fell in both blocks; the paired tau-b changes were **−0.0516** (95% CI [−0.0686, −0.0360]) and **−0.0379** (95% CI [−0.0502, −0.0263]). These intervals resample the 64 environment-seed summaries within each block; the two blocks are reported separately and are not pooled as independent model-family replications.

The per-action reward head was unchanged by the state-only adapter, so predicted-reward MAE and rank correlation were identical within each block. The state adapter showed a split-channel effect on the behavior-action transition:

| Block | Vital-state MAE: baseline → adapter | Paired change (95% CI) | All-channel MAE: baseline → adapter | Paired change (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| First | 0.02456 → 0.03611 | **+0.01156** [0.00964, 0.01346] | 0.08139 → 0.07689 | −0.00450 [−0.00654, −0.00250] |
| Replication | 0.01841 → 0.03139 | **+0.01299** [0.01128, 0.01479] | 0.07780 → 0.07371 | −0.00409 [−0.00607, −0.00207] |

Thus, lower all-channel error concealed worse prediction of the three vital channels used in pragmatic scoring. The pragmatic score's rank agreement also fell alongside total EFE, while predicted reward remained fixed. This points to the corrected vital-state estimates as a likely source of the action-ranking loss; it is a mechanism-level diagnostic, not a formal causal mediation result.

## Consequence

The adapter remains opt-in and is not suitable for default activation. Its forecast admission was distribution-specific: on these fresh, policy-independent behavior anchors it worsened vital prediction and action ranking even as aggregate observation error fell. Future adapter admission should retain per-channel and per-action metrics and include decision-ranking or regret measurements on the intended action/state distribution. A multi-step, policy-induced-state evaluation with a viable survival baseline is still needed before making control claims.

The implementation adds complete per-action score and prediction telemetry to `CausalEFEDecision`, displays the uncalibrated action ranking in the Poseidon panel, and records complete branch outcomes in the receipts. This makes the planner's alternatives inspectable without exposing privileged state to the agent or changing its policy. The experiments remain synthetic and limited to one agent initialization per block.
