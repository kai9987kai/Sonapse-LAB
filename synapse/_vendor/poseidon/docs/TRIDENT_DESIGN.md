# TRIDENT: identity-gated directed residual control

TRIDENT asks a different question from TITAN. TITAN composed several software sidecars around the frozen Tidal policy. TRIDENT instead implements the staged research left open after the 9 October 2026 portfolio review: a history-conditioned residual comparator, directed travel constrained by observed macro-action transitions, and identity-specific residual controls.

## Control path

At each decision the controller reads the current 16-feature observation and the frozen Tidal policy action. It scores each resource- and reserve-admissible action by correcting Tidal's one-step dynamics prediction with an exponentially weighted residual of realized minus predicted observation change. Travel actions (`explore`, `flee`) are refused when the current terrain/shelter fingerprint has been visited enough times to leave discovery, yet that fingerprint-action pair has never been executed. Resource waypoint bonuses use replenishment rates estimated from return visits; sparse revisits abstain instead of assuming a fixed renewal constant.

An intact override must beat the incumbent by a margin **and** beat the same proposal scored under an erased residual and under a time-shifted residual. The live controller never sees simulator coordinates, seeds, or future labels. After TidePool executes the action, `observe_transition` records the true next observation, updates the residual, indexes the directed edge, and only then allows the next plan.

## Matched comparison

The six arms share each evaluation seed and reset controller state between episodes. The `no_topology` arm disables the directed-support filter, but still uses observed transition counts for destination-yield estimates; it is not a complete graph-memory ablation:

| Arm | Mechanism |
|---|---|
| Frozen Tidal core | Incumbent policy |
| TRIDENT intact | Residual identity gating and observed topology |
| Erased residual | Zero innovation; otherwise identical scoring |
| Shifted residual | Circular permutation of the residual vector |
| Directed-support gate off (`no_topology`) | Residual scoring without the directed-support filter; observed transition counts still inform return estimates |
| Residual gate off (`ungated`) | Residual proposal without the identity or override-margin checks |

The receipt records every action, observation, reward and decision telemetry. Verification rebuilds each arm from reset state and replays policy decisions and TidePool transitions. `promotion` is always false.

## What this is not

Identity specificity is a ranking comparison at one observation, not a causal identification theorem or a biological memory result. Observed topology is an episode graph over synthetic fingerprints, not a learned world model of a real environment. Residual correction uses the existing Tidal delta head; it does not train new neural weights. Device, quantum, fluid and ecological labels from earlier composites are intentionally absent.

## Research grounding

The design draws on two adjacent research directions while keeping its claims narrower. [Why Linear Recurrent Memory Works in Partially Observable Reinforcement Learning](https://proceedings.mlr.press/v306/zhao26j.html), published at ICML 2026, analyzes when linear recurrent filters can preserve belief information in particular HMM settings. TRIDENT's exponentially weighted transition residual is not a belief state and has no such sufficiency guarantee. [Revisiting Topological Graphs for Macro Action based Closed-loop Reinforcement Learning of Vision Language Navigation in Continuous Environment](https://arxiv.org/abs/2609.03906), posted in September 2026, studies learned graph-based macro-action navigation. TRIDENT uses only a small observed-transition support graph in a deterministic toy world; it does not reproduce that method or its navigation results.

See the [portfolio review](research/README.md) for the inspected project mechanisms and remaining research questions.
