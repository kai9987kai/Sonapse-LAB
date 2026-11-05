# TITAN: transition-grounded composite controller

TITAN asks a narrow engineering question: can several existing Poseidon sidecar mechanisms be composed around the frozen Tidal policy while keeping state resets and reward attribution auditable in the synthetic TidePool environment?

## Control path

At each decision, the controller reads the current 16-feature observation, updates its simulated sidecars, and proposes one of TidePool's six integer actions. Causeway's amplitudes and collapse are computed in classical software. AURA, the lattice, NexusFlow, and other selected modules contribute heuristic signals; these signals are not learned from physical measurements.

After TidePool executes the action, `rollout_with_observed_transitions` calls `observe_transition` with the pre-action observation, executed action, returned reward, next observation, and environment info. The controller rejects a missing or mismatched pending action, an invalid reward, or an invalid next observation. Only then does it append the observed transition to Morpheus's trace, credit observed reward to Genesis's auxiliary fitness counter, and index the rewarded event in NexusSearch. Decision-time reward is not fabricated from the planner's potential scores.

The planner refuses a second decision while a transition is pending. Code that drives TITAN must therefore use the transition-aware rollout path. The runtime's `world --planner titan`, workbench experiment, benchmark runner, and verifier use that path.

## Reset and matched comparison

`reset(seed=...)` clears the controller sidecars between episodes, including Tessera's ratification memory and Genesis's ecology, generation, fitness, and chromosome. Genesis derives its starting chromosome deterministically from the low 32 bits of the episode seed. TidePool and the comparison arms share each benchmark seed; stateful controller sidecars reset before each arm and episode.

The six arms are the frozen core, AURA, METAMORPH, CHIMERA, HYPERION, and TITAN. The benchmark records each action, before/after observations, reward, resulting world state, final snapshot, and decision telemetry. Its receipt is hashed and bound to the Tidal checkpoint and TidePool source. The verifier rebuilds each arm from its reset state and replays every policy decision and world transition, then recomputes arm summaries, paired deltas, and TITAN telemetry.

## Interpretation and limits

This is a reproducibility and integration instrument, not a model-training or model-promotion procedure. It has no independent-rater design, held-out environment family, confidence-bound promotion gate, or external replication. The included four-seed run is a small descriptive pilot. Telemetry names such as "quantum", "trophic", "buoyancy", "intermittent", and "lineage" label classical software calculations or synthetic state; they do not indicate hardware or biological phenomena.

The Tidal checkpoint and default planner remain unchanged. No fitted artifact is produced, and every receipt sets `promotion` to false. See [TITAN results](TITAN_RESULTS.md) for the exact configuration and recorded summary.
