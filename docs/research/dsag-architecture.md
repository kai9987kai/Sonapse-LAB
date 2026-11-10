# DSAG rate-kernel v1

Dual-Soma Axodendritic Graft (DSAG) is an opt-in **synthetic continuous-rate** residual module. It adds two synchronously updated soma state vectors, independently routed inputs that enter along a directional axon chain, bounded terminal release, versioned packet sidecars, optional phase gating, optional three-factor local updates, and reversible bounded branch edits.

This is not a membrane-voltage model, an action-potential simulator, or a biological claim. The axon initial segment has a distinct initiation and polarity role in real neurons; DSAG's along-axon receiving branches are an engineered computational motif. A biological compartmental simulator remains a separate research task. See [Jenkins and Bender](https://journals.physiology.org/doi/full/10.1152/physrev.00030.2024) and the [Brian 2 multicompartment reference](https://brian2.readthedocs.io/en/2.7.0/user/multicompartmental.html).

## Defaults and opt-in

`DSAGSpecV1()` is disabled with gain zero. The legacy path does not instantiate or register the DSAG channel. Use the CLI `--dsag` option or load `configs/dsag/rate_pilot.yaml` to opt in. The rate pilot is a configuration, not a claim of measured benefit. Gain begins at zero and follows its configured ramp. `observe_only` advances the shadow state while emitting exact zeros.

The canonical SHA-256 spec digest covers every dataclass field, including topology, numeric coefficients, feature flags, target ports, seed, and parent reference. The model topology digest additionally binds host width, effective output targets, live branch map, and topology version.

JSON config loading uses the standard library. YAML configs require the optional dependency installed with `python -m pip install -e '.[yaml]'`.

## Dynamics and interfaces

The two soma equations use the same previous-tick soma values for cross-coupling. Branch and segment updates use bounded `tanh` rates, deterministic clocks, finite delay histories, batched tensor state, and no RNG draws during stepping. Branches address one active axon segment. Without supplied phase data, the phase gate is neutral. NeuroWave-X2 packets retain their existing immutable wire bytes; `DSAGLocalEnvelope` is an internal, checksummed adapter object carrying the original packet checksum, routing destination, age, TTL, phase, validity, and provenance.

`DSAGBusChannel` implements the existing `NeuralGraftProtocol` and participates once in `NeuralGraftCommunicationBus`. The bus remains the sole owner of final current clipping. DSAG never adds current after the cap.

`advance=False` evaluates without mutating recurrent state, ramp, packet queue, metrics, or RNG. `adapt=False` prevents eligibility and weight updates. Packet duplicates, stale topology versions, out-of-order sequences, non-finite values, malformed destinations, and queue overflow are rejected. Checkpoints use an isolated `DSAGCheckpointV1` namespace with a canonical state digest, spec digest, parent commit, source digest, tensor inventory, recurrent state, histories, queue, RNG, topology, and lineage. Legacy checkpoints without DSAG remain on the legacy config and loader path.

## Learning and topology

Three-factor updates require an explicit modulation vector delivered by a routed envelope; generic context or absent packet metadata is not treated as a reward or neuromodulator. Eligibility and branch-to-segment weights are bounded. Branch add, prune, and move operations require `structural_growth_enabled`, enforce configured limits, refuse edits with packets pending, and return an exact rollback receipt. Other proposed operators and sleep/spiking modes are intentionally outside this rate-kernel implementation.

## Minimal run

```powershell
python -m synapse dsag validate --config configs/dsag/default_off.yaml
python -m synapse dsag smoke --seed 42 --steps 24
python -m synapse --nodes 16 --dsag --dsag-gain 0.05 run --steps 24
```

These commands validate wiring and synthetic fixtures only. They do not test task-level generalization or justify promotion.
