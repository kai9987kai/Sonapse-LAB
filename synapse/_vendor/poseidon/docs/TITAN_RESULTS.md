# TITAN matched-seed pilot results

This report matches the replay-verified release receipt at `outputs/titan_experiments/RECEIPT.json`. It is a small synthetic-environment measurement, not evidence that TITAN is a better general-purpose controller.

## Configuration and integrity

| Setting | Value |
|---|---|
| Profile | `titan-six-arm-paired-replay-v2` |
| Environment | TidePool `tidepool-v1` |
| Scarcity | 2.5 |
| Horizon | 64 steps per episode |
| Seeds | 99000000–99000003 |
| Arms | Core, AURA, METAMORPH, CHIMERA, HYPERION, TITAN |
| TITAN decisions | 256 across four episodes |
| Receipt canonical SHA-256 | `0de9949f57162a0d5ec133fddf47c00d9a81a7f45d8f0e6187fd5119b0340925` |
| Receipt file SHA-256 | `34ec27bcae3c602aa938c4c34731e733de26d20e668b3357953b69f47f19692e` |
| Tidal checkpoint SHA-256 | `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27` |
| TidePool source SHA-256 | `f64b06b9434827b8ad6ce84212d8a7c62648abefa70661283de8b281c0acdbee` |

All six controllers were run once per shared seed. The receipt records each action, observation, returned reward, next observation, world state, and final snapshot. Verification rebuilt each controller and replayed the policy decisions and TidePool transitions. The bundled release smoke check reports `verified: true`; `promotion` is false.

## Descriptive outcomes

| Controller | Mean reward | Population SD | Survival | Mean steps |
|---|---:|---:|---:|---:|
| Frozen Tidal core | 6.4783 | 0.1148 | 100% (4/4) | 64.00 |
| AURA | 5.9836 | 0.4187 | 100% (4/4) | 64.00 |
| METAMORPH | 6.0458 | 0.5567 | 100% (4/4) | 64.00 |
| CHIMERA | 6.3234 | 0.1773 | 100% (4/4) | 64.00 |
| HYPERION | 6.1590 | 0.1764 | 100% (4/4) | 64.00 |
| TITAN | 6.1590 | 0.1764 | 100% (4/4) | 64.00 |

The mean paired TITAN reward deltas were −0.3193 versus the core, +0.1754 versus AURA, +0.1132 versus METAMORPH, −0.1644 versus CHIMERA, and 0.0000 versus HYPERION. TITAN and HYPERION had identical per-seed rewards, action counts, and episode lengths in these four trials, while their recorded decision traces differ. This pilot therefore shows no measured reward advantage over HYPERION and lower mean reward than the frozen core and CHIMERA. Four synthetic seeds are too few to support a general performance claim.

## TITAN telemetry

The receipt records 256 decisions, mean spectral divergence 0.044500, synthetic quantum entropy 0.466375, lattice coherence 0.715510, composite coherence 0.601132, classical Bell-fidelity proxy 1.000000, simulated buoyancy force 0.509382, and toy trophic richness 0.436104. It also records 244 memory replays, four simulated teleportations, and zero modder overrides. These are implementation telemetry fields. “Fidelity,” “quantum,” “buoyancy,” “teleportation,” and “trophic” do not refer to quantum hardware, real fluids, physical transport, or a biological ecosystem.

## Reproduce and verify

```powershell
python -m poseidon titan-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 99000000
python -m poseidon verify-titan outputs/titan_experiments/RECEIPT.json
```

The seed range, active Tidal checkpoint, and TidePool source must match this receipt to reproduce its exact digest. The benchmark does not train weights, create a fitted artifact, or activate TITAN as the default policy. See [design and limitations](TITAN_DESIGN.md).
