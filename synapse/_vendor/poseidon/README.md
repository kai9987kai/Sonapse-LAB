# Supermix Poseidon v3.14.0: Local Research Workbench

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

Poseidon is an experimental **cognitive AI workbench** combining a trained Tidal core with optional prediction, memory, planning, adaptation, exact reasoning, and synthetic research tools. It runs locally on a conventional CPU. Version 3.14.0 is the current source identity; a source version or a passing local check does not establish a published model release.

Use the CLI or browser workbench to inspect available artifacts, run seeded synthetic episodes, compare experimental controllers, and replay recorded evidence. The [learning foundation report](docs/research/learning-foundation-2026-10-12.md) records the trained-predictor integration and its measured limitations. The [research index](docs/research/README.md) links subsequent experiments; the [11 October architecture audit](docs/research/architecture-audit-2026-10-11.md) preserves the earlier baseline findings.

## Core architecture and experimental extensions

The cognitive coordinator combines six experimental subsystems around the frozen Tidal core checkpoint:

1. **ORACLE**: CognitiveAgent and HyperAgent accept an explicit trained research artifact through `--oracle-checkpoint`. The episode-bootstrap ensemble predicts state changes, reward and an uncalibrated death target. Artifact/source checks and training-action support gate its use. Without a selected artifact, both controllers exactly preserve frozen Tidal actions. No candidate is automatically promoted.
2. **MNEMOSYNE (`poseidon/mnemosyne.py`)**: Biologically inspired 3-tier memory hierarchy:
   - **Tier 1 (Working Memory)**: Recurrent GRU context cell dynamically updated by prediction surprise ($e_{t-1} = \|o_t - \hat{o}_t\|_2$).
   - **Tier 2 (Episodic Store)**: Value-gated experiential memory weighted by reward, surprise, and recency.
   - **Tier 3 (Spatial Map)**: Online Bayesian replenishment estimation with conservative lower-confidence bounds (LCB).
   - **Matched Lesion Modes**: `intact`, `erased`, `shifted`, `random_retrieval`, and `memory_free` are implemented. Resource estimates feed a heuristic score adjustment when trained planning is enabled; learned incremental memory value remains a research objective.
3. **HORIZON-X (`poseidon/horizon_x.py`)**: Reserve-dependent predictive scoring. The current trained artifact supports one-step planning only; unsupported horizons and insufficient action exposure fall back to Tidal. State interval widths are diagnostics, and uncalibrated death outputs never become survival certificates.
4. **ADAPT (`poseidon/adapt.py`)**: Experimental residual adaptation, replay and anchor checks. Useful adaptation and retention require separate held-out evidence.
5. **NEURO (`poseidon/neuro.py`)**: Sparse routing with graph masks and gain scaling. The v2 coordinator reports its outputs as telemetry; they do not currently alter the planner input.
6. **WORLDLAB (`poseidon/worldlab.py`)**: Synthetic partially observed TidePool with fog, sensor noise, tidal effects and an external simulation bridge contract. It is not a validated physical or biological model.

See the [subsystem documentation](docs/POSEIDON_V2_RESEARCH.md), [24-project source review](docs/research/portfolio-refresh-2026-10-11.md) and [learning and transfer design](docs/superpowers/specs/2026-10-11-learning-transfer-design.md). Cognitive v3 receipts replay environment transitions and controller decisions, including noisy sensor RNG, and bind core/model/source identities. Parameter search uses isolated copies and retains inactive candidates. Later modules, including Project NEUROGRAFT-II, Project CROSSWIRE, Project FAULTLINE, Project TETHER, NEXUS-Ω, CHRONOS, NOETHER, and CARTAN, expose successor representations (SR-TD), instant zero-shot reward revaluation, sharp-wave ripple reverse replay, Graph Laplacian topological embeddings, trainable graph circuit routing, unfamiliar interface version space tracking, active diagnostic probing, all-hypothesis contract verification, closed-loop runtime shielding, condition-stratified joint failure selection, continuous reliability gating, 4-view causal invariance across 7 arms, synthetic experiments, and parsed problem-solving tools with domain-specific evidence scopes. Their names and status telemetry do not establish general learning or scientific discovery capabilities.

## Run locally

Use Python 3.10 or newer with a compatible CPU PyTorch installation. The declared runtime dependencies include SciPy for the scientific audit modules. From the repository root:

```powershell
python -m pip install -e .
python -m poseidon status
python -m poseidon serve --port 8792
```

Open `http://127.0.0.1:8792`. Installing the source package does not fetch model weights. Model-backed commands require their local artifacts under `runs/` and `models/`; `status` reports availability. A verified model release includes its packaged artifacts. The source launch scripts use port 8787.

The workbench can run a seeded TRIDENT episode from the command line when the selected Tidal checkpoint is present:

```powershell
python -m poseidon world --planner trident --seed 42 --max-steps 64 --scarcity 2.5
```

Run and replay the paired TRIDENT experiment:

```powershell
python -m poseidon trident-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 88000000
python -m poseidon verify-trident outputs/trident_experiments/RECEIPT.json
```

The receipt records all six arms and is sealed to the core checkpoint and TidePool source hashes. A short benchmark is useful for reproducibility checks, not for a robust performance conclusion.

## Verify changes

The [21 October workbench upgrade](docs/PROJECT_UPGRADE_2026-10-21.md) describes
saved-job recovery, resilient status diagnostics, source tracking, and the
distinction between answer checks, synthetic demonstrations, checksums and replay.

Install the optional test tools and run the focused integrity checks or the full suite:

```powershell
python -m pip install -e ".[test]"
python -m pytest tests/test_packaging.py tests/test_release_verification.py tests/test_jobs.py tests/test_provenance.py -q
python -m pytest -q
```

The packaging checks generate metadata in a temporary project and verify that the distribution version matches `poseidon.__version__` without importing the runtime. Release integrity tests require no model download. The full suite also exercises synthetic controllers, protocol validation, and local server behavior; passing tests establish the checked software behavior rather than new model performance.

The [30 October research-grounded upgrade](docs/PROJECT_UPGRADE_2026-10-30.md)
adds independent generated reasoning checks, repairs controller adapters, and
requires actual supplied evidence for the opt-in coordinator. Changed ensemble
predictors preserve the baseline until their own calibration is available.

Run the browser's **Independent reasoning audit** or save and replay a generated
suite without model weights:

```powershell
python -m poseidon reasoning-audit --seed 20261030 --samples 32 --output outputs/reasoning-audit.json
python -m poseidon verify-reasoning-audit outputs/reasoning-audit.json
```

The audit independently checks arithmetic, equations, determinants, gcd, Hanoi
moves and edit distance, plus invalid and unsupported controls. It reports errors,
refusals and accepted-answer error separately. These finite checks do not establish
general model accuracy or statistical risk guarantees. `--skip-replay` checks
report integrity only.

The [31 October robustness upgrade](docs/PROJECT_UPGRADE_2026-10-31.md) preserves
exact decimal input across the solvers, adds paired related-input reasoning
checks, and screens supplied held-out episode outcomes for a fixed policy:

```powershell
python -m poseidon metamorphic-audit --seed 20261031 --samples 32 --output outputs/metamorphic.json
python -m poseidon verify-metamorphic-audit outputs/metamorphic.json
python -m poseidon episode-risk study.json --output outputs/episode-risk.json
python -m poseidon verify-episode-risk outputs/episode-risk.json
```

Both tools are available in the browser. Consistency is measured separately from
independent correctness. Episode risk bounds require the stated sampling and
fixed-policy assumptions; they do not establish per-action risk or activate a
controller. See the upgrade document for the study JSON protocol.

## Model components and boundaries

| Component | Role | Release status |
|---|---|---|
| Tidal | Compact locally trained scene, action, and dynamics policy | Existing selected checkpoint; unchanged in this upgrade |
| SmolLM2 | Pretrained conversation model | Separate backend with its own upstream license |
| Project CROSSWIRE | Interface permutation hypotheses, diagnostic probing, finite contract checks and runtime shielding | Synthetic interface experiments; TidePool abstraction is heuristic |
| Project FAULTLINE | Condition-stratified joint failure selection, parity audit, covariance and panel triage | Synthetic audits; changed ORACLE panels require their own calibration |
| Project TETHER | Continuous aleatoric-epistemic reliability gating, 4-view causal invariance (7 arms), and evidence-calibrated frozen Tidal fallback gate | Stage 28 research foundation; no uncalibrated overrides |
| Problem Forge | Parsed mathematical, physical, and engineering tools, including synthetic demonstrations | Scope varies by domain; display-token smoke checks are not accuracy certificates |
| TITAN, TRIDENT, and other planners | Optional controllers for TidePool | Experimental; no default-policy promotion |
| Image, video, and mesh tools | Bounded procedural outputs | Separate software paths |
| Math and carrier memory | Exact solver and external retrieval | Separate tools; not weights in Tidal |

TidePool is a deterministic toy environment. TITAN's ecology, fluid, quantum-inspired, and device-inspired telemetry is simulated on a conventional computer. Poseidon makes no claim of consciousness, biological validation, physical RF harvesting, NTAG hardware, analog video, quantum hardware, or general autonomous learning. The LoRA remains an inactive candidate unless explicitly selected.

## Release and licensing

The release manifest binds packaged files to a source commit and SHA-256 inventory. Distribution metadata reads its version from `poseidon.__version__`; the release builder and verifier share their supported version list. Use the [release guide](docs/HUGGINGFACE_RELEASE.md) to package, verify, and publish a version after its evidence requirements pass. The Hub repository root is a custom PyTorch application package; `AutoModel.from_pretrained` is not its runtime interface.

Poseidon code and Tidal artifacts use MIT terms. Bundled pretrained language weights and tokenizer retain their upstream Apache-2.0 terms and attribution. See the packaged license files and upstream model card before redistribution.
