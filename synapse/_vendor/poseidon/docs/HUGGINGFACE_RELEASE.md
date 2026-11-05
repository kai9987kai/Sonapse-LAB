# Hugging Face release procedure

Poseidon is a custom PyTorch runtime package hosted at [Kai9987kai/Supermix-Poseidon](https://huggingface.co/Kai9987kai/Supermix-Poseidon). The repository root is not a Transformers `AutoModel` checkpoint. The current source identifies itself as **3.14.0**, with exact decimal/scientific parsing, related-input reasoning audits, and offline held-out episode screening around the unchanged selected Tidal core. The [31 October upgrade report](PROJECT_UPGRADE_2026-10-31.md) records the implementation, research sources and finite validation scope. This guide describes release preparation; it does not assert that a v3.14.0 Hub tag has been published.

`poseidon.__version__` is the version source for both runtime and distribution metadata. The release builder and verifier share the supported version list, including historical v3.0.0 through v3.14.0 profiles. New v3 profiles retain the required artifact digests and replay evidence from v2.0.0.

## Download and run

```powershell
python -m pip install huggingface-hub
$HubRevision = "REPLACE_WITH_A_VERIFIED_HUB_COMMIT_SHA"
hf download Kai9987kai/Supermix-Poseidon --revision $HubRevision --local-dir Poseidon-release
Set-Location Poseidon-release
python tools/verify_huggingface_release.py .
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon serve --port 8792
```

Replace `$HubRevision` with the immutable commit SHA of the release you have verified. Hash verification uses only the Python standard library and can run before installing the package; smoke verification requires its runtime dependencies, including SciPy. Use Python 3.10+ with compatible CPU PyTorch. The packaged snapshot contains the default runtime weights and tokenizer; loading the custom project uses the Poseidon runtime rather than `AutoModel.from_pretrained`.

To evaluate TRIDENT locally:

```powershell
python -m poseidon world --planner trident --seed 42 --max-steps 64 --scarcity 2.5
python -m poseidon trident-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 88000000
python -m poseidon verify-trident outputs/trident_experiments/RECEIPT.json
```

The receipt is bound to the active Tidal checkpoint and TidePool source bytes. It includes action-level decisions and observed environment transitions for the core, TRIDENT, erased, shifted, directed-support-gate-off, and residual-gate-off arms. Replay verifies integrity and deterministic behavior; it is not an independent scientific replication or evidence of performance beyond the recorded synthetic task.

## Build, verify, and upload

Start with a clean Git checkout and a completed `outputs/trident_experiments/RECEIPT.json`. The packager checks the active checkpoints, pinned upstream language files, existing experiment evidence, and the TRIDENT receipt. It rejects dirty source so the manifest can bind a commit. A source checkout, wheel build, or passing unit suite alone does not satisfy these model-release requirements. The build may download pinned historical release assets when they are absent from the local Hub cache.

```powershell
python tools/package_huggingface.py --output outputs/releases/poseidon-v3.14.0
python tools/verify_huggingface_release.py outputs/releases/poseidon-v3.14.0 --smoke --timeout 1800
hf upload Kai9987kai/Supermix-Poseidon outputs/releases/poseidon-v3.14.0 . --commit-message "Release Poseidon v3.14.0: exact reasoning and research robustness"
```

The v3.14.0 package includes fresh independent and metamorphic reasoning audits under `outputs/research/robustness-upgrade-20261031/`; packaging and isolated smoke verification replay them and recompute the synthetic episode fixture. Historical v3.13.0 audit bytes remain preserved at their original paths. These finite tool checks do not establish general model accuracy; the episode fixture tests the screening software and is not controller-performance evidence.

After upload, verify the Hub's new commit SHA, intended version tag, model card, manifest, file inventory, and remote file hashes against the local release package. Upload does not itself create a version tag. A local package or successful CLI upload alone is not proof that publication completed. Preserve the existing Hub history and release assets; do not force-push.

To check a source wheel without downloading model artifacts or installing the result:

```powershell
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir outputs/upgrade-build
```

This wheel contains application source and web assets. Model checkpoints remain separate release artifacts. Build tooling reads the literal runtime version without importing the package's scientific modules.

## Scope and licensing

Tidal is a compact learned scene/control model. SmolLM2 is a separate pretrained conversation model with separate upstream terms. Procedural media, exact arithmetic, external carrier retrieval, and TidePool are distinct runtime components. TRIDENT is a local heuristic over synthetic observations and recorded transitions; its small paired pilot does not establish a general performance improvement. TITAN's ecology, fluid, quantum, intermittent-power, and device telemetry are software simulations, not physical capabilities or biological evidence.

Poseidon source and Tidal artifacts retain MIT terms. SmolLM2 weights, tokenizer, and attribution retain the upstream Apache-2.0 terms. Review included license files and upstream model documentation before redistribution.
