# Helm v0.6

Helm is a separate, small, CPU-fitted critic for TidePool control. Its reservoir carries observation history, the preceding executed action and prediction innovation (current observation minus Tidal's preceding forecast). Ridge bootstrap heads estimate six frozen-core continuation returns over 4 and 16 steps. The language model, Tidal's reset protocol and default policy are unchanged.

Three fitted families test observation-only input, history without innovation, and history with innovation. A disjoint selection set chooses the family and ridge setting; observation-only wins ties and improvements below the configured threshold. The yoked control substitutes a frozen training-only prediction-error schedule. Each family/control has its own disjoint calibration. Episode-level maximum advantage overestimation radii are descriptive diagnostic margins, without a safety or future-coverage guarantee.

An override must pass both horizons, support, remaining-time and physical action filters. There is no depletion exception that bypasses calibration. The ungated control is explicitly diagnostic. Only observations and previous executed-action predictions enter online state; no coordinates, simulator snapshots, seeds or future labels enter inference.

```powershell
python -m poseidon fit-helm --train-episodes 12 --selection-episodes 8 --calibration-episodes 8 --anchors 12 --max-steps 96
python -m poseidon helm-experiment --seed 123000001 --episodes 4 --max-steps 64 --scarcity 2.5
python -m poseidon verify-helm outputs/helm_experiments/RECEIPT.json
python -m poseidon world --planner helm --seed 42 --max-steps 96 --scarcity 2.5
python -m poseidon serve --port 8792
```

The workbench submits cancellable Helm experiments through `POST /api/jobs`. Jobs retain producer/source identity, generation, normalized settings, deadline, phase and final result. Polling and cancellation run outside the inference lock. Cancellation is cooperative at bounded experiment/replay chunks; a stuck native call cannot be forcibly stopped by the thread manager. Restarted unfinished work becomes interrupted failure and cannot become completed evidence. Existing synchronous experiment APIs remain available.

Paired evaluations compare core policy with selected, observation-only, no-innovation, yoked and ungated critics. They record full trajectories and actual six-action branches at regular anchors and every override. Replay validates recorded actions and dynamics without model weights. It does not prove recorded continuation actions came from those weights or verify forecast accuracy outside recorded branches. Threshold curves are descriptive and never silently tune deployed settings using evaluation labels. Conditional benefit/harm is undefined when a threshold accepts no changes.

Source lineage, license boundaries and current research are in [the complete portfolio refresh](research/README.md). The implementation interfaces and acceptance criteria are in [the spec](superpowers/specs/2026-10-09-helm.md).
