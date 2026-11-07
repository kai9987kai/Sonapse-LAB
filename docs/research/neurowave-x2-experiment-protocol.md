# Sonapse NeuroWave-X2 — Reproducible CLI & Experiment Protocol

## 1. CLI Commands & Recipes

NeuroWave-X2 is fully accessible via the unified `synapse neurowave-x2` CLI tool.

### 1.1 Status & Diagnostics Inspection
Display the current configuration, active communication topology, and spectral basis condition numbers:
```powershell
python -m synapse neurowave-x2 status
```

### 1.2 Scientific Conformance Test
Run the strict cross-engine validation comparing PyTorch autograd against the independent pure-NumPy reference engine:
```powershell
python -m synapse neurowave-x2 conformance --trials 20 --tolerance 0.0001
```

### 1.3 13-Arm Benchmark Matrix
Execute the comparative experimental matrix (Arms A0–A12) across independent random seeds:
```powershell
python -m synapse neurowave-x2 benchmark --arms A0,A1,A2,A7,A8,A10 --steps 50 --seed 42
```

### 1.4 Communication Stress Testing
Evaluate resilience to burst packet drops and Gaussian noise impairments:
```powershell
python -m synapse neurowave-x2 stress --loss-rate 0.15 --snr-db 15.0 --trials 10
```

### 1.5 Spectral Basis Analysis
Inspect eigenvalues, condition numbers, and subspace drift across graph perturbations:
```powershell
python -m synapse neurowave-x2 spectral --modes 8
```

### 1.6 Causal Ablations
Evaluate performance degradation under do-calculus interventions:
```powershell
python -m synapse neurowave-x2 ablate --type zero_comm
python -m synapse neurowave-x2 ablate --type shuffle_messages
python -m synapse neurowave-x2 ablate --type randomize_phase
```

### 1.7 Compute & Resource Profiling
Profile parameter counts, buffer memory, and empirical CPU execution latencies:
```powershell
python -m synapse neurowave-x2 profile --steps 100
```

### 1.8 Deterministic Replay Verification
Execute a step-by-step trace replay and verify action/reward trajectories:
```powershell
python -m synapse neurowave-x2 replay --steps 50 --tolerance 0.0001
```

### 1.9 Developmental Complexity Progression
Run developmental complexity progression from level 0 (isolation) to level 6 (full wave channel):
```powershell
python -m synapse neurowave-x2 development --target-level 6
```
