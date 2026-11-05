---
language:
- en
license: mit
tags:
- pytorch
- world-models
- reinforcement-learning
- dagger
- imitation-learning
- mixture-of-experts
- model-predictive-control
- procedural-generation
- connectome
- multimodal
pipeline_tag: reinforcement-learning
---

# Supermix Poseidon & Beyond: Adaptive Cognitive World Model

Official model weights, epistemic ensembles, and research receipts for **Supermix Poseidon & Beyond**, a compact, auditable cognitive world model system designed for local consumer CPU execution.

- **GitHub Repository**: [https://github.com/kai9987kai/Supermix-Poseidon](https://github.com/kai9987kai/Supermix-Poseidon)
- **Model Parameters**: 327,230 (~0.33M trunk) + ~150K (3-Head Dynamics Ensemble)
- **Architecture**: Sparse Adaptive Mixture-of-Experts, 3-Head Uncertainty Ensemble, Connectome Recurrence, Persistent Dual Memory
- **Inference Latency**: ~2.9 ms (Reactive) / ~5.6 ms (Uncertainty-Aware MPC) on consumer CPU

---

## What is Supermix Beyond?

Supermix Beyond evolves Poseidon from isolated capability demonstrations into an **adaptive cognitive world model** that learns from experience, estimates epistemic uncertainty, retains cross-step memory, and improves its behavior over time.

```
                  ┌──────────────────────┐
   Text Prompt ──>│ Blake2b Lemma-Bigram │──┐
                  └──────────────────────┘  │   ┌───────────────────────────┐   ┌──> Scene Attributes (shape, color, motion, count, scale)
                                            ├──>│ Sparse Top-2 Adaptive MoE ├───┼──> Macro-Action Policy (6 survival choices)
                  ┌──────────────────────┐  │   │ (Adaptive Recurrent Depth)│   └──> 3-Head Dynamics Ensemble f_θ(s, a)
   16-d State  ──>│  Observation Project │──┘   └───────────────────────────┘          │
                  └──────────────────────┘                                             v
                                                                            Epistemic Disagreement: U(s, a)
                                                                            Failure Risk: P(failure | a)
                                                                                       │
                                                                                       v
                                                                            Risk-Sensitive MPC Planner:
                                                                            Q*(s, a) = E[R] - λ U(s, a) - β P(fail)
```

---

## Checkpoints Included

| File | Size | Description |
|---|---|---|
| `core.pt` | 1.33 MB | **Active Promoted Checkpoint** (Tidal Core trained via joint curriculum + on-policy DAgger) |
| `ensemble_adapted.pt` | 382 KB | **Beyond 3-Head Dynamics Ensemble** (Trained on surprise replay buffer and verified by promotion audit) |
| `beyond_audit_receipt.json` | 954 B | Audit receipt verifying 100% held-out test survival and 81% relative dynamics loss reduction |
| `core_base.pt` | 4.00 MB | Supervised baseline checkpoint (120,000 examples / 360,000 exposures) |
| `config.json` | 192 B | Model architecture specification |
| `active_core.json` | 287 B | Promotion manifest verifying validation score improvement |
| `calibration.json` | 2.24 KB | Temperature scaling parameters on dev split (Guo et al. 2017) |
| `evaluation.json` | 100.6 KB | Frozen test split evaluation across 2,048 scenes and 24 survival seeds |
| `receipt.json` | 2.42 KB | Training receipt with dataset SHA-256 hashes and exposure counters |
| `adapter/` | ~254 KB | Pinned LoRA adapter experiment for conversation fine-tuning on SmolLM2-135M |

---

## Benchmark Results & Scientific Discoveries

### 1. 100-Episode Paired Planning Benchmark (`scarcity = 2.5`)
Evaluated across **100 independently seeded paired episodes** (seeds 1000–1099, max steps = 256):

| Planner | Survival Rate | 95% Confidence Interval | Mean Steps | Mean Return | Latency / Step |
|---|---|---|---|---|---|
| **Reactive Policy (DAgger Prior)** | **100.0%** | $\pm 0.00\%$ | 256.0 | 25.68 | 2.90 ms |
| **Single-Model MPC (v0.1)** | **100.0%** | $\pm 0.00\%$ | 256.0 | 25.67 | 8.47 ms |
| **Uncertainty-Aware MPC (Beyond)** | **99.0%** | $\pm 1.95\%$ | 255.4 | 25.53 | **5.67 ms** |

*Finding*: The vectorized 3-model uncertainty planner achieves comparable survival to single-model MPC while running **33.1% faster** on CPU.

### 2. The "Pessimism Trap" Under Severe Starvation (`scarcity = 4.0`)
When tested under extreme resource starvation:
- When resources are virtually non-existent, **all possible exploratory moves carry high epistemic transition uncertainty**.
- With a high uncertainty penalty ($\lambda = 1.5$), the risk-sensitive planner exhibits the classical **Pessimism Trap**: it excessively penalizes high-variance exploratory paths, preferring to remain resting/sheltered even as energy runs out.
- Dynamic attenuation of uncertainty penalties during critical resource depletion restores exploratory survival.

### 3. Connectome Circuitry Comparison (Testing Expanse's Hypothesis)
Matched 256-node recurrent cells with 4,118 synapses (6.28% density):
- **Authentic Fly Connectome**: state variance = 0.0525, autocorrelation = 0.1368
- **Maslov-Sneppen Degree-Preserving Shuffled**: state variance = 0.0543, autocorrelation = 0.1843
- *Finding*: Authentic biological wiring exhibits **no automatic mathematical advantage** over degree-preserving shuffled wiring in unadapted state variance, reproducing the negative finding from Expanse while achieving a **93.72% reduction in synaptic parameters**.

### 4. Delayed-Recall Memory Retention
Testing the agent's ability to retain environmental cues hidden during intermediate distractor steps:
- **Dual Persistent Memory**: **100.0% recall** across horizons $K \in \{2, 4, 8\}$.
- **Reactive Baseline (No Memory)**: Collapses to chance ($26\% - 36\%$).
- **Shuffled Memory**: Drops to $12\% - 16\%$ (worse than chance), proving that corrupting historical episodic records actively damages decisions.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/kai9987kai/Supermix-Poseidon.git
cd Supermix-Poseidon
pip install -e .
```

### 2. Python Inference: Uncertainty-Aware Planning

```python
from huggingface_hub import hf_hub_download
from poseidon.core import CoreRuntime
from poseidon.ensemble import UncertaintyAwarePlanner, WorldModelEnsemble, RiskConfig
from poseidon.world import TidePool

# Download core checkpoint
core_path = hf_hub_download(repo_id="Kai9987kai/Supermix-Poseidon", filename="core.pt")
runtime = CoreRuntime(core_path)

# Download adapted 3-model ensemble
ensemble_path = hf_hub_download(repo_id="Kai9987kai/Supermix-Poseidon", filename="ensemble_adapted.pt")
ensemble = WorldModelEnsemble(hidden_size=runtime.model.config.hidden_size, k=3)
checkpoint = torch.load(ensemble_path, map_location="cpu")
ensemble.load_state_dict(checkpoint["ensemble_state_dict"])

# Initialize Uncertainty-Aware Planner
planner = UncertaintyAwarePlanner(runtime, ensemble=ensemble, config=RiskConfig(horizon=2))

# Execute lookahead planning on an observation
obs = [1.0, 0.8, 0.8, 0.9, 0.1, 0.0, 0.5, 0.5, 0.8, 0.1, 0.5, 0.5, 0.2, 0.5, 0.0, 0.1]
plan = planner.plan(obs)

print(f"Chosen Action: {plan['action_name']}")
print(f"Epistemic Uncertainty U(s,a): {plan['epistemic_uncertainty']:.5f}")
print(f"Failure Probability: {plan['failure_probability']:.4f}")
```

### 3. Unity WorldLab Integration
Supermix Beyond includes a local socket bridge for Unity 3D environments:

```bash
# Start the Python bridge server on port 8085
python -c "from poseidon.unity_bridge import UnityBridgeServer; s = UnityBridgeServer(); s.start(); import time; time.sleep(60)"
```

Attach `docs/unity/SupermixAgentBridge.cs` to your Unity Agent GameObject to stream live 16-d sensor telemetry and receive uncertainty-calibrated actions in real time!

---

## Citation & Lineage

```bibtex
@misc{beyond2026,
  author = {Kai Piper},
  title = {Supermix Beyond: Adaptive Cognitive World Model, Uncertainty-Aware Planning, and Persistent Experience},
  year = {2026},
  url = {https://github.com/kai9987kai/Supermix-Poseidon}
}
```
