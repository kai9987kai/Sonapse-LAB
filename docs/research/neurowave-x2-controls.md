# Sonapse NeuroWave-X2 — Matched Controls & Causal Ablation Protocol

## 1. Matched Negative Control Protocol

To ensure rigorous scientific attribution, every empirical claim of performance enhancement is evaluated against matched negative controls:

### 1.1 Compute-Matched Control (FLOP-Matched)
- **Hypothesis Tested**: Does inter-regional communication provide true representational value, or does performance increase merely from additional FLOPs?
- **Protocol**: When an experimental NeuroWave route requires $K$ FLOPs per step, an equal computational budget is granted to the isolated recurrent brain via local multi-layer feed-forward transformations.
- **Criterion**: $J(\text{NeuroWave}) - J(\text{FLOP-Matched}) > 0$ with $p < 0.01$.

### 1.2 Bandwidth-Matched Control (Wire-Matched)
- **Hypothesis Tested**: Does structured information in the packet payload matter, or is the receiver merely reacting to added non-zero activation energy?
- **Protocol**: An identical number of bits is transmitted across routes, but packet payloads are replaced with matched Gaussian or uniform random noise.
- **Criterion**: Lesioning informative payload destroys task success ($J(\text{Wire-Noise}) \approx J(\text{No-Comm})$).

### 1.3 Topology-Matched Control (Random Orthogonal Basis)
- **Hypothesis Tested**: Do graph-Laplacian eigenvectors leverage authentic connectome structure, or is any orthogonal basis equally effective?
- **Protocol**: The connectome-derived eigenbasis is substituted with a Haar-distributed random orthonormal matrix $Q \in \mathbb{R}^{K \times D}$ generated via QR decomposition.
- **Criterion**: The topological basis must exhibit superior spectral energy concentration and lower reconstruction error on connectome-structured signals.

### 1.4 Phase-Causality Controls (Shuffled and Frozen Phase)
- **Hypothesis Tested**: Is Communication-Through-Coherence (CTC) dynamically causal?
- **Ablation do(shuffle_phase)**: Regional phases $\theta_i$ are randomly permuted across regions.
- **Ablation do(freeze_phase)**: Regional phases are locked to a constant angle $\theta \equiv 0.0$.
- **Criterion**: If phase synchronization is functional, shuffling must significantly degrade information transmission fidelity.

---

## 2. Causal Do-Calculus Ablation Suite

The `CausalAblationController` (`synapse/neurowave_x2/ablations.py`) implements pearlian interventions:

```python
from synapse.neurowave_x2.ablations import CausalAblationController

# 1. Zero out all communication signals
ctrl = CausalAblationController(ablation_type="zero_comm")

# 2. Shuffle messages across routes to break content alignment
ctrl = CausalAblationController(ablation_type="shuffle_messages", seed=42)

# 3. Intervene on specific route e = (src, dst)
ctrl = CausalAblationController(ablation_type="lesion_route", target_route=(0, 1))
```

### Route Causal Importance Matrix
The causal contribution of route $e = (i, j)$ is measured as:
$$\text{Importance}(e) = J(\text{unablated}) - J(\text{do}(\text{lesion } e))$$
A positive score confirms that the specific route is causally necessary for peak performance.
