# AURA Biomimetic Central Complex Controller v0.8

AURA is a biomimetic neural control architecture synthesized from experimental arthropod neurobiology (*Drosophila melanogaster* Central Complex and Mushroom Body connectomes) and bio-inspired robotics. It operates as a descending arbitration layer over the frozen Tidal core neural policy in TidePool survival environments without modifying model weights.

---

## 1. Neuroarchitectural Foundations

### 1.1 CX-Ring Attractor Compass (`CXRingCompass`)
The central navigation module models the ellipsoid body (EB) ring attractor:
- **16-Wedge Discrete Ring**: 16 azimuthal wedges centered at $\theta_i = \frac{2\pi i}{16}$ for $i \in \{0, \dots, 15\}$.
- **Cosine Recurrent Kernel**: Symmetric recurrent connectivity with local excitation and global inhibition:
  $$W_{ij} = \cos(\theta_i - \theta_j)$$
- **Angular Velocity Integration**: Rotational steering velocity $\Delta \theta$ continuously shifts the neural activation bump across the azimuth.
- **Population Vector Average (PVA)**:
  $$\hat{\theta} = \operatorname{atan2}\left(\sum_{i=0}^{15} A_i \sin \theta_i, \; \sum_{i=0}^{15} A_i \cos \theta_i\right)$$
- **PVA Coherence ($R$)**:
  $$R = \frac{\sqrt{\left(\sum_i A_i \cos \theta_i\right)^2 + \left(\sum_i A_i \sin \theta_i\right)^2}}{\sum_i A_i} \in [0, 1]$$
  High coherence ($R \approx 1$) indicates a sharply localized, confident heading representation; low coherence ($R \to 0$) signals sensory conflict or rotational disorientation.
- **Optomotor Reflex**: When bump coherence collapses below the critical threshold ($R < 0.35$), involuntary descending optomotor reflexes trigger stabilization turns to restore compass lock before navigating.

### 1.2 Sparse Neuropil Arbiter (`SparseNeuropilArbiter`)
The decision arbitration module models Kenyon cell (KC) expansion in the mushroom body calyx:
- **128 Virtual Kenyon Cells**: 16-dimensional observation vectors are sparsely projected into a 128-dimensional neuropil representation.
- **APL Inhibitory Gain Control ($k=8$ WTA)**: Simulating anterior paired lateral (APL) GABAergic feedback inhibition, only the top $k=8$ most active neurons retain positive activations, enforcing strict $6.25\%$ population sparsity.
- **Tri-Drive Homeostatic Arbitration**: Evaluates three orthogonal survival drives:
  1. *Metabolic Drive* ($\mathcal{D}_{\text{met}}$): Hunger and thirst deficit.
  2. *Fatigue Drive* ($\mathcal{D}_{\text{fat}}$): Stamina depletion and physical exhaustion.
  3. *Threat Drive* ($\mathcal{D}_{\text{thr}}$): Exposure, predator proximity, and adverse environmental weather.
- **Descending Motor Channels**:
  - `harvester`: Dominates under high metabolic deficit; biases actions toward foraging and drinking.
  - `sentinel`: Dominates under high threat; biases actions toward shelter seeking and defensive vigilance.
  - `escaper`: Dominates under acute danger or panic; executes reflexive flight or emergency rest.

### 1.3 Tessera Macro-Action Commons Interfacing
AURA directly monitors the ratified opcode table from the Tessera macro-action commons. When homeostatic drives trigger sequence execution, AURA seamlessly dispatches ratified multi-step macros (`OP_FORAGE_BURST`, `OP_TACTICAL_RETREAT`) validated for stamina and environmental safety.

---

## 2. Mathematical Formalism

$$\begin{aligned}
A^{(t+1)}_i &= \operatorname{ReLU}\left(A^{(t)}_i + \alpha \sum_{j} W_{ij} A^{(t)}_j + \beta I_i(\Delta \theta) - \gamma\right) \\
\mathbf{z}_{\text{KC}} &= \operatorname{APL\_WTA}\left(\mathbf{W}_{\text{proj}} \mathbf{x} + \mathbf{b}, \; k=8\right) \\
\mathbf{d}_{\text{homeo}} &= \sigma\left(\mathbf{W}_{\text{drive}} \mathbf{z}_{\text{KC}}\right) \in [0, 1]^3 \\
\text{Channel}^* &= \operatorname{argmax}\left(\mathbf{d}_{\text{homeo}}\right)
\end{aligned}$$

---

## 3. CLI & Benchmark Protocol

```powershell
# Run the 5-arm paired AURA evaluation study
python -m poseidon aura-experiment --seed 150000001 --episodes 4 --max-steps 64 --scarcity 2.5

# Replay and verify an AURA receipt offline
python -m poseidon verify-aura outputs/aura_experiments/<RECEIPT_ID>.json

# Run an interactive world simulation episode with AURA planner
python -m poseidon world --planner aura --seed 42 --max-steps 64 --scarcity 2.5
```

---

## 4. Experimental Arms

1. `policy`: Baseline frozen Tidal DAgger policy without biomimetic modulation.
2. `odysseus`: Empirical Bayesian replenishment navigator (v0.7).
3. `aura`: Full integrated biomimetic controller (CX-ring + Neuropil arbiter + Tessera macro-commons).
4. `aura_no_ring`: Ablation arm disabling ring attractor heading tracking and optomotor reflex.
5. `aura_no_neuropil`: Ablation arm disabling sparse Kenyon cell arbitration and tri-drive descending gates.
