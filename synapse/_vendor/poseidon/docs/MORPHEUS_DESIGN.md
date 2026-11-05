# MORPHEUS: Offline Oneiric Sleep Engine & Hippocampal Memory Consolidation

## 1. Motivation & Theoretical Foundations

In complex resource environments like TidePool, biological organisms do not learn solely during active sensory engagement. Sleep provides a critical offline window for **synaptic downscaling**, **replay of salient waking trajectories**, and **phase stabilization**.

Inspired by `morpheus` and `3d-animal-simulator-Hybrid-Agent`, **Morpheus** equips Poseidon agents with an offline two-phase oneiric consolidation engine operating on top of the immutable frozen core:

```
[WAKING SALIENCE BUFFER]
          │
          ▼  (Rest action / Physical fatigue / High surprise)
┌────────────────────────────────────────────────────────┐
│                   MORPHEUS SLEEP ENGINE                │
│                                                        │
│  Phase 1: Slow-Wave Sleep (SWS)                        │
│    - Synaptic downscaling & noise pruning              │
│    - High-surprise trace compression into HRR memory   │
│                                                        │
│  Phase 2: Rapid Eye Movement (REM)                     │
│    - Counterfactual mental trajectory replay           │
│    - Quantum phase annealing:                          │
│        phi_a <- phi_a + eta * grad_phi E[R_dream]      │
└────────────────────────────────────────────────────────┘
          │
          ▼
[STABILIZED WAKE QUANTUM COHERENCE]
```

---

## 2. Mathematical Formalization

### 2.1 Sleep Onset Identification
Sleep is triggered whenever:
1. The agent selects a `rest` action (`action = 0`).
2. Stamina falls below the exhaustion boundary ($S_t < 0.15$).
3. Extended stationary dwell cycles occur ($\text{consecutive\_rests} \ge 2$).

### 2.2 Slow-Wave Sleep (SWS) Memory Compression
During the initial phase of sleep (ticks 1–2), SWS extracts memories from the waking replay buffer $\mathcal{M}$ where prediction discrepancy exceeds the salience threshold:
$$\text{Surprise}_t = |R_t - \hat{R}_t| \ge \theta_{\text{compression}}$$

Salient memories are bound into Holographic Reduced Representation (HRR) vectors via circular convolution:
$$\mathbf{z}_t = \mathbf{x}_{\text{role}} \circledast \mathbf{y}_{\text{state}}$$
and consolidated into the long-term trace bundle $\mathbf{T}$, permanently indexing high-yield food, water, and shelter waypoints.

### 2.3 Rapid Eye Movement (REM) Phase Annealing
During ticks 3+, REM generates counterfactual mental rollouts across recent experience windows without consuming real environment energy. It computes action-conditional advantages:
$$A(a) = \bar{R}(a) - \frac{1}{|\mathcal{A}|} \sum_{b} \bar{R}(b)$$

For high-performing actions ($A(a) > 0$), the complex action phases $\phi_a$ in the Causeway engine are annealed towards zero phase error:
$$\Delta \phi_a = -\eta_{\text{phase}} \cdot A(a)$$
This constructive phase rotation ensures that high-yield actions experience maximum constructive wave interference $I(a, b) = 2\,\text{Re}(\alpha_a^* \alpha_b)$ upon waking!

---

## 3. Telemetry & Verification

The Morpheus engine exposes real-time telemetry:
- `sleep_state`: `"wake"`, `"sws"`, or `"rem"`.
- `total_sleep_cycles`: Number of complete 4-tick sleep cycles executed.
- `consolidated_count`: Number of salient memories permanently bound into holographic storage.
- `rem_replays`: Total counterfactual trajectory replays simulated.
- `phase_gain`: Accumulated phase advantage aligned during REM sleep.
