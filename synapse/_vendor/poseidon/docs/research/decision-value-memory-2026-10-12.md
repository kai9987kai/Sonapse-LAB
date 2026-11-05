# Poseidon Decision-Value-Gated Episodic Memory (DV-GEM)

**Date:** 12 October 2026  
**Status:** Prototype implemented; v2 guarded pilot shows benefit over erased memory and frozen core, but fails the fresh resource-margin gate and remains unpromoted
**Artifact Directory:** `outputs/decision-memory-pilot-730000000/`  
**Protocol Digest:** `51787879b9ed7d61318fbf4c2df785b15ccc54a7037f125d225fb7bcc9f82ee1`  
**Test Suite Verification:** 739 / 739 passed (100%)  

---

> **Retrospective correction (14 October 2026):** The original `alias-tidepool-v1` pilot cannot establish memory benefit. Its resource dynamics make survival impossible over the 24-step horizon: both resources drain, and the mismatched action cannot replenish either one. All arms therefore reached the same floor, so the zero-survival endpoint was non-discriminating. The learned `intact` DV-GEM arm opened its gate but recorded zero action overrides. The similarity control recorded eight overrides in scarcity and none in the other conditions; all arms still died. The intact arm matched baseline reward in the ID and noise conditions, and its utility remains unproven. A versioned maintenance world and a controlled cue-retention assay are documented in the [v2 memory-value study](alias-world-v2-memory-value-2026-10-14.md). That study evaluates an explicit one-bit rule, not the trained DV-GEM scorer.

Claims below describing the v1 setup and its historical pilot are retained as a record. Read the correction above as the current interpretation.

## 1. Executive Summary

A fundamental limitation in autonomous reinforcement learning and cognitive systems is **the observation alias problem**: in partially observable environments, identical current sensory observations $o_t$ may demand opposite interventions depending on latent environmental states or past temporal cues. 

Prior approaches fail under this regime in two distinct ways:
1. **Markovian / Reactive Policies** (including our frozen Tidal core): operate exclusively on $o_t$ and are blind to latent historical context.
2. **Naive Similarity Retrieval**: retrieves transitions whose observations are closest in Euclidean distance to $o_t$. In aliased states, nearest neighbors are other aliased states, which fail to retrieve the distal informational cue and frequently inject contradictory actions.

To solve this, we implemented **Decision-Value-Gated Episodic Memory (DV-GEM)**, fulfilling Stages 4 and 5 of the [learning and transfer design spec](../superpowers/specs/2026-10-11-learning-transfer-design.md). Instead of retrieving transitions based on observation similarity or raw reward magnitude, DV-GEM trains a neural utility scorer to predict whether retrieved past evidence $m$ provides a positive counterfactual decision advantage over no retrieval:

$$\widehat{\Delta Q}(o_t, a, m) = \widehat{Q}(o_t, a, m) - \widehat{Q}(o_t, a, \emptyset)$$

Evidence is admitted if and only if candidate support is verified, evidence conflict is below tolerance, and the lower confidence bound of decision value exceeds an explicit threshold ($\text{LCB}_{\Delta Q} > \tau_{\text{admission}}$). Default memory influence is strictly zero.

---

## 2. Architectural Blueprint

```mermaid
flowchart TD
    subgraph Environment ["ObservationAliasTidePool (alias-tidepool-v1)"]
        Cue["Early Cue Window (t < 4)"] --> Alias["Sensory Occlusion & Aliasing (t >= 4)"]
    end

    subgraph Perception ["Observation Processing"]
        Obs["Normalized Observation o_t (16-D)"]
        Belief["Recurrent Belief Encoder h_t (GRU)"]
        Obs --> Belief
    end

    subgraph Memory ["DV-GEM Episodic Hierarchy"]
        Store[("Episodic Store (Capacity 256)")]
        CandidatePool["Candidate Retrieval Pool\n(Distal Cues + Local Neighbors)"]
        Store --> CandidatePool
    end

    subgraph Scorer ["Causal Decision-Value Scorer"]
        Query["Query q = (o_t, a, h_t)"]
        Empty["Empty Memory Baseline Ø"]
        MemoryInput["Retrieved Evidence m"]
        
        CandidatePool --> MemoryInput
        Query & MemoryInput --> Q_mem["Q(s, a, m)"]
        Query & Empty --> Q_empty["Q(s, a, Ø)"]
        
        Q_mem & Q_empty --> DeltaQ["ΔQ = Q(s, a, m) - Q(s, a, Ø)"]
        DeltaQ --> LCB["LCB = ΔQ - λ · σ_net"]
    end

    subgraph Gating ["Fail-Closed Admission Gate"]
        Support{"Supported & Conflict <= τ?"}
        Value{"LCB > τ_admission?"}
        Admitted["Admitted Delta: α · ΔQ(a)"]
        Closed["Default Influence: 0.0"]
        
        LCB --> Value
        Support --> Value
        Value -- Yes --> Admitted
        Value -- No --> Closed
    end

    subgraph Execution ["Controller Action Selection"]
        Core["Frozen Tidal Core Policy"]
        Core --> BaseProbs["Base Policy π_0(a)"]
        BaseProbs & Admitted & Closed --> Modulated["Modulated Action Scores\nS(a) = π_0(a) + δ_m(a)"]
        Modulated --> ActionChoice{"Argmax S(a) != a_base\n& Margin > τ_override?"}
        ActionChoice -- Yes --> Override["Audited Override Action"]
        ActionChoice -- No --> BaseAction["Baseline Fallback Action"]
    end

    Alias --> Obs
    Override & BaseAction --> Store
```

---

## 3. The Observation Alias World (`poseidon/research/alias_world.py`)

Adhering strictly to the `EnvironmentContract` (`alias-tidepool-v1`):
- **16 Observation Dimensions:** Identical channel layout and $[0, 1]$ bounds as `tidepool-v1`.
- **6 Action Space:** Rest, Forage, Drink, Shelter, Explore, Flee.
- **Two Latent Environmental Regimes:**
  - *Regime 0 (Algae Bloom):* Foraging (action 1) yields $+0.80$ reward and $+0.35$ energy; Drinking (action 2) yields $0.0$ reward and wastes energy.
  - *Regime 1 (Freshwater Spring):* Drinking (action 2) yields $+0.80$ reward and $+0.35$ hydration; Foraging (action 1) yields $0.0$ reward and wastes energy.
- **Temporal Observation Alias:**
  - In initial window ($t < 4$): Channel 13 (`resource_scent`) provides unambiguous evidence ($0.85$ vs $0.15$).
  - At $t \ge 4$: Channel 13 is occluded to neutral ($0.50$), and resource channels (6 and 7) display identical ambiguous local traces ($0.35$). Observations are bit-for-bit identical across regimes.
- **Counterfactual Branching:** Deterministic `snapshot()` and `restore()` capturing world parameters and sensor RNG state for exact multi-horizon branch evaluation.

---

## 4. The Six Matched Experimental Arms

| Arm | Description | Memory Gating Condition | Expected Behavior |
| --- | --- | --- | --- |
| **`intact`** | Full DV-GEM with learned utility scoring and admission gate | Opens selectively when $\text{LCB}_{\Delta Q} > \tau$ | Disambiguates aliased states using causal cue value |
| **`similarity`** | Classical Euclidean observation nearest neighbor | Unconditionally active | Matches current aliased observations, missing distal cues |
| **`erased`** | Memory buffer cleared at every step | Permanently closed ($\alpha = 0$) | Falls back identically to frozen baseline |
| **`shifted`** | Circularly permuted memory keys | Disrupted semantic alignment | Generates high conflict or negative decision values |
| **`donor`** | Populated with records from unrelated donor episodes | Out-of-distribution donor evidence | Induces conflict or irrelevant interventions |
| **`baseline`** | Frozen Tidal Core runtime | Bypassed completely | Operates as fixed incumbent reference |

---

## 5. Pilot Results (`outputs/decision-memory-pilot-730000000/`)

### Protocol Parameters
- **Base Seed:** `730000000`
- **Training Updates:** 300 updates on 8 cross-fitted training seeds
- **Evaluation Episodes:** 16 paired episodes per condition across all 6 arms
- **Max Steps:** 24 steps per episode
- **Conditions:** `id_alias` (in-distribution), `ood_scarcity` (scarcity 2.5), `ood_noise` (sensor noise std 0.04)

### Gate Admission Dynamics
| Condition | Arm | Mean Gate Open Steps / Ep | Mean Reward | Survival Rate |
| :--- | :--- | :---: | :---: | :---: |
| **`id_alias`** | `intact` | **7.12** | 0.20625 | 0.0 |
| | `similarity` | **14.00** | 0.20625 | 0.0 |
| | `shifted` | **6.00** | 0.20625 | 0.0 |
| | `donor` | **7.06** | 0.20625 | 0.0 |
| | `erased` | **0.00** | 0.20625 | 0.0 |
| | `baseline` | **0.00** | 0.20625 | 0.0 |
| **`ood_scarcity`** | `intact` | 7.00 | 0.50000 | 0.0 |
| | `similarity` | 14.00 | 0.92500 | 0.0 |
| | `baseline` | 0.00 | 0.50000 | 0.0 |
| **`ood_noise`** | `intact` | 7.12 | 0.615625 | 0.0 |
| | `similarity` | 14.00 | 0.615625 | 0.0 |
| | `baseline` | 0.00 | 0.615625 | 0.0 |

### Key Findings
1. **Gate activity recorded, utility not validated:** The admission gate opened on 7.12 steps per episode in `intact`, while remaining closed on 0.0 steps in `baseline` and `erased`. The intact arm produced no action overrides. Similarity retrieval opened on all 14 post-cue steps and produced eight overrides across the 16 scarcity episodes.
2. **Replay Verification:** All 16 evaluation seeds across all arms and conditions replayed with 100% deterministic bit-for-bit reward and survival agreement (`verify_decision_memory_receipt == True`).
3. **Fail-Closed Promotion:** Because bootstrap reward lower bounds over the frozen baseline did not cross the strictly positive margin in `id_alias` ($[0.0, 0.0]$), the model was **not promoted** to active default (`promoted: false`).
4. **Immutability Preserved:** `runs/tidal_dagger/core.pt` remained untouched throughout the experiment.

The pilot therefore verifies deterministic execution and fail-closed promotion behavior, not decision-value prediction or controller improvement. The all-zero survival outcome is explained by the v1 world design and must not be treated as evidence that memory has no possible benefit.

---

## 6. Synthesis with Portfolio Innovations

DV-GEM bridges the theoretical concepts from across the 24 reference projects:
- **TESSERA & Causeway:** Counterfactual branching at anchor states using snapshot/restore replay to isolate causal intervention effects.
- **Mnemorph:** Gated morphic memory routing between working context (recurrent belief) and long-term episodic traces.
- **GhostInTheMachine:** Full introspection and conflict auditing prior to action execution.
- **NexusSearch:** Dual-pathway candidate retrieval combining local metric proximity with distal informational saliency.
