# MNEMORPH Archaeological Memory & Structural Lesions v0.8

MNEMORPH is an archaeological memory assay and associative network architecture adapted from memory graph and neural intervention paradigms. Rather than treating memory as an unstructured vector store, MNEMORPH models knowledge as an associative graph with typed semantic interventions.

---

## 1. Architectural Principles

### 1.1 Associative Concept Graph ($G = (V, E)$)
- **Nodes ($V$)**: Represent discrete entities, facts, and observations.
- **Edges ($E$)**: Represent semantic co-occurrence and associative bindings with edge weights $w_{uv} \in [0, 1]$.
- **Degree Centrality**: Hub nodes $h \in V$ exhibit high degree centrality $d(h) \gg \bar{d}$, acting as informational anchors.

### 1.2 Typed Structural Lesions & Interventions
To assess catastrophic forgetting and structural fault tolerance, MNEMORPH subjects the knowledge graph to four standardized experimental assays:
1. **Intact Baseline**: Full associative retrieval across the standard fact corpus.
2. **Hub Lesion Assay**: Selective deletion of the top 10% highest-degree hub nodes and their incident edges.
3. **Periphery Lesion Assay**: Selective deletion of low-degree peripheral nodes (degree $d(v) = 1$).
4. **Associative Regrowth Assay**: Hebbian associative reconstruction after lesioning, re-estimating lost paths via 2-hop neighbor transitive closure.

### 1.3 Key Evaluative Metrics
- **Veridical Retention Rate**: Proportion of facts successfully retrieved with matching ground-truth keywords under each assay.
- **Hub Vulnerability Ratio**:
  $$\text{HVR} = \frac{\Delta \text{Retention}_{\text{hub}}}{\Delta \text{Retention}_{\text{periphery}}}$$
  A high ratio confirms scale-free topology characteristics: extreme resilience to random/peripheral failures, but acute sensitivity to targeted hub lesions.
- **Associative Regrowth Recovery Rate**:
  $$\text{ARRR} = \frac{\text{Retention}_{\text{regrowth}} - \text{Retention}_{\text{lesioned}}}{\text{Retention}_{\text{intact}} - \text{Retention}_{\text{lesioned}}}$$

---

## 2. CLI & Benchmark Protocol

```powershell
# Run the archaeological memory study across all assays
python -m poseidon mnemorph-experiment

# Replay and verify the Mnemorph receipt offline
python -m poseidon verify-mnemorph outputs/mnemorph/RECEIPT.json
```
