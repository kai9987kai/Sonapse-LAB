# Memory Carrier Observatory (MCO) v0.7

The Memory Carrier Observatory provides causal, factorial evaluation of external language memory carriers in Poseidon without modifying core model weights.

---

## The Four Memory Carriers

Poseidon partitions explicit symbolic and language facts into four distinct semantic carriers:
1. `episodic`: Temporary sequential observations, recent events, and state transcripts.
2. `body`: Internal physiological vitals, metabolic depletion rates, exhaustion thresholds, and somatic invariants.
3. `habitat`: Environmental topologies, patch resource caps, weather cycles, shelter ratings, and tidal danger zones.
4. `social`: Conspecific behaviors, signal vocabulary, cooperative conventions, and threat alerts.

---

## Factorial Causal Design

To definitively measure whether memory retrieval helps or merely adds confounding noise, MCO evaluates:
- **All 16 Inclusion/Exclusion Subsets**: $\mathcal{P}(\{\text{episodic}, \text{body}, \text{habitat}, \text{social}\})$ from $\emptyset$ (empty memory) to all four carriers active simultaneously.
- **3 Falsifiable Negative Controls**:
  1. `intact`: Standard lexical BM25 indexing and carrier retrieval with unperturbed query-memory grounding.
  2. `shuffled`: Preserves retrieved fact frequency but randomly permutes the association across carrier bins.
  3. `irrelevant`: Injects topically unrelated distractor entries into retrieval context.
  4. `erased`: Empties the retrieval store completely ($K=0$), measuring baseline unaided performance.
- **Standardized Benchmark Corpus**: 32 curated domain facts across 4 carriers matched against 16 distinct retrieval and grounding tasks.
- **Metrics**:
  - `hit@1`: Target fact retrieved at rank 1.
  - `hit@3`: Target fact retrieved within top 3 ranks.
  - `mrr`: Mean reciprocal rank ($\frac{1}{\text{rank}}$).
  - `grounded_score`: Token-level precision, recall, and harmonic F1 overlap between retrieved context and canonical task solution.
- **Leave-One-Task-Out (LOTO) Sensitivity**: Measures whether overall empirical carrier superiority is sensitive to any single idiosyncratic benchmark query.

---

## CLI & Verification Commands

```powershell
# Run full 1,024-trial causal factorial benchmark
python -m poseidon mco-experiment

# Offline receipt verification
python -m poseidon verify-mco outputs/mco_experiments/RECEIPT.json
```
