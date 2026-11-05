# TESSERA Lineage-Ratified Macro-Action Commons v0.8

TESSERA is a decentralized macro-action commons architecture adapted for TidePool decision-making. It solves the vulnerability of greedy sub-goal heuristics by enforcing dual-lineage ratification, finite-domain semantic safety pre-checks, and epoch-based instruction set expiration.

---

## 1. Core Principles & Safeguards

### 1.1 Dual-Lineage Ratification Invariant
In reinforcement and imitation learning, single trajectories frequently discover fluke macro sequences that yield high episodic rewards through stochastic luck rather than genuine generalizability.
- **Quarantine Tier**: When a candidate macro-action sequence $S = [a_1, a_2, \dots, a_L]$ ($2 \le L \le 6$) is discovered by an agent lineage $L_A$ with reward improvement $\Delta R > 0$, it is assigned to `QUARANTINE`. It cannot be executed globally.
- **Ratification Threshold**: Only when an independent lineage $L_B$ ($L_B \neq L_A$) independently submits the identical action sequence with positive empirical reward advantage is the candidate ratified into the shared opcode table:
  $$\text{Promote to } \text{OP\_* } \iff \exists L_i, L_j \text{ such that } L_i \neq L_j \land \Delta R_i > 0 \land \Delta R_j > 0$$

### 1.2 Finite-Domain Semantic Safety Pre-Checks
Before any macro sequence can be executed or ratified, it must pass three deterministic invariant checks:
1. **Length Bound**: $2 \le \operatorname{len}(S) \le 6$. Sequences exceeding 6 steps are rejected to prevent unmonitored open-loop runaway drift.
2. **Action Domain Validity**: $\forall a \in S, \; a \in \{0, 1, 2, 3, 4, 5\}$.
3. **Stamina Exhaustion Guard**: Sequences cannot contain consecutive strenuous travel actions ($a \in \{4, 5\}$) without interspersed rest ($a=0$) that would exhaust stamina below $0.15$.

### 1.3 Epoch Expiration & Pruning
Ratified opcodes are not permanent. Each opcode has an epoch lifespan (default: 5 epochs). If an opcode is not independently reinforced with positive utility during an active epoch, its confidence decays and it is safely pruned from the active opcode dictionary.

---

## 2. Opcode Table Structure

| Opcode | Sequence | Source Lineages | Status | Description |
|---|---|---|---|---|
| `OP_FORAGE_BURST` | `[1, 1]` | `lineage_01`, `lineage_03` | RATIFIED | Concentrated resource harvesting at high-density food patches |
| `OP_DRINK_REST` | `[2, 0]` | `lineage_02`, `lineage_04` | RATIFIED | Hydration intake followed by metabolic recovery rest |
| `OP_TACTICAL_RETREAT` | `[5, 0]` | `lineage_01`, `lineage_05` | RATIFIED | Emergency flee followed by exhaustion stabilization |
| `CANDIDATE_0x4F` | `[4, 4, 1]` | `lineage_02` | QUARANTINED | Single-lineage exploration candidate pending independent ratification |

---

## 3. CLI & Verification Commands

```powershell
# Run the multi-lineage ratification experiment
python -m poseidon tessera-experiment --seeds 160000001 160000002 160000003 160000004

# Replay and verify the Tessera receipt offline
python -m poseidon verify-tessera outputs/tessera_experiments/RECEIPT.json
```
