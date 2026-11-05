# TESSERA Ratification Study Results v0.8

This document records the empirical results of the multi-lineage ratification and finite-domain safety study under the Tessera macro-action commons.

---

## 1. Study Specification & Artifacts

- **Experiment Schema**: `poseidon-tessera-experiment-v1`
- **Experiment ID**: `82ec1a7e61a61ed66cd0a07d53860dac`
- **Receipt SHA-256**: `9015df9d97fb532b7add894369876c009979c6461bed0c404d36a5888c065b25`
- **Artifact Path**: `outputs/tessera_experiments/RECEIPT.json`
- **Seeds Evaluated**: `[155000001, 155000002, 155000003, 155000004]`
- **Ratified Opcodes**: 2
- **Quarantined Candidates**: 1

---

## 2. Instruction Set Inventory

### Ratified Opcodes (Multi-Lineage Reinforcement)
1. **`OP_01` (`[1, 1]`)**:
   - Discovered independently by `seed_155000001` and `seed_155000003`.
   - Sequence: `[forage, forage]`.
   - Mean reward delta: $+1.35$.
   - Safety validation: Passed stamina exhaustion invariant ($\Delta \text{stam} = -0.16 \ge -0.85$).
2. **`OP_02` (`[2, 0]`)**:
   - Discovered independently by `seed_155000002` and `seed_155000004`.
   - Sequence: `[drink, rest]`.
   - Mean reward delta: $+0.95$.
   - Safety validation: Passed stamina exhaustion invariant (net stamina gain $+0.12$).

### Quarantined Candidate (Single-Lineage Discovery)
- **`QUARANTINED_01` (`[4, 4, 1]`)**:
  - Discovered only by `seed_155000001`.
  - Quarantined: Blocked from global execution commons pending independent confirmation by a distinct lineage line.
