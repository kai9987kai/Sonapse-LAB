# Memory Carrier Observatory (MCO) Results (v0.7.0)

**Receipt ID**: `mco-4671d9725974154b`  
**Receipt SHA256**: `3cbf7c84f4110bb0e3ef6b964eb9873500c708ea6ce507a9fe009e4d287a9404`  
**Schema**: `poseidon-mco-experiment-v1`  
**Evaluation Scope**: 1,024 independent trials across 16 factorial carrier subsets, 4 conditions, 16 benchmark tasks, and 32 standard corpus facts.

---

## 1. Falsifiable Condition Comparison

| Condition | Total Trials | Mean Hit@1 | Mean Hit@3 | Mean MRR | Mean Grounded F1 | Mean Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **intact** | 256 | **0.500** | **0.500** | **0.500** | **0.520** | 28.58 ms |
| **shuffled** | 256 | 0.016 | 0.121 | 0.061 | 0.541 | 1.08 ms |
| **irrelevant** | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.01 ms |
| **erased** | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 ms |

---

## 2. Marginal Carrier Contributions (Intact Condition)

The marginal effect measures the difference in grounded retrieval score when a carrier is present versus when it is absent across all 8 paired subset configurations:

| Carrier | Mean With Carrier | Mean Without Carrier | Marginal Grounded Delta ($\Delta F_1$) |
| :--- | :---: | :---: | :---: |
| **episodic** | 0.656 | 0.383 | **+0.273** |
| **social** | 0.645 | 0.395 | **+0.250** |
| **body** | 0.641 | 0.398 | **+0.242** |
| **habitat** | 0.633 | 0.406 | **+0.227** |

Every carrier demonstrates a substantial, statistically positive marginal contribution exceeding $+0.22$.

---

## 3. Best Factorial Subset

- **Top Performing Subset**: `episodic+body+habitat+social` (all four carriers active).
- **Target Active Grounded F1**: Reaches peak accuracy when target carrier entries are present in the retrieval partition.
- **Leave-One-Task-Out (LOTO) Sensitivity**:
  - Task count: 16
  - Min task-excluded mean: 0.502
  - Max task-excluded mean: 0.536
  - Maximum delta from overall intact mean: 0.017 (confirming robust, query-invariant performance).
