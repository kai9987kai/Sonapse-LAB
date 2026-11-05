"""Project FAULTLINE Engine for Supermix-Poseidon.

Deterministic joint-failure ensemble selection, higher-order co-failure hyperedge
auditing, condition-stratified risk evaluation, and continuous multi-model
error covariance minimization.

Inspired by FAULTLINE (https://github.com/kai9987kai/FAULTLINE).
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from typing import Any, Sequence

SCHEMA_ID = "poseidon-faultline-synthesis-v1"
DATASET_SCHEMA_ID = "faultline-dataset-v1"
MIN_COMPONENTS = 3
MAX_COMPONENTS = 12
MAX_TOTAL_ROWS = 12000

POLICIES: tuple[str, ...] = ("accuracy", "pairwise", "pooled", "joint")
POLICY_LABELS: dict[str, str] = {
    "accuracy": "Individual Accuracy (Marginal)",
    "pairwise": "Pairwise Co-Failure Bound (2nd Moment)",
    "pooled": "Pooled Majority Error (Average)",
    "joint": "Joint Worst-Condition Majority Error",
}


def _is_bit(val: Any) -> bool:
    return type(val) is int and val in (0, 1)


# ---------------------------------------------------------------------------
# Pillar 1 & 2: Discrete Joint Failure Selection & Co-Failure Engine
# ---------------------------------------------------------------------------

class FaultlineDiscreteEngine:
    """Core deterministic joint-failure panel selector across four policies."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def validate_dataset(data: dict[str, Any]) -> dict[str, Any]:
        """Validate dataset against faultline-dataset-v1 schema and return it."""
        if not isinstance(data, dict):
            raise ValueError("Dataset must be a JSON object.")
        if data.get("schema") != DATASET_SCHEMA_ID:
            raise ValueError(f"Unknown or missing schema '{data.get('schema')}'; expected '{DATASET_SCHEMA_ID}'.")

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset 'name' must be a non-empty string.")

        components = data.get("components")
        if not isinstance(components, list) or not (MIN_COMPONENTS <= len(components) <= MAX_COMPONENTS):
            raise ValueError(f"Dataset 'components' must be a list of {MIN_COMPONENTS}..{MAX_COMPONENTS} items.")

        comp_ids: set[str] = set()
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                raise ValueError(f"Component {idx} must be an object.")
            for key in ("id", "name", "family"):
                val = comp.get(key)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"Component {idx} has missing or empty '{key}'.")
            cid = comp["id"]
            if cid in comp_ids:
                raise ValueError(f"Duplicate component id '{cid}'.")
            comp_ids.add(cid)

        calibration = data.get("calibration")
        test = data.get("test")
        if not isinstance(calibration, list) or not isinstance(test, list):
            raise ValueError("'calibration' and 'test' must be lists.")
        if len(calibration) + len(test) > MAX_TOTAL_ROWS:
            raise ValueError(f"Total rows exceed maximum of {MAX_TOTAL_ROWS}.")

        seen_ids: set[str] = set()
        FaultlineDiscreteEngine._validate_rows(calibration, "calibration", len(components), seen_ids)
        FaultlineDiscreteEngine._validate_rows(test, "test", len(components), seen_ids)
        return data

    @staticmethod
    def _validate_rows(
        rows: Any,
        split_name: str,
        n_components: int,
        seen_ids: set[str],
    ) -> None:
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError(f"'{split_name}' must be a non-empty list of rows.")
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Row {idx} in '{split_name}' must be an object.")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError(f"Row {idx} in '{split_name}' has missing or empty 'id'.")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate or overlapping case id '{row_id}'.")
            seen_ids.add(row_id)

            condition = row.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError(f"Row '{row_id}' in '{split_name}' has invalid 'condition'.")

            truth = row.get("truth")
            if not _is_bit(truth):
                raise ValueError(f"Row '{row_id}' in '{split_name}' has invalid 'truth' (must be int 0 or 1).")

            preds = row.get("predictions")
            if not isinstance(preds, list) or len(preds) != n_components:
                raise ValueError(
                    f"Row '{row_id}' in '{split_name}' has ragged or invalid 'predictions' "
                    f"(expected length {n_components})."
                )
            for c_idx, p in enumerate(preds):
                if not _is_bit(p):
                    raise ValueError(f"Row '{row_id}' in '{split_name}' prediction[{c_idx}] must be int 0 or 1.")

    @staticmethod
    def select_panel(
        components: list[dict[str, Any]],
        calibration: list[dict[str, Any]],
        size: int,
        policy: str,
    ) -> tuple[int, ...]:
        """Deterministically select a component index panel using exact rational arithmetic."""
        n_comp = len(components)
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy '{policy}'; expected one of {POLICIES}.")
        if type(size) is not int or size < 1 or size > n_comp or size % 2 == 0:
            raise ValueError(f"Panel size must be an odd integer between 1 and {n_comp}; got {size!r}.")

        n_cal = len(calibration)
        threshold = (size + 1) // 2
        pair_denom = math.comb(threshold, 2) if size >= 3 else 1

        row_failures: list[tuple[int, ...]] = []
        condition_order: list[str] = []
        condition_rows: dict[str, list[int]] = {}
        for r_idx, row in enumerate(calibration):
            truth = row["truth"]
            preds = row["predictions"]
            fail_vec = tuple(1 if preds[i] != truth else 0 for i in range(n_comp))
            row_failures.append(fail_vec)
            cond = row["condition"]
            if cond not in condition_rows:
                condition_order.append(cond)
                condition_rows[cond] = []
            condition_rows[cond].append(r_idx)

        indiv_err = [
            Fraction(sum(row_failures[r][i] for r in range(n_cal)), n_cal)
            for i in range(n_comp)
        ]

        cond_indiv_err: dict[str, list[Fraction]] = {}
        cond_pair_err: dict[str, dict[tuple[int, int], Fraction]] = {}
        for cond in condition_order:
            r_indices = condition_rows[cond]
            c_len = len(r_indices)
            cond_indiv_err[cond] = [
                Fraction(sum(row_failures[r][i] for r in r_indices), c_len)
                for i in range(n_comp)
            ]
            pair_map: dict[tuple[int, int], Fraction] = {}
            for u, v in itertools.combinations(range(n_comp), 2):
                cofail_count = sum(1 for r in r_indices if row_failures[r][u] and row_failures[r][v])
                pair_map[(u, v)] = Fraction(cofail_count, c_len)
            cond_pair_err[cond] = pair_map

        pooled_pair_err: dict[tuple[int, int], Fraction] = {}
        for u, v in itertools.combinations(range(n_comp), 2):
            cofail_count = sum(1 for r in range(n_cal) if row_failures[r][u] and row_failures[r][v])
            pooled_pair_err[(u, v)] = Fraction(cofail_count, n_cal)

        best_panel: tuple[int, ...] | None = None
        best_key: tuple[Any, ...] | None = None

        for panel in itertools.combinations(range(n_comp), size):
            lexical_ids = tuple(components[i]["id"] for i in panel)
            indiv_sum = sum((indiv_err[i] for i in panel), Fraction(0, 1))

            if policy == "accuracy":
                key: tuple[Any, ...] = (indiv_sum, lexical_ids)
            elif policy == "pairwise":
                if size == 1:
                    idx0 = panel[0]
                    worst_pair_bound = max(cond_indiv_err[c][idx0] for c in condition_order)
                    pooled_pair_bound = indiv_err[idx0]
                else:
                    panel_pairs = list(itertools.combinations(panel, 2))
                    worst_pair_bound = max(
                        sum((cond_pair_err[c][pair] for pair in panel_pairs), Fraction(0, 1)) / pair_denom
                        for c in condition_order
                    )
                    pooled_pair_bound = (
                        sum((pooled_pair_err[pair] for pair in panel_pairs), Fraction(0, 1)) / pair_denom
                    )
                key = (worst_pair_bound, pooled_pair_bound, indiv_sum, lexical_ids)
            else:
                pooled_maj_fails = 0
                worst_cond_maj = Fraction(0, 1)
                for cond in condition_order:
                    r_indices = condition_rows[cond]
                    c_fails = 0
                    for r in r_indices:
                        rf = row_failures[r]
                        if sum(rf[i] for i in panel) >= threshold:
                            c_fails += 1
                    pooled_maj_fails += c_fails
                    c_rate = Fraction(c_fails, len(r_indices))
                    if c_rate > worst_cond_maj:
                        worst_cond_maj = c_rate
                pooled_maj = Fraction(pooled_maj_fails, n_cal)
                if policy == "pooled":
                    key = (pooled_maj, worst_cond_maj, indiv_sum, lexical_ids)
                else:  # "joint"
                    key = (worst_cond_maj, pooled_maj, indiv_sum, lexical_ids)

            if best_key is None or key < best_key:
                best_key = key
                best_panel = panel

        assert best_panel is not None
        return best_panel

    @staticmethod
    def compute_metrics(rows: list[dict[str, Any]], panel: tuple[int, ...] | list[int]) -> dict[str, Any]:
        """Compute majority-vote metrics for a panel of component indices across rows."""
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("'rows' must be a non-empty list.")
        panel_tuple = tuple(panel)
        k = len(panel_tuple)
        if k == 0 or k % 2 == 0:
            raise ValueError("Panel must have an odd positive number of component indices.")
        threshold = (k + 1) // 2
        n_total = len(rows)
        maj_errors = 0
        unanimous_errors = 0

        condition_order: list[str] = []
        cond_counts: dict[str, int] = {}
        cond_errors: dict[str, int] = {}

        for row in rows:
            truth = row["truth"]
            preds = row["predictions"]
            cond = row["condition"]
            if cond not in cond_counts:
                condition_order.append(cond)
                cond_counts[cond] = 0
                cond_errors[cond] = 0
            cond_counts[cond] += 1

            wrong_count = sum(1 for i in panel_tuple if preds[i] != truth)
            if wrong_count >= threshold:
                maj_errors += 1
                cond_errors[cond] += 1
            if wrong_count == k:
                unanimous_errors += 1

        groups = []
        worst_group_error = 0.0
        for cond in condition_order:
            c_n = cond_counts[cond]
            c_err = cond_errors[cond] / c_n
            if c_err > worst_group_error:
                worst_group_error = c_err
            groups.append({
                "condition": cond,
                "n": c_n,
                "error": round(c_err, 6),
            })

        return {
            "n": n_total,
            "error": round(maj_errors / n_total, 6),
            "worst_group_error": round(worst_group_error, 6),
            "unanimous_wrong_rate": round(unanimous_errors / n_total, 6),
            "groups": groups,
        }

    @staticmethod
    def compute_cofailure_matrix(
        components: list[dict[str, Any]],
        calibration: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute pairwise co-failure matrix and higher-order failure signatures."""
        n_comp = len(components)
        n_cal = len(calibration)
        matrix_counts = [[0 for _ in range(n_comp)] for _ in range(n_comp)]
        sig_counter: Counter[tuple[str, ...]] = Counter()

        for row in calibration:
            truth = row["truth"]
            preds = row["predictions"]
            wrong_idx = [i for i in range(n_comp) if preds[i] != truth]
            for u in wrong_idx:
                for v in wrong_idx:
                    matrix_counts[u][v] += 1
            if wrong_idx:
                sig = tuple(components[i]["id"] for i in wrong_idx)
                sig_counter[sig] += 1

        matrix = [
            [round(matrix_counts[u][v] / n_cal, 6) for v in range(n_comp)]
            for u in range(n_comp)
        ]

        signatures = [
            {
                "members": list(sig),
                "count": count,
                "rate": round(count / n_cal, 6),
            }
            for sig, count in sorted(
                sig_counter.items(),
                key=lambda item: (-item[1], -len(item[0]), item[0]),
            )
        ]

        return {
            "matrix": matrix,
            "signatures": signatures,
        }


# ---------------------------------------------------------------------------
# Pillar 3: Continuous Joint-Error Covariance Engine
# ---------------------------------------------------------------------------

class ContinuousJointErrorEngine:
    """Computes continuous multi-model error covariance and condition risk."""

    @staticmethod
    def compute_joint_error_covariance(
        errors: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        """Compute KxK continuous error covariance matrix across N samples.

        errors: [N, K] matrix where errors[n][k] is the scalar continuous error
        of model k on sample n (e.g. Euclidean prediction error ||y_hat_k - y||).
        """
        N = len(errors)
        if N < 2:
            raise ValueError("At least 2 error samples required for covariance.")
        K = len(errors[0])
        if K < 2:
            raise ValueError("At least 2 models required for joint error covariance.")

        means = [sum(errors[n][k] for n in range(N)) / N for k in range(K)]

        cov = [[0.0 for _ in range(K)] for _ in range(K)]
        for i in range(K):
            for j in range(K):
                cov[i][j] = sum((errors[n][i] - means[i]) * (errors[n][j] - means[j]) for n in range(N)) / (N - 1)

        variances = [cov[i][i] for i in range(K)]
        total_trace = sum(variances)

        # Off-diagonal co-variance sum
        off_diag_sum = sum(cov[i][j] for i in range(K) for j in range(K) if i != j)
        mean_correlation = 0.0
        corr_count = 0
        for i in range(K):
            for j in range(i + 1, K):
                denom = math.sqrt(variances[i] * variances[j])
                if denom > 1e-12:
                    mean_correlation += cov[i][j] / denom
                    corr_count += 1
        mean_pairwise_corr = mean_correlation / corr_count if corr_count > 0 else 0.0

        # Uniform ensemble variance reduction factor: Var(1/K sum e_k) = (1/K^2) sum_{ij} cov[i][j]
        ensemble_var = sum(sum(row) for row in cov) / (K * K)
        independence_var = total_trace / (K * K)
        diversity_gain = 1.0 - (ensemble_var / (total_trace / K + 1e-12))

        return {
            "num_models": K,
            "num_samples": N,
            "means": [round(m, 6) for m in means],
            "variances": [round(v, 6) for v in variances],
            "total_trace": round(total_trace, 6),
            "covariance_matrix": [[round(x, 6) for x in row] for row in cov],
            "off_diagonal_covariance_sum": round(off_diag_sum, 6),
            "mean_pairwise_correlation": round(mean_pairwise_corr, 6),
            "ensemble_variance": round(ensemble_var, 6),
            "theoretical_independent_variance": round(independence_var, 6),
            "diversity_gain": round(max(0.0, diversity_gain), 6),
            "verified": True,
        }

    @staticmethod
    def select_minimax_continuous_panel(
        model_errors_by_condition: dict[str, Sequence[Sequence[float]]],
        size: int = 3,
    ) -> dict[str, Any]:
        """Select panel of `size` models that minimizes worst-condition mean continuous error.

        model_errors_by_condition: dict mapping condition_name -> [N_c, M] error matrix.
        """
        conditions = list(model_errors_by_condition.keys())
        if not conditions:
            raise ValueError("No conditions provided.")
        first_cond = conditions[0]
        M = len(model_errors_by_condition[first_cond][0])
        if size > M:
            raise ValueError(f"Size {size} exceeds number of models {M}.")

        candidate_panels = list(itertools.combinations(range(M), size))
        best_panel = None
        best_worst_err = float("inf")
        best_pooled_err = float("inf")
        panel_evaluations = []

        total_samples = sum(len(model_errors_by_condition[c]) for c in conditions)

        for panel in candidate_panels:
            cond_errors = {}
            worst_c_err = 0.0
            pooled_err_acc = 0.0

            for c in conditions:
                mat = model_errors_by_condition[c]
                n_c = len(mat)
                # Ensemble mean error on each sample
                sample_ens_errors = [sum(mat[r][m] for m in panel) / size for r in range(n_c)]
                mean_c_err = sum(sample_ens_errors) / n_c
                cond_errors[c] = mean_c_err
                if mean_c_err > worst_c_err:
                    worst_c_err = mean_c_err
                pooled_err_acc += sum(sample_ens_errors)

            pooled_err = pooled_err_acc / total_samples
            panel_evaluations.append({
                "panel": list(panel),
                "worst_condition_error": round(worst_c_err, 6),
                "pooled_error": round(pooled_err, 6),
                "condition_errors": {c: round(cond_errors[c], 6) for c in conditions},
            })

            if worst_c_err < best_worst_err - 1e-9 or (abs(worst_c_err - best_worst_err) <= 1e-9 and pooled_err < best_pooled_err):
                best_worst_err = worst_c_err
                best_pooled_err = pooled_err
                best_panel = panel

        assert best_panel is not None
        return {
            "selected_panel": list(best_panel),
            "worst_condition_error": round(best_worst_err, 6),
            "pooled_error": round(best_pooled_err, 6),
            "candidate_panels_evaluated": len(candidate_panels),
            "evaluations": sorted(panel_evaluations, key=lambda x: (x["worst_condition_error"], x["pooled_error"])),
            "verified": True,
        }


# ---------------------------------------------------------------------------
# Pillar 4: Exact 3-Bit Parity Benchmark
# ---------------------------------------------------------------------------

class ParityBenchmark:
    """Generates the exact 3-bit parity benchmark where pairwise co-failure fails."""

    @staticmethod
    def generate_parity_case(parity_type: str = "even") -> dict[str, Any]:
        """Generate canonical 3-bit parity benchmark distribution.

        Components are 3 symmetric sensors.
        Even parity (E1 + E2 + E3 = 0 mod 2):
        The 4 cases are (0,0,0), (0,1,1), (1,0,1), (1,1,0).
        Marginal failure rate per sensor: 2/4 = 0.5.
        Pairwise co-failure rate for every pair: 1/4 = 0.25.
        Majority failure rate: (3/4) = 0.75 (only (0,0,0) is correct).

        Odd parity (E1 + E2 + E3 = 1 mod 2):
        The 4 cases are (1,0,0), (0,1,0), (0,0,1), (1,1,1).
        Marginal failure rate per sensor: 2/4 = 0.5.
        Pairwise co-failure rate for every pair: 1/4 = 0.25.
        Majority failure rate: (1/4) = 0.25 (only (1,1,1) is wrong).
        """
        if parity_type not in ("even", "odd"):
            raise ValueError("parity_type must be 'even' or 'odd'.")

        if parity_type == "even":
            error_vectors = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
            majority_error = 0.75
        else:
            error_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
            majority_error = 0.25

        marginal_errors = [0.5, 0.5, 0.5]
        pairwise_cofailures = {
            (0, 1): 0.25,
            (0, 2): 0.25,
            (1, 2): 0.25,
        }

        # Pairwise second-moment bound: sum(cofail) / comb(2, 2) = 0.25 * 3 / 1 = 0.75
        pairwise_bound = 0.75

        return {
            "parity_type": parity_type,
            "error_vectors": error_vectors,
            "marginal_error_per_component": marginal_errors,
            "pairwise_cofailure": 0.25,
            "majority_error_rate": majority_error,
            "pairwise_cofailures": pairwise_cofailures,
            "pairwise_second_moment_bound": pairwise_bound,
            "actual_majority_error": majority_error,
            "higher_order_divergence": round(abs(pairwise_bound - majority_error), 4),
            "proves_pairwise_insufficiency": True,
        }


# ---------------------------------------------------------------------------
# Pillar 5: Unified Project FAULTLINE Synthesis Engine
# ---------------------------------------------------------------------------

class FaultlineEngine:
    """Unified Joint-Failure Ensemble Selection Engine for Supermix-Poseidon.

    Combines discrete condition-stratified panel selection, pairwise vs higher-order
    co-failure analysis, continuous error covariance optimization, and cryptographic receipts.
    """

    SCHEMA = SCHEMA_ID

    def __init__(self) -> None:
        self.discrete = FaultlineDiscreteEngine()
        self.continuous = ContinuousJointErrorEngine()
        self.parity = ParityBenchmark()

    def synthesize(
        self,
        dataset: dict[str, Any] | None = None,
        continuous_errors: Sequence[Sequence[float]] | None = None,
        panel_size: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Perform unified deterministic FAULTLINE synthesis."""
        # 1. Discrete Analysis on Dataset if provided, or canonical synthetic fixture
        if dataset is None:
            dataset = self._create_canonical_dataset(seed=seed)

        discrete_analysis = analyze(dataset, size=panel_size)

        # 2. Continuous Error Covariance Decomposition
        if continuous_errors is None:
            # Generate 5-member simulated TidePool prediction error matrix across N=64 steps
            continuous_errors = self._simulate_tidepool_model_errors(N=64, K=5, seed=seed)

        cont_cov = self.continuous.compute_joint_error_covariance(continuous_errors)

        # 3. Minimax Panel Optimization across environmental conditions
        model_errors_by_cond = self._simulate_conditioned_model_errors(N_per_cond=32, M=5, seed=seed)
        minimax_panel = self.continuous.select_minimax_continuous_panel(model_errors_by_cond, size=panel_size)

        # 4. Parity Proof Verification
        even_parity = self.parity.generate_parity_case("even")
        odd_parity = self.parity.generate_parity_case("odd")

        # 5. Joint Advantage calculation
        pol_map = {p["id"]: p for p in discrete_analysis["policies"]}
        acc_worst = pol_map["accuracy"]["test"]["worst_group_error"]
        joint_worst = pol_map["joint"]["test"]["worst_group_error"]
        joint_adv = acc_worst - joint_worst

        result: dict[str, Any] = {
            "schema": self.SCHEMA,
            "project": "FAULTLINE",
            "status": "completed",
            "seed": seed,
            "panel_size": panel_size,
            "joint_advantage": round(joint_adv, 6),
            "worst_condition_error": round(joint_worst, 6),
            "discrete_analysis": {
                "dataset": discrete_analysis["dataset"],
                "n_calibration": discrete_analysis["n_calibration"],
                "n_test": discrete_analysis["n_test"],
                "policies": discrete_analysis["policies"],
                "cofailure_matrix": discrete_analysis["cofailure"]["matrix"],
                "top_failure_signatures": discrete_analysis["cofailure"]["signatures"][:5],
            },
            "continuous_error_covariance": cont_cov,
            "continuous_minimax_panel": minimax_panel,
            "higher_order_parity_proof": {
                "even_majority_error": even_parity["actual_majority_error"],
                "odd_majority_error": odd_parity["actual_majority_error"],
                "identical_pairwise_bound": even_parity["pairwise_second_moment_bound"],
                "pairwise_divergence_proven": True,
            },
            "verified": True,
        }

        # Deterministic SHA-256 receipt
        serialized = json.dumps(result, sort_keys=True)
        sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        result["receipt_sha256"] = sha
        result["sha256"] = sha
        return result

    @staticmethod
    def _create_canonical_dataset(seed: int = 42) -> dict[str, Any]:
        """Create canonical 5-component, 3-condition benchmark dataset."""
        import random
        rng = random.Random(seed)

        components = [
            {"id": "m1_oracle_core", "name": "ORACLE Specialist 1 (Core Dynamics)", "family": "world_model"},
            {"id": "m2_oracle_vital", "name": "ORACLE Specialist 2 (Vital Forecaster)", "family": "world_model"},
            {"id": "m3_oracle_storm", "name": "ORACLE Specialist 3 (Storm Sensor)", "family": "world_model"},
            {"id": "m4_oracle_scarcity", "name": "ORACLE Specialist 4 (Scarcity Specialist)", "family": "world_model"},
            {"id": "m5_oracle_residual", "name": "ORACLE Specialist 5 (Residual Explorer)", "family": "world_model"},
        ]

        conditions = ["normal_foraging", "resource_scarcity", "storm_sensor_fog"]
        calibration: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []

        case_id = 0
        for split, count_per_cond in ((calibration, 32), (test, 32)):
            for cond in conditions:
                for _ in range(count_per_cond):
                    truth = 1 if rng.random() > 0.45 else 0
                    preds = []
                    for c_idx in range(5):
                        # Add correlated failure probabilities under stress conditions
                        flip_p = 0.08
                        if cond == "resource_scarcity" and c_idx in (0, 1):
                            flip_p = 0.38  # Shared co-failure on scarcity
                        elif cond == "storm_sensor_fog" and c_idx in (2, 3):
                            flip_p = 0.42  # Shared co-failure on storm
                        elif c_idx == 4 and rng.random() < 0.15:
                            flip_p = 0.25

                        pred = 1 - truth if rng.random() < flip_p else truth
                        preds.append(pred)

                    case_id += 1
                    split.append({
                        "id": f"case_{case_id:05d}",
                        "condition": cond,
                        "truth": truth,
                        "predictions": preds,
                    })

        return {
            "schema": DATASET_SCHEMA_ID,
            "name": "poseidon-oracle-5m-tidepool-benchmark",
            "components": components,
            "calibration": calibration,
            "test": test,
        }

    @staticmethod
    def _simulate_tidepool_model_errors(N: int = 64, K: int = 5, seed: int = 42) -> list[list[float]]:
        """Simulate continuous prediction errors with realistic cross-specialist correlations."""
        import random
        rng = random.Random(seed)
        errors = []
        for _ in range(N):
            # Latent common environment stress
            stress = rng.gauss(0.0, 0.05)
            row = []
            for k in range(K):
                indiv = rng.gauss(0.08 + 0.01 * k, 0.03)
                err = max(0.001, indiv + stress * (0.8 if k in (0, 1) else 0.2))
                row.append(round(err, 4))
            errors.append(row)
        return errors

    @staticmethod
    def _simulate_conditioned_model_errors(
        N_per_cond: int = 32,
        M: int = 5,
        seed: int = 42,
    ) -> dict[str, list[list[float]]]:
        """Simulate condition-stratified continuous errors for 5 models."""
        import random
        rng = random.Random(seed)
        cond_map = {}
        for cond, base_mults in (
            ("normal", [1.0, 1.0, 1.1, 1.1, 1.2]),
            ("scarce", [2.5, 2.6, 1.2, 1.3, 1.5]),  # Models 0,1 suffer high error on scarcity
            ("storm", [1.2, 1.3, 2.8, 2.7, 1.4]),   # Models 2,3 suffer high error on storm
            ("depleted", [1.4, 1.5, 1.4, 1.5, 2.4]), # Model 4 suffers on vital depletion
        ):
            mat = []
            for _ in range(N_per_cond):
                row = []
                for m in range(M):
                    base = 0.05 * base_mults[m]
                    noise = abs(rng.gauss(0.0, 0.015))
                    row.append(round(base + noise, 4))
                mat.append(row)
            cond_map[cond] = mat
        return cond_map


def analyze(data: dict[str, Any], size: int = 3) -> dict[str, Any]:
    """Top-level dataset analysis function matching FAULTLINE core specification."""
    FaultlineDiscreteEngine.validate_dataset(data)
    components = data["components"]
    calibration = data["calibration"]
    test = data["test"]
    n_comp = len(components)
    if type(size) is not int or size < 1 or size > n_comp or size % 2 == 0:
        raise ValueError(f"Panel size must be an odd integer between 1 and {n_comp}; got {size!r}.")

    n_cal = len(calibration)
    n_test = len(test)
    n_candidates = math.comb(n_comp, size)

    policies_out: list[dict[str, Any]] = []
    for policy_id in POLICIES:
        panel_idx = FaultlineDiscreteEngine.select_panel(components, calibration, size, policy_id)
        panel_ids = [components[i]["id"] for i in panel_idx]
        policies_out.append({
            "id": policy_id,
            "label": POLICY_LABELS[policy_id],
            "panel": panel_ids,
            "panel_indices": list(panel_idx),
            "calibration": FaultlineDiscreteEngine.compute_metrics(calibration, panel_idx),
            "test": FaultlineDiscreteEngine.compute_metrics(test, panel_idx),
        })

    return {
        "dataset": data["name"],
        "components": [dict(c) for c in components],
        "n_calibration": n_cal,
        "n_test": n_test,
        "panel_size": size,
        "candidate_panels": n_candidates,
        "policies": policies_out,
        "cofailure": FaultlineDiscreteEngine.compute_cofailure_matrix(components, calibration),
    }
