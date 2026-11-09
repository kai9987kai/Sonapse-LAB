"""Measured sandbox for SONAPSE-EVO circuit mutations.

Replaces hand-written ``delta_reward`` formulas with an actual measurement.

Host circuit
    A Dale's-law recurrent rate network (leaky tanh units, ~80% excitatory
    presynaptic columns) driven by a scalar input stream.

Task / reward
    Short-term memory capacity (Jaeger 2001): a ridge readout is fitted on the
    first half of a rollout to reconstruct u(t-k) for k = 1..K plus one
    nonlinear product term u(t-1)*u(t-2); the reward is the *sum* of held-out
    R^2 scores (clipped to [0, 1] each). Higher means the circuit retains more
    usable information about its recent input.

Causal estimate
    Every candidate is scored against the current host with paired seeds:
    both networks see the identical input sequence and initial state, so
    ``delta = R(candidate) - R(host)`` isolates the structural change. The
    mean and standard error over seeds are reported; callers can require the
    lower confidence bound to be positive before promotion.

Scope / honesty
    This is one synthetic benchmark. A positive delta means "more memory
    capacity on this task", not general capability. All RNG is local, so the
    sandbox never perturbs the global NumPy / PyTorch random streams used by
    the live agent.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclasses.dataclass
class SandboxMeasurement:
    delta_mean: float
    delta_se: float
    baseline_mean: float
    candidate_mean: float
    per_seed_delta: List[float]
    applied: bool
    host_nodes: int
    candidate_nodes: int
    note: str = ""

    def lcb(self, z: float = 1.0) -> float:
        return float(self.delta_mean - z * self.delta_se)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_mean": round(self.delta_mean, 5),
            "delta_se": round(self.delta_se, 5),
            "baseline_reward": round(self.baseline_mean, 5),
            "candidate_reward": round(self.candidate_mean, 5),
            "per_seed_delta": [round(d, 5) for d in self.per_seed_delta],
            "applied": self.applied,
            "host_nodes": self.host_nodes,
            "candidate_nodes": self.candidate_nodes,
            "note": self.note,
        }


class RateCircuitSandbox:
    """Dale's-law recurrent host circuit with a measurable memory-capacity reward."""

    def __init__(
        self,
        num_nodes: int = 48,
        excitatory_ratio: float = 0.8,
        spectral_radius: float = 0.9,
        leak: float = 0.6,
        density: float = 0.2,
        rollout_steps: int = 400,
        washout: int = 50,
        max_delay: int = 6,
        ridge: float = 1e-3,
        seed: int = 42,
        max_nodes: int = 160,
    ):
        self.leak = float(leak)
        self.rollout_steps = int(rollout_steps)
        self.washout = int(washout)
        self.max_delay = int(max_delay)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.max_nodes = int(max_nodes)
        self.target_radius = float(spectral_radius)

        rng = np.random.default_rng(seed)
        n = int(num_nodes)
        num_inh = max(1, round(n * (1.0 - excitatory_ratio)))
        self.signs = np.ones(n)
        self.signs[-num_inh:] = -1.0
        mag = np.abs(rng.normal(0.0, 1.0, (n, n))) * (rng.random((n, n)) < density)
        np.fill_diagonal(mag, 0.0)
        W = mag * self.signs[None, :]  # column j carries presynaptic sign of neuron j
        self.W = self._scale_radius(W, self.target_radius)
        self.w_in = rng.uniform(-1.0, 1.0, n)
        self._stack: List[Dict[str, np.ndarray]] = []
        self.commits = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _radius(W: np.ndarray) -> float:
        if W.size == 0:
            return 0.0
        return float(np.max(np.abs(np.linalg.eigvals(W))))

    def _scale_radius(self, W: np.ndarray, target: float) -> np.ndarray:
        r = self._radius(W)
        return W if r < 1e-9 else W * (target / r)

    @property
    def num_nodes(self) -> int:
        return int(self.W.shape[0])

    def dales_law_ok(self, W: Optional[np.ndarray] = None, signs: Optional[np.ndarray] = None) -> bool:
        W = self.W if W is None else W
        signs = self.signs if signs is None else signs
        exc, inh = signs > 0, signs < 0
        return bool(np.all(W[:, exc] >= -1e-12) and np.all(W[:, inh] <= 1e-12))

    @staticmethod
    def _proposal_seed(proposal: Any) -> int:
        digest = hashlib.sha256(str(getattr(proposal, "proposal_id", "p")).encode()).digest()
        return int.from_bytes(digest[:4], "little")

    # -------------------------------------------------------------- mutations
    def _apply(self, proposal: Any):
        """Returns (W, w_in, signs, applied, note) for the mutated circuit; host untouched."""
        W, w_in, signs = self.W.copy(), self.w_in.copy(), self.signs.copy()
        mtype = getattr(proposal, "mutation_type", "")
        weights = getattr(proposal, "weights", None)
        if weights is not None and hasattr(weights, "detach"):
            weights = weights.detach().cpu().numpy()
        rng = np.random.default_rng(self._proposal_seed(proposal))

        if mtype == "motif_duplication" and weights is not None:
            k = int(weights.shape[0])
            n = W.shape[0]
            if n + k > self.max_nodes:
                return W, w_in, signs, False, f"node budget {self.max_nodes} reached"
            block = np.asarray(weights, dtype=np.float64)
            # Paralog presynaptic signs come from the proposal's column signs.
            col_sum = block.sum(axis=0)
            p_signs = np.where(col_sum < 0, -1.0, 1.0)
            block = np.abs(block) * p_signs[None, :]
            # Sparse Dale-consistent coupling: host->paralog and paralog->host.
            scale = float(np.mean(np.abs(W[W != 0]))) if np.any(W != 0) else 0.1
            h2p = np.abs(rng.normal(0, scale, (k, n))) * (rng.random((k, n)) < 0.15) * signs[None, :]
            p2h = np.abs(rng.normal(0, scale, (n, k))) * (rng.random((n, k)) < 0.15) * p_signs[None, :]
            W = np.block([[W, p2h], [h2p, block]])
            signs = np.concatenate([signs, p_signs])
            w_in = np.concatenate([w_in, rng.uniform(-1.0, 1.0, k)])
            radius = self._radius(W)
            if radius > 0.99:
                W *= 0.99 / radius
            return W, w_in, signs, True, f"duplicated {k}-node motif"

        if mtype == "topological_rewire" and weights is not None:
            m = min(int(weights.shape[0]), W.shape[0])
            delta = np.asarray(weights, dtype=np.float64)[:m, :m]
            n_edges = max(1, int(getattr(proposal, "delta_edges", 2)))
            flat = np.argsort(-np.abs(delta).ravel())[:n_edges]
            rows, cols = np.unravel_index(flat, (m, m))
            for r, c in zip(rows, cols):
                if r == c:
                    continue
                W[r, c] = signs[c] * abs(W[r, c] + delta[r, c])  # keep presynaptic sign
            radius = self._radius(W)
            if radius > 0.99:
                W *= 0.99 / radius
            return W, w_in, signs, True, f"rewired {len(rows)} edges"

        return W, w_in, signs, False, f"mutation type '{mtype}' not supported by sandbox (no-op)"

    # ----------------------------------------------------------------- reward
    def _rollout_reward(self, W: np.ndarray, w_in: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        T = self.rollout_steps
        u = rng.uniform(-1.0, 1.0, T)
        x0 = rng.normal(0.0, 0.1, self.max_nodes)  # shared prefix -> identical for paired nets
        n = W.shape[0]
        x = x0[:n].copy()
        states = np.empty((T, n))
        for t in range(T):
            x = (1.0 - self.leak) * x + self.leak * np.tanh(W @ x + w_in * u[t])
            states[t] = x
        lo = max(self.washout, self.max_delay + 2)
        X = np.hstack([states[lo:], np.ones((T - lo, 1))])
        targets = [u[lo - k:T - k] for k in range(1, self.max_delay + 1)]
        targets.append(u[lo - 1:T - 1] * u[lo - 2:T - 2])
        Y = np.stack(targets, axis=1)
        half = X.shape[0] // 2
        Xtr, Xte, Ytr, Yte = X[:half], X[half:], Y[:half], Y[half:]
        beta = np.linalg.solve(Xtr.T @ Xtr + self.ridge * np.eye(X.shape[1]), Xtr.T @ Ytr)
        pred = Xte @ beta
        ss_res = np.sum((Yte - pred) ** 2, axis=0)
        ss_tot = np.sum((Yte - Yte.mean(axis=0)) ** 2, axis=0) + 1e-12
        r2 = np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)
        return float(np.sum(r2))

    def host_reward(self, seeds: Sequence[int]) -> float:
        return float(np.mean([self._rollout_reward(self.W, self.w_in, s) for s in seeds]))

    def evaluation_seeds(self, num_seeds: int, round_index: int = 0) -> List[int]:
        return [self.seed * 1000 + round_index * 97 + i for i in range(int(num_seeds))]

    # ------------------------------------------------------------- interface
    def evaluate(self, proposal: Any, seeds: Sequence[int]) -> SandboxMeasurement:
        Wc, w_in_c, signs_c, applied, note = self._apply(proposal)
        if not self.dales_law_ok(Wc, signs_c):
            raise AssertionError("sandbox mutation violated Dale's law")
        base = [self._rollout_reward(self.W, self.w_in, s) for s in seeds]
        if applied:
            cand = [self._rollout_reward(Wc, w_in_c, s) for s in seeds]
        else:
            cand = list(base)
        deltas = np.asarray(cand) - np.asarray(base)
        se = float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("inf")
        return SandboxMeasurement(
            delta_mean=float(deltas.mean()),
            delta_se=se,
            baseline_mean=float(np.mean(base)),
            candidate_mean=float(np.mean(cand)),
            per_seed_delta=deltas.tolist(),
            applied=applied,
            host_nodes=self.num_nodes,
            candidate_nodes=int(Wc.shape[0]),
            note=note,
        )

    def commit(self, proposal: Any) -> bool:
        """Applies a promoted mutation to the host. Previous host kept for rollback."""
        Wc, w_in_c, signs_c, applied, _ = self._apply(proposal)
        if not applied:
            return False
        self._stack.append({"W": self.W, "w_in": self.w_in, "signs": self.signs})
        self._stack = self._stack[-16:]
        self.W, self.w_in, self.signs = Wc, w_in_c, signs_c
        self.commits += 1
        return True

    def rollback(self) -> bool:
        if not self._stack:
            return False
        prev = self._stack.pop()
        self.W, self.w_in, self.signs = prev["W"], prev["w_in"], prev["signs"]
        self.commits -= 1
        return True

    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.W, self.w_in, self.signs):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()

    def status(self) -> Dict[str, Any]:
        return {
            "host_nodes": self.num_nodes,
            "excitatory_fraction": round(float(np.mean(self.signs > 0)), 4),
            "spectral_radius": round(self._radius(self.W), 4),
            "dales_law_ok": self.dales_law_ok(),
            "committed_mutations": self.commits,
            "rollback_depth": len(self._stack),
            "reward_definition": "sum of held-out R^2: u(t-k), k=1..%d, plus u(t-1)*u(t-2)" % self.max_delay,
            "host_digest": self.digest()[:16],
        }
