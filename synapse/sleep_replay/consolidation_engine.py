"""Multi-Scale Sleep Consolidation, Sharp-Wave Ripple (SWR) Replay & Synaptic Homeostasis (SHY).

Implements biological sleep-dependent memory consolidation mechanisms:
1. Tononi & Cirelli's Synaptic Homeostasis Hypothesis (SHY):
   - Multiplicative global downscaling during Slow-Wave Sleep (SWS), as implemented:
     s_ij = |W_ij| / (max(|W|) + eps)
     W_ij <- W_ij * clip(1 - gamma_shy * (1 - 0.5 * s_ij), 0.70, 1.0)
   - Every synapse shrinks; weak synapses shrink by up to gamma, the strongest by gamma/2,
     so relative signal-to-noise of strong traces increases while sign is preserved.
2. 10x Accelerated Sharp-Wave Ripple (SWR) Replay:
   - Replays stored high-surprise episodic transitions at 10x temporal compression through connectome recurrent loops.
3. Counterfactual Dream Synthesis:
   - Explores hypothetical action alternatives around high-salience branch points, evaluating potential trajectories
     via causal world models and generating synthetic training tuples without physical risk.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class ReplayEpisode:
    episode_id: str
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    surprise_scores: List[float]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "length": len(self.observations),
            "total_reward": sum(self.rewards),
            "mean_surprise": float(np.mean(self.surprise_scores)) if self.surprise_scores else 0.0,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass
class ConsolidationReceipt:
    receipt_id: str
    timestamp: float
    sleep_mode: str  # 'sws_downscaling' | 'swr_replay' | 'counterfactual_dream'
    pre_synaptic_norm: float
    post_synaptic_norm: float
    percent_downscaled: float
    replayed_transitions: int
    synthetic_dreams_generated: int
    evidence: EvidenceReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "sleep_mode": self.sleep_mode,
            "pre_synaptic_norm": round(float(self.pre_synaptic_norm), 4),
            "post_synaptic_norm": round(float(self.post_synaptic_norm), 4),
            "percent_downscaled": round(float(self.percent_downscaled), 3),
            "replayed_transitions": self.replayed_transitions,
            "synthetic_dreams_generated": self.synthetic_dreams_generated,
            "evidence": self.evidence.to_dict(),
        }


class MemoryConsolidationAndSHYEngine:
    """Manages Slow-Wave Sleep SHY downscaling, SWR accelerated replay, and counterfactual dreaming."""

    def __init__(
        self,
        gamma_shy: float = 0.08,
        swr_compression_factor: float = 10.0,
        dream_branching_factor: int = 3,
        seed: int = 42,
    ):
        self.gamma_shy = float(gamma_shy)
        self.swr_compression = float(swr_compression_factor)
        self.branching = int(dream_branching_factor)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.episode_buffer: List[ReplayEpisode] = []
        self.consolidation_history: List[ConsolidationReceipt] = []
        self._receipt_counter = 0
        # Empirical reward model fitted from replayed transitions (None until fitted).
        self.reward_model: Optional[Dict[str, Any]] = None
        self.ridge_lambda = 1.0

    def record_episode(
        self,
        obs_seq: List[np.ndarray],
        act_seq: List[np.ndarray],
        rew_seq: List[float],
        surprise_seq: Optional[List[float]] = None,
    ) -> str:
        """Stores a completed episodic run in the replay buffer for sleep consolidation."""
        ep_id = f"ep_{int(time.time()*1000)}_{len(self.episode_buffer)+1}"
        surprises = surprise_seq or [float(abs(r)) for r in rew_seq]
        ep = ReplayEpisode(
            episode_id=ep_id,
            observations=[np.asarray(o, dtype=np.float32) for o in obs_seq],
            actions=[np.asarray(a, dtype=np.float32) for a in act_seq],
            rewards=[float(r) for r in rew_seq],
            surprise_scores=surprises,
            timestamp=time.time(),
        )
        self.episode_buffer.append(ep)
        if len(self.episode_buffer) > 64:
            self.episode_buffer.pop(0)
        return ep_id

    def execute_shy_synaptic_downscaling(
        self,
        weights: torch.Tensor,
        scaling_rate: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Applies SHY-style downscaling: W_ij <- W_ij * clip(1 - gamma*(1 - 0.5*|W_ij|/max|W|), 0.70, 1.0)."""
        gamma = float(scaling_rate if scaling_rate is not None else self.gamma_shy)
        pre_norm = float(torch.norm(weights).item())

        with torch.no_grad():
            max_val = float(torch.max(torch.abs(weights)).item()) + 1e-6
            relative_strength = torch.abs(weights) / max_val
            # Proportional downscaling preserves sign and relative order while compressing high weights
            downscale_factor = torch.clamp(1.0 - gamma * (1.0 - 0.5 * relative_strength), 0.70, 1.0)
            scaled_weights = weights * downscale_factor

        post_norm = float(torch.norm(scaled_weights).item())
        pct_change = float((pre_norm - post_norm) / max(1e-6, pre_norm)) * 100.0
        return scaled_weights, pre_norm, post_norm

    def execute_swr_replay(
        self,
        connectome_step_fn: Optional[Any] = None,
        max_episodes: int = 2,
    ) -> int:
        """Executes 10x-accelerated Sharp-Wave Ripple playback through the connectome."""
        if not self.episode_buffer:
            return 0

        # Sort episodes by total surprise salience
        sorted_eps = sorted(self.episode_buffer, key=lambda e: np.mean(e.surprise_scores) if e.surprise_scores else 0.0, reverse=True)
        replayed_steps = 0

        for ep in sorted_eps[:max_episodes]:
            # Sample every 2nd or 3rd step to simulate 10x temporal acceleration
            stride = max(1, int(len(ep.observations) // min(8, len(ep.observations))))
            for idx in range(0, len(ep.observations), stride):
                replayed_steps += 1
                if connectome_step_fn is not None:
                    try:
                        connectome_step_fn(ep.observations[idx], ep.actions[idx])
                    except Exception:
                        pass
        return replayed_steps

    def synthesize_counterfactual_dreams(
        self,
        world_model_fn: Optional[Any] = None,
        num_dreams: int = 4,
    ) -> List[Dict[str, Any]]:
        """Generates stochastic counterfactual trajectories around high-surprise states."""
        dreams = []
        if not self.episode_buffer:
            return dreams

        ep = self.episode_buffer[-1]
        if not ep.observations:
            return dreams

        # Select bifurcation state (highest surprise point)
        best_idx = int(np.argmax(ep.surprise_scores)) if ep.surprise_scores else 0
        pivot_obs = ep.observations[best_idx]
        baseline_act = ep.actions[best_idx]

        if world_model_fn is None and self.reward_model is None:
            self.fit_empirical_reward_model()

        for d in range(num_dreams):
            # Stochastic action perturbation
            noise = self.rng.normal(0.0, 0.35, size=baseline_act.shape).astype(np.float32)
            alt_act = np.clip(baseline_act + noise, -1.0, 1.0)

            # Counterfactual outcome. Rewards are only reported when a model
            # actually predicts them; otherwise they are None (never invented).
            pred_rew: Optional[float] = None
            if world_model_fn is not None:
                try:
                    _pred_obs, wm_rew = world_model_fn(pivot_obs, alt_act)
                    pred_rew, source = float(wm_rew), "world_model"
                except Exception:
                    source = "world_model_error"
            else:
                pred_rew = self.predict_empirical_reward(pivot_obs, alt_act)
                source = "empirical_ridge" if pred_rew is not None else "unavailable"

            dreams.append({
                "dream_id": f"dream_{int(time.time()*1000)}_{d+1}",
                "pivot_step": best_idx,
                "perturbed_action": alt_act.tolist(),
                "simulated_reward": pred_rew,
                "reward_source": source,
                "novelty": float(np.linalg.norm(noise)),
            })

        return dreams

    def run_full_consolidation_cycle(
        self,
        weights: torch.Tensor,
        connectome_step_fn: Optional[Any] = None,
        world_model_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, ConsolidationReceipt]:
        """Runs complete offline sleep consolidation: SHY downscaling + SWR replay + counterfactual dreams."""
        self._receipt_counter += 1
        rcpt_id = f"cons_rcpt_{int(time.time()*1000)}_{self._receipt_counter}"

        # 1. SHY Downscaling
        scaled_w, pre_n, post_n = self.execute_shy_synaptic_downscaling(weights)
        pct_down = float((pre_n - post_n) / max(1e-6, pre_n)) * 100.0

        # 2. SWR 10x Replay
        replayed = self.execute_swr_replay(connectome_step_fn=connectome_step_fn, max_episodes=2)

        # 3. Counterfactual Dreams
        dreams = self.synthesize_counterfactual_dreams(world_model_fn=world_model_fn, num_dreams=4)

        receipt = ConsolidationReceipt(
            receipt_id=rcpt_id,
            timestamp=time.time(),
            sleep_mode="sws_swr_dream_consolidated",
            pre_synaptic_norm=pre_n,
            post_synaptic_norm=post_n,
            percent_downscaled=pct_down,
            replayed_transitions=replayed,
            synthetic_dreams_generated=len(dreams),
            evidence=EvidenceReceipt(
                item_id=rcpt_id,
                kind=EvidenceKind.CONFIRMATORY_RESULT,
                source="MemoryConsolidationAndSHYEngine",
                metrics={
                    "pre_synaptic_norm": pre_n,
                    "post_synaptic_norm": post_n,
                    "percent_downscaled": pct_down,
                    "replayed_transitions": replayed,
                    "synthetic_dreams": len(dreams),
                },
            ),
        )
        self.consolidation_history.append(receipt)
        return scaled_w, receipt

    def get_summary(self) -> Dict[str, Any]:
        """Returns engine status summary."""
        return {
            "buffered_episodes": len(self.episode_buffer),
            "completed_consolidation_cycles": len(self.consolidation_history),
            "gamma_shy": self.gamma_shy,
            "swr_compression": self.swr_compression,
            "last_receipt": self.consolidation_history[-1].to_dict() if self.consolidation_history else None,
            "reward_model": None if self.reward_model is None else {
                k: v for k, v in self.reward_model.items() if k != "weights"
            },
        }

    # ------------------------------------------------------------------
    # Empirical reward model (ridge regression on recorded transitions)
    # ------------------------------------------------------------------
    def _transition_matrix(self) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        """Stacks buffered (obs, act) -> reward pairs that share the latest episode's shapes."""
        if not self.episode_buffer or not self.episode_buffer[-1].observations:
            return None
        ref = self.episode_buffer[-1]
        d_o, d_a = ref.observations[0].size, ref.actions[0].size
        rows, ys = [], []
        for ep in self.episode_buffer:
            for o, a, r in zip(ep.observations, ep.actions, ep.rewards):
                if o.size == d_o and a.size == d_a:
                    rows.append(np.concatenate([o.ravel(), a.ravel(), [1.0]]))
                    ys.append(r)
        if not rows:
            return None
        return np.asarray(rows, dtype=np.float64), np.asarray(ys, dtype=np.float64), d_o, d_a

    def fit_empirical_reward_model(self, min_transitions: int = 8) -> Optional[Dict[str, Any]]:
        """Fits r ~ [obs, act, 1] by ridge regression and reports a held-out R^2.

        Returns None (and leaves the model unset) when there is too little data.
        The held-out score uses the chronologically last 25% of transitions so
        callers can judge whether dream rewards are worth anything.
        """
        data = self._transition_matrix()
        if data is None or len(data[1]) < min_transitions:
            self.reward_model = None
            return None
        X, y, d_o, d_a = data
        n = len(y)
        n_tr = max(min_transitions // 2, int(round(n * 0.75)))
        lam = self.ridge_lambda

        def solve(Xs, ys):
            return np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ ys)

        holdout_r2: Optional[float] = None
        if n - n_tr >= 2:
            w_tr = solve(X[:n_tr], y[:n_tr])
            y_te = y[n_tr:]
            ss_res = float(np.sum((y_te - X[n_tr:] @ w_tr) ** 2))
            ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
            holdout_r2 = None if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        w = solve(X, y)  # final model uses all data
        self.reward_model = {
            "weights": w,
            "obs_dim": int(d_o),
            "act_dim": int(d_a),
            "n_transitions": int(n),
            "holdout_r2": None if holdout_r2 is None else round(holdout_r2, 4),
            "ridge_lambda": lam,
            "evidence_kind": EvidenceKind.EXPLORATORY_RESULT.value,
        }
        return self.reward_model

    def predict_empirical_reward(self, obs: np.ndarray, act: np.ndarray) -> Optional[float]:
        """Predicts reward with the fitted ridge model, or None if unavailable/incompatible."""
        m = self.reward_model
        if m is None:
            return None
        o, a = np.asarray(obs).ravel(), np.asarray(act).ravel()
        if o.size != m["obs_dim"] or a.size != m["act_dim"]:
            return None
        return float(np.concatenate([o, a, [1.0]]) @ m["weights"])


class LiveEpisodeRecorder:
    """Passive runtime tap that turns live transitions into replay episodes.

    Signature matches ``LabRuntime.transition_taps``:
    ``tap(prev_obs, action, reward, next_obs, done)``. Surprise is the absolute
    reward prediction error against an exponential moving average of reward
    (a simple, transparent salience proxy, not a learned model).
    """

    def __init__(self, engine: MemoryConsolidationAndSHYEngine, max_len: int = 64, ema_rate: float = 0.05):
        self.engine = engine
        self.max_len = int(max_len)
        self.ema_rate = float(ema_rate)
        self._ema: Optional[float] = None
        self._obs: List[np.ndarray] = []
        self._act: List[np.ndarray] = []
        self._rew: List[float] = []
        self._surp: List[float] = []
        self.transitions_seen = 0
        self.episodes_flushed = 0

    def __call__(self, prev_obs, action, reward, next_obs, done) -> None:
        r = float(reward)
        baseline = r if self._ema is None else self._ema
        self._surp.append(abs(r - baseline))
        self._ema = r if self._ema is None else (1 - self.ema_rate) * self._ema + self.ema_rate * r
        self._obs.append(np.asarray(prev_obs, dtype=np.float32).copy())
        self._act.append(np.asarray(action, dtype=np.float32).copy())
        self._rew.append(r)
        self.transitions_seen += 1
        if done or len(self._obs) >= self.max_len:
            self.flush()

    def flush(self) -> Optional[str]:
        if not self._obs:
            return None
        ep_id = self.engine.record_episode(self._obs, self._act, self._rew, self._surp)
        self._obs, self._act, self._rew, self._surp = [], [], [], []
        self.episodes_flushed += 1
        return ep_id

    def status(self) -> Dict[str, Any]:
        return {
            "transitions_seen": self.transitions_seen,
            "episodes_flushed": self.episodes_flushed,
            "pending_transitions": len(self._obs),
            "surprise_proxy": "abs(reward - EMA(reward))",
        }
