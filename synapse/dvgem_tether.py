"""Poseidon DV-GEM (Decision-Value-Gated Episodic Memory), Project TETHER
Specialist Reliability Gate, TD Successor Features, and Poincaré/Lyapunov Monitor.

Synthesizes four flagship architectural mechanisms from the kai9987kai ecosystem
(Supermix-Poseidon, Project TETHER, Poincaré Synthesis, and Tidepool v2 DV-GEM):

1. **DV-GEM (`DecisionValueEpisodicMemory`)**:
   - Recurrent GRU belief encoder h_t = GRU(h_{t-1}, [o_t, a_{t-1}, r_{t-1}]) for
     partially observable and observation-aliased environments.
   - Counterfactual episodic retrieval with Support Screening, Conflict Screening,
     Predictive Viability Reserve Guardrail, and an Explicit Admission Gate:
       Admitted <=> supported and (conflict <= theta_c)
                    and (Delta_Q - lambda * sigma_Q > theta_admit)
                    and (viability_reserve >= reserve_floor)
   - Supports the 6 matched lesion arms (`intact`, `similarity`, `erased`,
     `shifted`, `donor`, `baseline`) for controlled causal ablation studies.

2. **Project TETHER (`TetherSpecialistCoordinator`)**:
   - Evaluates continuous action proposals from multiple active specialists
     (`base_actor`, `neural_graft`, `experience_mpc`, `recurrent_latent`, `dvgem`).
   - Computes consensus centroid deviation, directional cosine coherence,
     physical bound violations, zero-shot specialist corruption flags, and
     evidence-calibrated fallback gate tau in [0, 1].

3. **TD Successor Features (`SuccessorFeatureEngine`)**:
   - Learns discounted feature expectations psi(s, a) = E[sum_{t>=0} gamma^t phi(s_t)]
     decoupled from task reward weights w_r (Q(s, a) = psi(s, a)^T w_r), enabling
     rapid transfer across changing reward regimes.

4. **Poincaré Hyperbolic & Lyapunov Stability Monitor (`PoincareLyapunovMonitor`)**:
   - Embeds neuropil population activity into the Poincaré ball B^d, computing
     hyperbolic hierarchy depth and inter-neuropil geodesic distances d_H(u, v),
     alongside local finite-time Lyapunov exponent (FTLE) lambda_max estimation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DVGEM_MODES = ("intact", "similarity", "erased", "shifted", "donor", "baseline")


@dataclass
class DVGEMMemoryRecord:
    step: int
    belief: np.ndarray        # [belief_dim]
    obs_summary: np.ndarray   # [16]
    action: np.ndarray        # [5]
    reward: float
    next_obs_summary: np.ndarray  # [16]
    viability_reserve: float


class DecisionValueEpisodicMemory(nn.Module):
    """Poseidon DV-GEM with recurrent belief encoding, conflict screening, and viability guard."""

    def __init__(
        self,
        obs_dim: int = 155,
        action_dim: int = 5,
        belief_dim: int = 24,
        capacity: int = 256,
        top_k: int = 4,
        admission_threshold: float = 0.02,
        conflict_threshold: float = 0.45,
        max_distance: float = 1.6,
        reserve_floor: float = 0.22,
        seed: int = 42,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.belief_dim = int(belief_dim)
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.admission_threshold = float(admission_threshold)
        self.conflict_threshold = float(conflict_threshold)
        self.max_distance = float(max_distance)
        self.reserve_floor = float(reserve_floor)
        self.seed = int(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 5101)
            self.obs_proj = nn.Linear(obs_dim + action_dim + 1, belief_dim)
            self.belief_gru = nn.GRUCell(belief_dim, belief_dim)
            self.q_head = nn.Sequential(
                nn.Linear(belief_dim + action_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            # Zero-init final layer so untrained Q-head starts neutral
            nn.init.zeros_(self.q_head[-1].weight)
            nn.init.zeros_(self.q_head[-1].bias)

        self.register_buffer("belief_state", torch.zeros(belief_dim, dtype=torch.float32))
        self. register_buffer("prev_action", torch.zeros(action_dim, dtype=torch.float32))
        self.register_buffer("prev_reward", torch.zeros(1, dtype=torch.float32))

        self.entries: List[DVGEMMemoryRecord] = []
        self.step_count = 0
        self.admitted_count = 0
        self.rejected_conflict_count = 0
        self.rejected_reserve_count = 0
        self.last_diagnostics: Dict[str, Any] = {}

    def reset_state(self) -> None:
        with torch.no_grad():
            self.belief_state.zero_()
            self.prev_action.zero_()
            self.prev_reward.zero_()

    def update_belief(self, obs: np.ndarray) -> np.ndarray:
        """Advance recurrent belief state h_t = GRU(h_{t-1}, proj([o_t, a_{t-1}, r_{t-1}]))."""
        with torch.no_grad():
            o_t = torch.as_tensor(obs, dtype=torch.float32).flatten()[: self.obs_dim]
            inp = torch.cat([o_t, self.prev_action, self.prev_reward], dim=0)
            x_emb = torch.tanh(self.obs_proj(inp))
            h_next = self.belief_gru(x_emb.unsqueeze(0), self.belief_state.unsqueeze(0)).squeeze(0)
            self.belief_state.copy_(h_next)
            return h_next.detach().cpu().numpy().copy()

    def query_and_gate(
        self,
        obs: np.ndarray,
        baseline_action: np.ndarray,
        viability_reserve: float = 1.0,
        mode: str = "intact",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Query episodic belief store and apply the explicit DV-GEM admission & reserve gate."""
        if mode not in DVGEM_MODES:
            raise ValueError(f"mode must be one of {DVGEM_MODES}")

        self.step_count += 1
        belief = self.update_belief(obs)
        base_act = np.asarray(baseline_action, dtype=np.float32).copy()

        if mode in ("baseline", "erased") or len(self.entries) < 2:
            diag = {
                "enabled": True,
                "mode": mode,
                "admitted": False,
                "reason": "insufficient_entries" if len(self.entries) < 2 else f"mode_{mode}",
                "entries": len(self.entries),
                "delta_q": 0.0,
                "conflict": 0.0,
                "min_distance": 0.0,
                "viability_reserve": round(float(viability_reserve), 4),
                "reserve_guard_triggered": False,
                "admitted_total": self.admitted_count,
            }
            self.last_diagnostics = diag
            return base_act, diag

        # Viability reserve guard (Poseidon Tidepool v2 predictive reserve floor)
        if viability_reserve < self.reserve_floor:
            self.rejected_reserve_count += 1
            diag = {
                "enabled": True,
                "mode": mode,
                "admitted": False,
                "reason": "viability_reserve_guard",
                "entries": len(self.entries),
                "delta_q": 0.0,
                "conflict": 0.0,
                "min_distance": 0.0,
                "viability_reserve": round(float(viability_reserve), 4),
                "reserve_guard_triggered": True,
                "admitted_total": self.admitted_count,
            }
            self.last_diagnostics = diag
            return base_act, diag

        obs_summary = np.asarray(obs[:16], dtype=np.float32)
        distances = []
        for idx, rec in enumerate(self.entries):
            if mode == "similarity":
                d = float(np.linalg.norm(obs_summary - rec.obs_summary))
            elif mode == "shifted":
                shifted_b = np.roll(rec.belief, 5)
                d = float(np.linalg.norm(belief - shifted_b))
            elif mode == "donor":
                donor_b = -rec.belief
                d = float(np.linalg.norm(belief - donor_b))
            else:
                d = float(np.linalg.norm(belief - rec.belief))
            distances.append((d, idx))

        distances.sort(key=lambda pair: pair[0])
        top = distances[: min(self.top_k, len(distances))]
        min_dist = float(top[0][0])
        supported = min_dist <= self.max_distance

        weights = np.array([math.exp(-d * 2.0) for d, _ in top], dtype=np.float32)
        weights = weights / (float(weights.sum()) + 1e-8)
        rewards = np.array([self.entries[idx].reward for _, idx in top], dtype=np.float32)
        actions = np.stack([self.entries[idx].action for _, idx in top], axis=0)

        mean_r = float(np.sum(weights * rewards))
        conflict = float(np.sqrt(np.sum(weights * (rewards - mean_r) ** 2)))

        with torch.no_grad():
            b_t = torch.as_tensor(belief, dtype=torch.float32)
            a_t = torch.as_tensor(base_act, dtype=torch.float32)
            q_base = float(self.q_head(torch.cat([b_t, a_t], dim=0)).item())

        delta_q = float(mean_r - q_base)
        effective_advantage = delta_q - 0.5 * conflict

        if mode == "similarity":
            # Similarity lesion bypasses conflict and decision-value threshold
            admitted = supported and (mean_r > 0.0)
        else:
            admitted = (
                supported
                and (conflict <= self.conflict_threshold)
                and (effective_advantage > self.admission_threshold)
            )

        if not admitted and conflict > self.conflict_threshold:
            self.rejected_conflict_count += 1

        if admitted:
            self.admitted_count += 1
            mem_act = np.sum(weights[:, None] * actions, axis=0).astype(np.float32)
            alpha = float(np.clip(0.35 * math.tanh(max(0.0, effective_advantage) * 2.5), 0.05, 0.35))
            gated_action = np.clip((1.0 - alpha) * base_act + alpha * mem_act, -1.0, 1.0).astype(np.float32)
            reason = "admitted_decision_value"
        else:
            gated_action = base_act
            reason = (
                "out_of_support"
                if not supported
                else ("high_conflict" if conflict > self.conflict_threshold else "below_value_threshold")
            )

        diag = {
            "enabled": True,
            "mode": mode,
            "admitted": bool(admitted),
            "reason": reason,
            "entries": len(self.entries),
            "delta_q": round(delta_q, 4),
            "effective_advantage": round(effective_advantage, 4),
            "conflict": round(conflict, 4),
            "min_distance": round(min_dist, 4),
            "viability_reserve": round(float(viability_reserve), 4),
            "reserve_guard_triggered": False,
            "admitted_total": self.admitted_count,
        }
        self.last_diagnostics = diag
        return gated_action, diag

    def record_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        viability_reserve: float = 1.0,
    ) -> Dict[str, Any]:
        """Store realized transition in episodic memory and update local Q-baseline."""
        act = np.asarray(action, dtype=np.float32).copy()
        rew = float(reward)
        with torch.no_grad():
            self.prev_action.copy_(torch.as_tensor(act, dtype=torch.float32))
            self.prev_reward.fill_(rew)
            # Local delta rule on q_head output layer to track baseline return scale
            b_t = self.belief_state.detach()
            a_t = torch.as_tensor(act, dtype=torch.float32)
            h_mid = F.silu(self.q_head[0](torch.cat([b_t, a_t], dim=0)))
            pred_q = float(self.q_head[2](h_mid).item())
            td_err = float(np.clip(rew - pred_q, -2.0, 2.0))
            self.q_head[2].weight.add_(0.015 * td_err * h_mid.unsqueeze(0))
            self.q_head[2].bias.add_(0.015 * td_err)

        rec = DVGEMMemoryRecord(
            step=self.step_count,
            belief=self.belief_state.detach().cpu().numpy().copy(),
            obs_summary=np.asarray(obs[:16], dtype=np.float32).copy(),
            action=act,
            reward=rew,
            next_obs_summary=np.asarray(next_obs[:16], dtype=np.float32).copy(),
            viability_reserve=float(viability_reserve),
        )
        self.entries.append(rec)
        if len(self.entries) > self.capacity:
            # Retain high-absolute-reward or recent memories
            drop_idx = int(np.argmin([abs(e.reward) + 0.002 * i for i, e in enumerate(self.entries[:-8])]))
            self.entries.pop(drop_idx)

        return {
            "entries": len(self.entries),
            "td_error": round(td_err, 4),
            "admitted_total": self.admitted_count,
        }

    def run_six_arm_lesion_assay(self, obs: np.ndarray, baseline_action: np.ndarray, viability_reserve: float = 0.85) -> Dict[str, Any]:
        """Run all 6 Poseidon DV-GEM matched lesion arms on the current state without mutating live belief."""
        saved_belief = self.belief_state.clone()
        saved_step = self.step_count
        saved_admitted = self.admitted_count
        saved_conflict = self.rejected_conflict_count
        saved_reserve = self.rejected_reserve_count
        saved_diag = deepcopy(self.last_diagnostics)

        arms: Dict[str, Any] = {}
        try:
            for mode in DVGEM_MODES:
                self.belief_state.copy_(saved_belief)
                act, diag = self.query_and_gate(obs, baseline_action, viability_reserve=viability_reserve, mode=mode)
                arms[mode] = {
                    "action": [round(float(x), 4) for x in act],
                    "action_shift_l2": round(float(np.linalg.norm(act - baseline_action)), 4),
                    "admitted": bool(diag["admitted"]),
                    "delta_q": float(diag["delta_q"]),
                    "conflict": float(diag["conflict"]),
                    "reason": str(diag["reason"]),
                }
        finally:
            self.belief_state.copy_(saved_belief)
            self.step_count = saved_step
            self.admitted_count = saved_admitted
            self.rejected_conflict_count = saved_conflict
            self.rejected_reserve_count = saved_reserve
            self.last_diagnostics = saved_diag

        return {
            "schema": "poseidon-dvgem-6arm-lesion-v1",
            "entries_evaluated": len(self.entries),
            "arms": arms,
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "weights": deepcopy(self.state_dict()),
            "step_count": int(self.step_count),
            "admitted_count": int(self.admitted_count),
            "rejected_conflict_count": int(self.rejected_conflict_count),
            "rejected_reserve_count": int(self.rejected_reserve_count),
            "last_diagnostics": deepcopy(self.last_diagnostics),
            "entries": [
                {
                    "step": int(e.step),
                    "belief": e.belief.copy(),
                    "obs_summary": e.obs_summary.copy(),
                    "action": e.action.copy(),
                    "reward": float(e.reward),
                    "next_obs_summary": e.next_obs_summary.copy(),
                    "viability_reserve": float(e.viability_reserve),
                }
                for e in self.entries
            ],
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "weights" in state:
            self.load_state_dict(deepcopy(state["weights"]))
        self.step_count = int(state.get("step_count", 0))
        self.admitted_count = int(state.get("admitted_count", 0))
        self.rejected_conflict_count = int(state.get("rejected_conflict_count", 0))
        self.rejected_reserve_count = int(state.get("rejected_reserve_count", 0))
        self.last_diagnostics = deepcopy(state.get("last_diagnostics", {}))
        self.entries = [
            DVGEMMemoryRecord(
                step=int(r["step"]),
                belief=np.asarray(r["belief"], dtype=np.float32).copy(),
                obs_summary=np.asarray(r["obs_summary"], dtype=np.float32).copy(),
                action=np.asarray(r["action"], dtype=np.float32).copy(),
                reward=float(r["reward"]),
                next_obs_summary=np.asarray(r["next_obs_summary"], dtype=np.float32).copy(),
                viability_reserve=float(r.get("viability_reserve", 1.0)),
            )
            for r in state.get("entries", [])
        ]


class TetherSuccessorCoordinator:
    """Project TETHER Specialist Reliability Gate + TD Successor Features + Poincaré/Lyapunov Monitor."""

    def __init__(self, feature_dim: int = 16, action_dim: int = 5, gamma: float = 0.92, seed: int = 42):
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        # Successor feature matrix W_psi: maps [phi(s), a] (21D) -> psi (16D)
        self.W_psi = np.zeros((self.feature_dim, self.feature_dim + self.action_dim), dtype=np.float32)
        self.W_psi[:, : self.feature_dim] = np.eye(self.feature_dim, dtype=np.float32)
        # Task reward weight vector w_r (16D): r(s) approx phi(s)^T w_r
        self.w_reward = np.zeros(self.feature_dim, dtype=np.float32)

        # Poincaré & Lyapunov tracking state
        self.prev_rates_norm: Optional[float] = None
        self.prev_delta_norm: float = 1e-3
        self.lyapunov_ema: float = -0.05
        self.steps: int = 0
        self.fallback_triggers: int = 0
        self.last_summary: Dict[str, Any] = {}

    def reset(self) -> None:
        self.prev_rates_norm = None
        self.prev_delta_norm = 1e-3

    def _encode_phi(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features[: self.feature_dim], dtype=np.float32)
        norm = float(np.linalg.norm(raw)) + 1e-6
        return (raw / max(1.0, norm)).astype(np.float32)

    def predict_successor_q(self, features: np.ndarray, action: np.ndarray) -> Tuple[float, np.ndarray]:
        phi = self._encode_phi(features)
        sa = np.concatenate([phi, np.asarray(action, dtype=np.float32)])
        psi = self.W_psi @ sa
        q_val = float(np.dot(psi, self.w_reward))
        return q_val, psi

    def update_successor_td(
        self,
        features: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_features: np.ndarray,
        next_action: np.ndarray,
        done: bool = False,
    ) -> Dict[str, float]:
        """TD(0) update for Successor Features psi(s, a) and reward weights w_r."""
        phi = self._encode_phi(features)
        phi_next = self._encode_phi(next_features)
        sa = np.concatenate([phi, np.asarray(action, dtype=np.float32)])
        sa_next = np.concatenate([phi_next, np.asarray(next_action, dtype=np.float32)])

        psi_curr = self.W_psi @ sa
        psi_next = np.zeros_like(psi_curr) if done else (self.W_psi @ sa_next)
        td_target = phi + (0.0 if done else self.gamma * psi_next)
        td_err = td_target - psi_curr
        self.W_psi += 0.02 * np.outer(td_err, sa)
        self.W_psi = np.clip(self.W_psi, -3.0, 3.0)

        # Reward weight update: phi^T w_r -> reward
        r_pred = float(np.dot(phi, self.w_reward))
        r_err = float(np.clip(float(reward) - r_pred, -2.0, 2.0))
        self.w_reward += 0.03 * r_err * phi
        self.w_reward = np.clip(self.w_reward, -4.0, 4.0)

        return {
            "successor_td_norm": round(float(np.linalg.norm(td_err)), 4),
            "reward_weight_norm": round(float(np.linalg.norm(self.w_reward)), 4),
            "predicted_q": round(float(np.dot(psi_curr, self.w_reward)), 4),
        }

    def evaluate_poincare_and_lyapunov(
        self,
        rates: np.ndarray,
        origins: np.ndarray,
    ) -> Dict[str, Any]:
        """Project neuropil centroids into Poincaré ball B^3 and estimate local FTLE lambda_max."""
        r = np.asarray(rates, dtype=np.float32).flatten()
        orig = np.asarray(origins, dtype=np.int64).flatten()

        # Compute 3D neuropil embedding vectors inside Poincaré ball (||u|| < 1)
        embeddings = []
        radii = []
        for dom in range(3):
            mask = orig == dom
            dom_r = r[mask] if np.any(mask) else r[:8]
            m1 = float(np.mean(dom_r))
            m2 = float(np.std(dom_r))
            m3 = float(np.mean(np.tanh(dom_r * 2.0)))
            vec = np.array([m1, m2, m3], dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            # Map into Poincaré ball radius in [0, 0.92]
            scaled = vec * (0.85 * math.tanh(norm) / (norm + 1e-8))
            embeddings.append(scaled)
            radii.append(float(np.linalg.norm(scaled)))

        # Hyperbolic geodesic distance between Neuropil 0 (FlyWire) and Neuropil 1 (MANC)
        u, v = embeddings[0], embeddings[1]
        uu = min(0.98, float(np.dot(u, u)))
        vv = min(0.98, float(np.dot(v, v)))
        diff_sq = float(np.sum((u - v) ** 2))
        arg = 1.0 + (2.0 * diff_sq) / max(1e-6, (1.0 - uu) * (1.0 - vv))
        d_hyp = float(math.acosh(max(1.0, arg)))

        # Finite-Time Lyapunov Exponent (FTLE) surrogate from recurrent trajectory divergence
        curr_norm = float(np.linalg.norm(r)) / math.sqrt(max(1, len(r)))
        if self.prev_rates_norm is not None:
            delta = abs(curr_norm - self.prev_rates_norm) + 1e-5
            local_lambda = float(np.clip(math.log(delta / max(1e-5, self.prev_delta_norm)), -1.5, 1.5))
            self.lyapunov_ema = 0.90 * self.lyapunov_ema + 0.10 * local_lambda
            self.prev_delta_norm = delta
        self.prev_rates_norm = curr_norm

        regime = (
            "chaotic"
            if self.lyapunov_ema > 0.08
            else ("edge_of_chaos" if self.lyapunov_ema > -0.12 else "attractor")
        )
        return {
            "poincare_mean_radius": round(float(np.mean(radii)), 4),
            "hyperbolic_distance_fw_manc": round(d_hyp, 4),
            "lyapunov_exponent": round(float(self.lyapunov_ema), 4),
            "dynamical_regime": regime,
        }

    def coordinate_specialists(
        self,
        proposals: Dict[str, np.ndarray],
        features: np.ndarray,
        rates: np.ndarray,
        origins: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Project TETHER continuous reliability weighting & fallback gate across specialist proposals."""
        self.steps += 1
        names = list(proposals.keys())
        mats = np.stack([np.asarray(proposals[k], dtype=np.float32) for k in names], axis=0)
        K = len(names)
        centroid = np.mean(mats, axis=0)
        c_norm = float(np.linalg.norm(centroid)) + 1e-8

        deviations = []
        cosines = []
        violations = []
        scores = []
        for k in range(K):
            vec = mats[k]
            dev = float(np.linalg.norm(vec - centroid))
            v_norm = float(np.linalg.norm(vec)) + 1e-8
            cos = float(np.dot(vec, centroid) / (v_norm * c_norm))
            viol = float(np.sum(np.maximum(0.0, np.abs(vec) - 1.0) ** 2))
            q_succ, _ = self.predict_successor_q(features, vec)
            score = -1.2 * dev + 0.5 * cos - 2.5 * viol + 0.25 * math.tanh(q_succ)
            deviations.append(dev)
            cosines.append(cos)
            violations.append(viol)
            scores.append(score)

        scores_arr = np.asarray(scores, dtype=np.float64)
        scores_arr -= np.max(scores_arr)
        weights = np.exp(scores_arr)
        weights = (weights / (np.sum(weights) + 1e-8)).astype(np.float32)

        # Epistemic trust tau in [0, 1] from specialist consensus
        mean_dev = float(np.mean(deviations))
        epistemic_trust = float(np.clip(math.exp(-1.1 * mean_dev), 0.0, 1.0))

        base_act = np.asarray(proposals.get("base_actor", mats[0]), dtype=np.float32)
        consensus_act = np.sum(weights[:, None] * mats, axis=0).astype(np.float32)

        fallback_triggered = epistemic_trust < 0.35
        if fallback_triggered:
            self.fallback_triggers += 1
            final_act = base_act.copy()
        else:
            # Blend consensus with base_actor proportional to epistemic trust
            tau = float(np.clip((epistemic_trust - 0.35) / 0.65, 0.0, 1.0))
            final_act = np.clip((1.0 - 0.5 * tau) * base_act + (0.5 * tau) * consensus_act, -1.0, 1.0).astype(np.float32)

        geom = self.evaluate_poincare_and_lyapunov(rates, origins)
        succ_q, _ = self.predict_successor_q(features, final_act)

        summary = {
            "enabled": True,
            "specialists": names,
            "trust_weights": {names[i]: round(float(weights[i]), 4) for i in range(K)},
            "epistemic_trust": round(epistemic_trust, 4),
            "mean_specialist_deviation": round(mean_dev, 4),
            "fallback_triggered": bool(fallback_triggered),
            "fallback_total": int(self.fallback_triggers),
            "successor_q_value": round(succ_q, 4),
            **geom,
        }
        self.last_summary = summary
        return final_act, summary

    def export_state(self) -> Dict[str, Any]:
        return {
            "W_psi": self.W_psi.copy(),
            "w_reward": self.w_reward.copy(),
            "prev_rates_norm": self.prev_rates_norm,
            "prev_delta_norm": float(self.prev_delta_norm),
            "lyapunov_ema": float(self.lyapunov_ema),
            "steps": int(self.steps),
            "fallback_triggers": int(self.fallback_triggers),
            "last_summary": deepcopy(self.last_summary),
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "W_psi" in state:
            self.W_psi = np.asarray(state["W_psi"], dtype=np.float32).copy()
        if "w_reward" in state:
            self.w_reward = np.asarray(state["w_reward"], dtype=np.float32).copy()
        self.prev_rates_norm = None if state.get("prev_rates_norm") is None else float(state["prev_rates_norm"])
        self.prev_delta_norm = float(state.get("prev_delta_norm", 1e-3))
        self.lyapunov_ema = float(state.get("lyapunov_ema", -0.05))
        self.steps = int(state.get("steps", 0))
        self.fallback_triggers = int(state.get("fallback_triggers", 0))
        self.last_summary = deepcopy(state.get("last_summary", {}))
        if "rng" in state:
            self.rng.bit_generator.state = deepcopy(state["rng"])
