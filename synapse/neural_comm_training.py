"""SANCF Explicit, Leakage-Free Differentiable Training & Multi-Objective Optimization.

Trains ``NeuralCommFabric`` (encoder, router, decoder, VQ/orthogonal parameters, and
warm-started residual gate ``gamma``) using executed ``(obs, action, reward, next_obs, done)``
trajectories without leaking future state into online inference and without mutating
frozen donor connectome edges or frozen pretrained Tidal core weights.

Multi-objective loss:
    L = L_task + lambda_pred * L_prediction + lambda_bits * R_bits
        + lambda_energy * R_compute + lambda_interf * R_interference + lambda_stab * R_stability
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_comm import NeuralCommFabric


@dataclass
class CommTrainingConfig:
    lr: float = 3e-3
    lambda_pred: float = 0.50
    lambda_bits: float = 0.05
    lambda_energy: float = 0.02
    lambda_interf: float = 0.10
    lambda_stab: float = 0.10
    gamma_warmup_target: float = 0.08
    gamma_warmup_rate: float = 0.20
    max_grad_norm: float = 1.0


class NeuralCommTrainer:
    """Leakage-free replay trainer and candidate validator for ``NeuralCommFabric``."""

    def __init__(self, fabric: NeuralCommFabric, config: Optional[CommTrainingConfig] = None, obs_dim: int = 155, act_dim: int = 5):
        self.fabric = fabric
        self.config = config or CommTrainingConfig()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Auxiliary heads mapping target-region post-communication state to next-obs summary & action value
        self.next_obs_head = nn.Linear(32, 16)
        self.action_value_head = nn.Linear(32 + self.act_dim, 1)
        nn.init.orthogonal_(self.next_obs_head.weight, gain=0.3)
        nn.init.zeros_(self.next_obs_head.bias)
        nn.init.orthogonal_(self.action_value_head.weight, gain=0.3)
        nn.init.zeros_(self.action_value_head.bias)

        params = (
            list(self.fabric.parameters())
            + list(self.next_obs_head.parameters())
            + list(self.action_value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
        self.replay_buffer: List[Dict[str, Any]] = []
        self.max_replay: int = 256
        self.history: List[Dict[str, Any]] = []

    def record_executed_transition(
        self,
        obs: np.ndarray,
        rates: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> None:
        """Record a real executed transition (no future oracle state)."""
        entry = {
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "rates": np.asarray(rates, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "done": bool(done),
        }
        self.replay_buffer.append(entry)
        if len(self.replay_buffer) > self.max_replay:
            self.replay_buffer = self.replay_buffer[-self.max_replay :]

    def train_step_on_batch(self, transitions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run one differentiable multi-objective update over executed transitions."""
        batch = transitions if transitions is not None else self.replay_buffer[-16:]
        if not batch:
            return {"updated": False, "reason": "empty_replay"}

        was_enabled = self.fabric.enabled
        self.fabric.enabled = True
        # Warm residual gate gamma slightly off zero so gradients flow cleanly into encoder/decoder/router
        if abs(float(self.fabric.residual_gain.detach().item())) < 1e-4:
            self.fabric.warm_gate(target_gamma=self.config.gamma_warmup_target, rate=self.config.gamma_warmup_rate)

        self.optimizer.zero_grad()

        loss_task_total = torch.tensor(0.0, dtype=torch.float32)
        loss_pred_total = torch.tensor(0.0, dtype=torch.float32)
        r_bits_total = torch.tensor(0.0, dtype=torch.float32)
        r_comp_total = torch.tensor(0.0, dtype=torch.float32)
        r_interf_total = torch.tensor(0.0, dtype=torch.float32)
        r_stab_total = torch.tensor(0.0, dtype=torch.float32)

        for idx, item in enumerate(batch):
            r_in = torch.as_tensor(item["rates"], dtype=torch.float32)
            act_t = torch.as_tensor(item["action"], dtype=torch.float32)
            rew_t = torch.tensor(float(item["reward"]), dtype=torch.float32)
            next_obs_t = torch.as_tensor(item["next_obs"][:16], dtype=torch.float32)

            self.fabric.begin_environment_step(step_id=idx, read_only=False)
            comm_out = self.fabric.compute_current(r_in, microstep=0, advance=True, adapt=True)
            aux_cur = comm_out.residual_current

            # Differentiable post-communication regional representation
            r_post = torch.clamp(r_in + aux_cur, 0.0, 1.0)
            pooled_32 = F.adaptive_avg_pool1d(r_post.reshape(1, 1, -1), 32).reshape(32)

            # 1. Task value / policy-aligned prediction loss
            pred_q = self.action_value_head(torch.cat([pooled_32, act_t], dim=0)).squeeze(0)
            loss_task_total = loss_task_total + F.smooth_l1_loss(pred_q, rew_t)

            # 2. Held-out next-observation channel prediction loss (no future leakage at inference)
            pred_next = self.next_obs_head(pooled_32)
            loss_pred_total = loss_pred_total + F.mse_loss(pred_next, next_obs_t)

            # 3. Bitrate & gate regularization R_bits
            bit_frac = float(comm_out.payload_bits) / float(max(1, self.fabric.spec.max_payload_bits_per_tick))
            gate_mean = torch.mean(torch.abs(aux_cur)) / max(1e-4, self.fabric.max_aux_current)
            r_bits_total = r_bits_total + gate_mean * (0.5 + bit_frac)

            # 4. Compute cost regularization R_compute
            r_comp_total = r_comp_total + torch.tensor(float(comm_out.compute_ops) / 10000.0, dtype=torch.float32)

            # 5. Cross-route interference penalty R_interference (penalizes excessive overlap on target currents)
            r_interf_total = r_interf_total + torch.mean(aux_cur ** 4)

            # 6. Dynamical stability penalty R_stability
            r_stab_total = r_stab_total + F.relu(torch.max(torch.abs(aux_cur)) - self.fabric.max_aux_current) + 0.1 * (self.fabric.residual_gain ** 2)

        n_b = float(len(batch))
        loss_task = loss_task_total / n_b
        loss_pred = loss_pred_total / n_b
        r_bits = r_bits_total / n_b
        r_comp = r_comp_total / n_b
        r_interf = r_interf_total / n_b
        r_stab = r_stab_total / n_b

        total_loss = (
            loss_task
            + self.config.lambda_pred * loss_pred
            + self.config.lambda_bits * r_bits
            + self.config.lambda_energy * r_comp
            + self.config.lambda_interf * r_interf
            + self.config.lambda_stab * r_stab
        )

        total_loss.backward()

        # Verify finite gradients across encoder, router, and decoder
        grad_norms: Dict[str, float] = {}
        for name, module in (
            ("encoder", self.fabric.encoder),
            ("decoder", self.fabric.decoder),
            ("router", self.fabric.router),
        ):
            sq_sum = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                    sq_sum += float((p.grad.detach() ** 2).sum().item())
            grad_norms[name] = round(math.sqrt(sq_sum), 6)

        nn.utils.clip_grad_norm_(self.fabric.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            self.fabric.residual_gain.clamp_(-self.fabric.max_aux_current, self.fabric.max_aux_current)
        self.fabric.training_steps += 1
        self.fabric.enabled = was_enabled

        receipt = {
            "updated": True,
            "training_steps": int(self.fabric.training_steps),
            "batch_size": len(batch),
            "total_loss": round(float(total_loss.detach().item()), 6),
            "loss_task": round(float(loss_task.detach().item()), 6),
            "loss_prediction": round(float(loss_pred.detach().item()), 6),
            "r_bits": round(float(r_bits.detach().item()), 6),
            "r_compute": round(float(r_comp.detach().item()), 6),
            "r_interference": round(float(r_interf.detach().item()), 6),
            "r_stability": round(float(r_stab.detach().item()), 6),
            "residual_gain_gamma": round(float(self.fabric.residual_gain.detach().item()), 6),
            "grad_norms": grad_norms,
        }
        self.history.append(receipt)
        if len(self.history) > 64:
            self.history = self.history[-64:]
        return receipt

    def train_with_candidate_rollback(
        self,
        transitions: List[Dict[str, Any]],
        held_out_transitions: List[Dict[str, Any]],
        epochs: int = 3,
    ) -> Dict[str, Any]:
        """Train on support transitions in isolation and rollback atomically if held-out loss degrades."""
        saved_state = self.fabric.export_state()
        saved_opt = deepcopy(self.optimizer.state_dict())

        def eval_held_out() -> float:
            if not held_out_transitions:
                return 0.0
            was_en = self.fabric.enabled
            self.fabric.enabled = True
            losses = []
            with torch.no_grad():
                for idx, item in enumerate(held_out_transitions):
                    r_in = torch.as_tensor(item["rates"], dtype=torch.float32)
                    next_obs_t = torch.as_tensor(item["next_obs"][:16], dtype=torch.float32)
                    comm_out = self.fabric.compute_current(r_in, microstep=0, advance=False, adapt=False)
                    r_post = torch.clamp(r_in + comm_out.residual_current, 0.0, 1.0)
                    pooled_32 = F.adaptive_avg_pool1d(r_post.reshape(1, 1, -1), 32).reshape(32)
                    pred_next = self.next_obs_head(pooled_32)
                    losses.append(float(F.mse_loss(pred_next, next_obs_t).item()))
            self.fabric.enabled = was_en
            return float(np.mean(losses))

        baseline_val_loss = eval_held_out()
        step_receipts = []
        try:
            for _ in range(max(1, int(epochs))):
                step_receipts.append(self.train_step_on_batch(transitions))
            candidate_val_loss = eval_held_out()
            if not math.isfinite(candidate_val_loss) or candidate_val_loss > baseline_val_loss * 1.15 + 0.05:
                self.fabric.import_state(saved_state)
                self.optimizer.load_state_dict(saved_opt)
                return {
                    "promoted": False,
                    "rolled_back": True,
                    "reason": "held_out_validation_degraded",
                    "baseline_val_loss": round(baseline_val_loss, 6),
                    "candidate_val_loss": round(candidate_val_loss, 6),
                }
            return {
                "promoted": True,
                "rolled_back": False,
                "baseline_val_loss": round(baseline_val_loss, 6),
                "candidate_val_loss": round(candidate_val_loss, 6),
                "epochs": len(step_receipts),
                "last_step": step_receipts[-1] if step_receipts else {},
            }
        except Exception as exc:
            self.fabric.import_state(saved_state)
            self.optimizer.load_state_dict(saved_opt)
            return {"promoted": False, "rolled_back": True, "reason": f"exception:{exc}"}
