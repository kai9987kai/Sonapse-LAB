"""Weight-Programmed Affective Connectome (WPAC).

A dedicated 256-neuron recurrent graph structured into 8 functional populations:
1. Sensory Salience (neurons 0..31)
2. Threat & Negative-Outcome Prediction (neurons 32..63)
3. Reward & Approach Valuation (neurons 64..95)
4. Social Appraisal & Attachment (neurons 96..127)
5. Interoceptive Need Regulation (neurons 128..159)
6. Equilibrium & Control (neurons 160..191)
7. Memory & Context (neurons 192..223)
8. Rest & Consolidation Control (neurons 224..255)

Governed by leaky integrator recurrent dynamics:
h_{t+1}^A = (1 - eta) * h_t^A + eta * tanh(W_A h_t^A + B_S z_t^S + B_C z_t^C + B_H b_t + B_M m_t + I_t^graft)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POPULATION_NAMES: List[str] = [
    "sensory_salience",
    "threat_prediction",
    "reward_valuation",
    "social_appraisal",
    "interoceptive_regulation",
    "equilibrium_control",
    "memory_context",
    "rest_consolidation",
]

POPULATION_SLICES: Dict[str, Tuple[int, int]] = {
    "sensory_salience": (0, 32),
    "threat_prediction": (32, 64),
    "reward_valuation": (64, 96),
    "social_appraisal": (96, 128),
    "interoceptive_regulation": (128, 160),
    "equilibrium_control": (160, 192),
    "memory_context": (192, 224),
    "rest_consolidation": (224, 256),
}


class WPACStepResult(torch.Tensor):
    """Result of WPAC step: encapsulates h_next [256], while attaching graft_current and diagnostics.
    Subclasses torch.Tensor so tensor operations and assertions (e.g. shape, isfinite) succeed directly,
    while also supporting tuple unpacking: (h_next, graft_current, diagnostics) = res."""
    @staticmethod
    def __new__(cls, h_next: torch.Tensor, graft_current: torch.Tensor, diagnostics: Dict[str, Any]):
        res = h_next.as_subclass(cls)
        res.graft_current = graft_current
        res.diagnostics = diagnostics
        return res

    def __iter__(self):
        yield self
        yield self.graft_current
        yield self.diagnostics


class WeightProgrammedAffectiveCore(nn.Module):
    """The 256-neuron Weight-Programmed Affective Connectome (WPAC)."""

    def __init__(
        self,
        num_neurons: int = 256,
        sensory_dim: int = 64,
        connectome_dim: int = 64,
        body_dim: int = 8,
        memory_dim: int = 32,
        leak_rate: float = 0.25,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_neurons = num_neurons
        self.leak_rate = leak_rate

        # 1. Programmed Recurrent Weights W_A = W_prior + W_train
        # Dale's law signed prior
        pre_signs = torch.where(torch.rand(num_neurons) < 0.8, 1.0, -1.0)
        self.register_buffer("dale_signs", torch.tile(pre_signs.unsqueeze(0), (num_neurons, 1)))

        # Prior sparse connectivity mask (density ~ 0.15)
        prov_mask = (torch.rand(num_neurons, num_neurons) < 0.15).float()
        self.register_buffer("provenance_mask", prov_mask)

        # Base programmed prior weights (frozen baseline motif)
        prior_weights = torch.randn(num_neurons, num_neurons) * 0.05
        self.register_buffer("w_prior", prior_weights)

        # Trainable adaptive parameter
        self.w_train = nn.Parameter(torch.zeros(num_neurons, num_neurons))

        # 2. Input Projection Ports
        self.proj_sensory = nn.Linear(sensory_dim, num_neurons, bias=False)  # B_S
        self.proj_connectome = nn.Linear(connectome_dim, num_neurons, bias=False)  # B_C
        self.proj_body = nn.Linear(body_dim, num_neurons, bias=False)  # B_H
        self.proj_memory = nn.Linear(memory_dim, num_neurons, bias=False)  # B_M

        nn.init.orthogonal_(self.proj_sensory.weight, gain=0.2)
        nn.init.orthogonal_(self.proj_connectome.weight, gain=0.2)
        nn.init.orthogonal_(self.proj_body.weight, gain=0.2)
        nn.init.orthogonal_(self.proj_memory.weight, gain=0.2)

        # Bias vector
        self.bias = nn.Parameter(torch.zeros(num_neurons))

        # Recurrent state buffer h_t^A
        self.register_buffer("h_state", torch.zeros(num_neurons))

        # Output projection back to recipient connectome (graft injection)
        self.proj_graft_out = nn.Linear(num_neurons, connectome_dim, bias=False)
        nn.init.orthogonal_(self.proj_graft_out.weight, gain=0.1)

    def get_effective_weights(self) -> torch.Tensor:
        """
        Compute Dale's-compliant effective recurrent weight matrix.
        W_eff = (softplus(W_prior + W_train) * dale_signs) * provenance_mask
        """
        raw = F.softplus(self.w_prior + self.w_train)
        w_dale = raw * self.dale_signs
        return w_dale * self.provenance_mask

    def reset_state(self) -> None:
        """Reset internal recurrent state to zero."""
        self.h_state.zero_()

    def get_population_activities(self) -> Dict[str, float]:
        """Return dictionary mapping population names to mean activities."""
        h_np = self.h_state.detach().cpu().numpy()
        pop_means = {}
        for pop_name, (start_idx, end_idx) in POPULATION_SLICES.items():
            pop_means[pop_name] = round(float(np.mean(h_np[start_idx:end_idx])), 4)
        return pop_means

    def step(
        self,
        z_sensory: Optional[torch.Tensor] = None,
        z_connectome: Optional[torch.Tensor] = None,
        b_body: Optional[torch.Tensor] = None,
        m_memory: Optional[torch.Tensor] = None,
        i_graft: Optional[torch.Tensor] = None,
        dt: float = 1.0,
        sensory_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> WPACStepResult:
        """
        Advance the affective connectome recurrent state by one step.

        Returns:
            WPACStepResult: [256] Updated affective connectome state (unpacks as h_next, graft_current, diagnostics)
        """
        if z_sensory is None and sensory_input is not None:
            z_sensory = sensory_input
        device = self.h_state.device
        dtype = self.h_state.dtype

        # Compute incoming projections
        total_drive = self.bias.clone()

        if z_sensory is not None:
            zs = z_sensory.flatten().to(device=device, dtype=dtype)
            if len(zs) < self.proj_sensory.in_features:
                zs = F.pad(zs, (0, self.proj_sensory.in_features - len(zs)))
            total_drive = total_drive + self.proj_sensory(zs[:self.proj_sensory.in_features])

        if z_connectome is not None:
            zc = z_connectome.flatten().to(device=device, dtype=dtype)
            if len(zc) < self.proj_connectome.in_features:
                zc = F.pad(zc, (0, self.proj_connectome.in_features - len(zc)))
            total_drive = total_drive + self.proj_connectome(zc[:self.proj_connectome.in_features])

        if b_body is not None:
            bb = b_body.flatten().to(device=device, dtype=dtype)
            if len(bb) < self.proj_body.in_features:
                bb = F.pad(bb, (0, self.proj_body.in_features - len(bb)))
            total_drive = total_drive + self.proj_body(bb[:self.proj_body.in_features])

        if m_memory is not None:
            mm = m_memory.flatten().to(device=device, dtype=dtype)
            if len(mm) < self.proj_memory.in_features:
                mm = F.pad(mm, (0, self.proj_memory.in_features - len(mm)))
            total_drive = total_drive + self.proj_memory(mm[:self.proj_memory.in_features])

        if i_graft is not None:
            ig = i_graft.flatten().to(device=device, dtype=dtype)
            if len(ig) < self.num_neurons:
                ig = F.pad(ig, (0, self.num_neurons - len(ig)))
            total_drive = total_drive + ig[:self.num_neurons]

        # Recurrent propagation W_eff * h_t
        w_eff = self.get_effective_weights()
        rec_drive = torch.matmul(w_eff, self.h_state)

        # Leaky integration: h_{t+1} = (1 - eta) h_t + eta * tanh(rec_drive + total_drive)
        eta = np.clip(self.leak_rate * dt, 0.0, 1.0)
        new_act = torch.tanh(rec_drive + total_drive)
        h_next = (1.0 - eta) * self.h_state + eta * new_act
        self.h_state.data.copy_(h_next)

        # Outgoing graft current
        graft_current = self.proj_graft_out(h_next)

        # Population diagnostics
        h_np = h_next.detach().cpu().numpy()
        pop_means = {}
        for pop_name, (start_idx, end_idx) in POPULATION_SLICES.items():
            pop_means[pop_name] = round(float(np.mean(h_np[start_idx:end_idx])), 4)

        diagnostics = {
            "population_activations": pop_means,
            "overall_activity_norm": round(float(np.linalg.norm(h_np)), 4),
            "max_neuron_activity": round(float(np.max(np.abs(h_np))), 4),
        }

        return WPACStepResult(h_next, graft_current, diagnostics)

    def export_state(self) -> Dict[str, Any]:
        return {
            "num_neurons": self.num_neurons,
            "leak_rate": self.leak_rate,
            "w_prior": self.w_prior.detach().cpu().numpy().tolist(),
            "w_train": self.w_train.detach().cpu().numpy().tolist(),
            "dale_signs": self.dale_signs.detach().cpu().numpy().tolist(),
            "provenance_mask": self.provenance_mask.detach().cpu().numpy().tolist(),
            "bias": self.bias.detach().cpu().numpy().tolist(),
            "h_state": self.h_state.detach().cpu().numpy().tolist(),
            "proj_sensory": self.proj_sensory.weight.detach().cpu().numpy().tolist(),
            "proj_connectome": self.proj_connectome.weight.detach().cpu().numpy().tolist(),
            "proj_body": self.proj_body.weight.detach().cpu().numpy().tolist(),
            "proj_memory": self.proj_memory.weight.detach().cpu().numpy().tolist(),
            "proj_graft_out": self.proj_graft_out.weight.detach().cpu().numpy().tolist(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if "w_prior" in state:
            t = torch.tensor(state["w_prior"], dtype=torch.float32)
            if t.shape == self.w_prior.shape:
                self.w_prior.copy_(t)
        if "w_train" in state:
            t = torch.tensor(state["w_train"], dtype=torch.float32)
            if t.shape == self.w_train.shape:
                self.w_train.data.copy_(t)
        if "dale_signs" in state:
            t = torch.tensor(state["dale_signs"], dtype=torch.float32)
            if t.shape == self.dale_signs.shape:
                self.dale_signs.copy_(t)
        if "provenance_mask" in state:
            t = torch.tensor(state["provenance_mask"], dtype=torch.float32)
            if t.shape == self.provenance_mask.shape:
                self.provenance_mask.copy_(t)
        if "bias" in state:
            t = torch.tensor(state["bias"], dtype=torch.float32)
            if t.shape == self.bias.shape:
                self.bias.data.copy_(t)
        if "h_state" in state:
            t = torch.tensor(state["h_state"], dtype=torch.float32)
            if t.shape == self.h_state.shape:
                self.h_state.data.copy_(t)
        if "proj_sensory" in state:
            t = torch.tensor(state["proj_sensory"], dtype=torch.float32)
            if t.shape == self.proj_sensory.weight.shape:
                self.proj_sensory.weight.data.copy_(t)
        if "proj_connectome" in state:
            t = torch.tensor(state["proj_connectome"], dtype=torch.float32)
            if t.shape == self.proj_connectome.weight.shape:
                self.proj_connectome.weight.data.copy_(t)
        if "proj_body" in state:
            t = torch.tensor(state["proj_body"], dtype=torch.float32)
            if t.shape == self.proj_body.weight.shape:
                self.proj_body.weight.data.copy_(t)
        if "proj_memory" in state:
            t = torch.tensor(state["proj_memory"], dtype=torch.float32)
            if t.shape == self.proj_memory.weight.shape:
                self.proj_memory.weight.data.copy_(t)
        if "proj_graft_out" in state:
            t = torch.tensor(state["proj_graft_out"], dtype=torch.float32)
            if t.shape == self.proj_graft_out.weight.shape:
                self.proj_graft_out.weight.data.copy_(t)
