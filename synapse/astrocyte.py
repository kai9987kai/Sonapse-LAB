"""Tripartite Neuro-Glia Astrocyte Modulation & BCM Sliding Metaplasticity.

Models biological astrocytic endfeet wrapping tripartite synapses across
connectome domains (flywire, manc, optic-lobe). Astrocytes maintain internal
intracellular calcium ([Ca2+]_i) dynamics driven by regional neurotransmitter
spillover, releasing gliotransmitters (ATP, D-serine, adenosine) that shift
the Bienenstock-Cooper-Munro (BCM) modification threshold and gate synaptic gain.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class AstrocyteDomainState:
    domain_id: int
    name: str
    calcium: float = 0.1
    gliotransmission: float = 1.0
    bcm_threshold: float = 0.5
    spillover_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'domain_id': self.domain_id,
            'name': self.name,
            'calcium': round(float(self.calcium), 4),
            'gliotransmission': round(float(self.gliotransmission), 4),
            'bcm_threshold': round(float(self.bcm_threshold), 4),
        }

class AstrocyteNetwork(nn.Module):
    """Multi-domain neuro-glial astrocyte network.

    Parameters:
        num_domains: Number of connectome neuropil regions (default 3: flywire, manc, optic-lobe).
        decay: Intracellular calcium clearance rate.
        calcium_gain: Coupling from regional neural firing rate to astrocytic Ca2+ influx.
    """
    def __init__(self, num_domains: int = 3, decay: float = 0.92, calcium_gain: float = 0.25):
        super().__init__()
        self.num_domains = num_domains
        self.decay = decay
        self.calcium_gain = calcium_gain
        self.domain_names = ['flywire', 'manc', 'optic-lobe']
        self.domains = [
            AstrocyteDomainState(i, self.domain_names[i % len(self.domain_names)])
            for i in range(num_domains)
        ]
        self.register_buffer('calcium_levels', torch.full((num_domains,), 0.1))
        self.register_buffer('bcm_thresholds', torch.full((num_domains,), 0.5))

    def reset(self):
        for d in self.domains:
            d.calcium = 0.1
            d.gliotransmission = 1.0
            d.bcm_threshold = 0.5
            d.spillover_history.clear()
        self.calcium_levels.fill_(0.1)
        self.bcm_thresholds.fill_(0.5)

    def step(self, neural_rates: torch.Tensor) -> torch.Tensor:
        """Update astrocytic calcium and return synaptic transmission scaling factors.

        Args:
            neural_rates: Tensor of connectome firing rates [N].

        Returns:
            domain_gains: Tensor of transmission gains [num_domains].
        """
        n = neural_rates.numel()
        domain_size = math.ceil(n / self.num_domains)
        gains = []

        for i in range(self.num_domains):
            start = i * domain_size
            end = min((i + 1) * domain_size, n)
            if start >= n:
                gains.append(1.0)
                continue
            regional_activity = neural_rates[start:end]
            mean_rate = float(regional_activity.mean().item())
            rate_var = float(regional_activity.var().item()) if len(regional_activity) > 1 else 0.0

            # 1. Intracellular calcium wave equation: dCa/dt = -decay * Ca + gain * mean_rate
            ca = float(self.calcium_levels[i].item())
            new_ca = ca * self.decay + self.calcium_gain * mean_rate
            new_ca = float(np.clip(new_ca, 0.01, 2.5))
            self.calcium_levels[i] = new_ca

            # 2. Gliotransmitter release: sigmoidal response to calcium wave
            # At resting levels, gain is 1.0; at high calcium, adenosine inhibits runaway activity
            gliotransmission = float(2.0 / (1.0 + math.exp(0.8 * (new_ca - 0.4))))
            gliotransmission = float(np.clip(gliotransmission, 0.2, 1.8))

            # 3. Sliding BCM modification threshold: theta_M = <y^2> * tau
            bcm = float(self.bcm_thresholds[i].item())
            target_bcm = float(regional_activity.pow(2).mean().item())
            new_bcm = bcm * 0.95 + 0.05 * target_bcm
            self.bcm_thresholds[i] = new_bcm

            # Update domain record
            dom = self.domains[i]
            dom.calcium = new_ca
            dom.gliotransmission = gliotransmission
            dom.bcm_threshold = new_bcm
            dom.spillover_history.append(mean_rate)
            if len(dom.spillover_history) > 64:
                dom.spillover_history.pop(0)

            gains.append(gliotransmission)

        return torch.tensor(gains, dtype=neural_rates.dtype, device=neural_rates.device)

    def modulate_graft_weights(self, graft_weights: torch.Tensor, pre_indices: torch.Tensor, total_neurons: int) -> torch.Tensor:
        """Modulate graft synaptic weights according to presynaptic astrocyte domain.

        Args:
            graft_weights: Tensor of graft edge weights [E].
            pre_indices: Presynaptic neuron indices [E].
            total_neurons: Total number of neurons in connectome.

        Returns:
            Modulated graft weights [E].
        """
        domain_size = math.ceil(total_neurons / self.num_domains)
        pre_domains = torch.clamp((pre_indices // domain_size).long(), 0, self.num_domains - 1)
        domain_gains = torch.tensor([d.gliotransmission for d in self.domains], dtype=graft_weights.dtype, device=graft_weights.device)
        return graft_weights * domain_gains[pre_domains]

    def summary(self) -> dict:
        return {
            'num_domains': self.num_domains,
            'mean_calcium': float(self.calcium_levels.mean().item()),
            'mean_bcm_threshold': float(self.bcm_thresholds.mean().item()),
            'domains': [d.to_dict() for d in self.domains],
            'scope': 'tripartite neuro-glia astrocyte calcium and BCM metaplasticity'
        }

    def export_state(self) -> dict:
        return {
            'calcium_levels': [float(x) for x in self.calcium_levels.detach().cpu().tolist()],
            'bcm_thresholds': [float(x) for x in self.bcm_thresholds.detach().cpu().tolist()],
            'domains': [
                {
                    'domain_id': d.domain_id,
                    'name': d.name,
                    'calcium': float(d.calcium),
                    'gliotransmission': float(d.gliotransmission),
                    'bcm_threshold': float(d.bcm_threshold),
                    'spillover_history': [float(x) for x in d.spillover_history]
                }
                for d in self.domains
            ]
        }

    def import_state(self, state: dict):
        if not state:
            return
        if 'calcium_levels' in state:
            self.calcium_levels.copy_(torch.tensor(state['calcium_levels'], dtype=self.calcium_levels.dtype))
        if 'bcm_thresholds' in state:
            self.bcm_thresholds.copy_(torch.tensor(state['bcm_thresholds'], dtype=self.bcm_thresholds.dtype))
        if 'domains' in state:
            for i, d_state in enumerate(state['domains']):
                if i < len(self.domains):
                    self.domains[i].calcium = float(d_state.get('calcium', 0.1))
                    self.domains[i].gliotransmission = float(d_state.get('gliotransmission', 1.0))
                    self.domains[i].bcm_threshold = float(d_state.get('bcm_threshold', 0.5))
                    self.domains[i].spillover_history = list(d_state.get('spillover_history', []))
