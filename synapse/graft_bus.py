"""Unified Neural Graft Communication Bus.

Provides a unified multi-channel neural communication bus multiplexer
enforcing the bounded composition contract:

    I_aux,t = clip_B ( sum_{k=1}^K gamma_k * g_k(c_t) * I_hat_{k,t} )

Features:
- Unifies SANCF (neural_comm), NeuroWave-X2 (spectral/phase), CrossConnectomeGraft,
  and AETHER GraftBridge into a single coherent recurrent integration interface.
- Preserves zero-gain identity: when all channels are disabled or all gamma_k = 0,
  I_aux,t = 0.0 bit-for-bit.
- Allows independent ablation, per-channel gain scaling, and live diagnostics.
- Dispatches context state c_t across all participating channels.
- Provides seamless drop-in compatibility with GraftedBrain's comm_fabric interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. ABSTRACT PROTOCOL INTERFACE
# ==============================================================================

class NeuralGraftProtocol(ABC):
    """Abstract base class for all neural graft communication protocols."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique channel identifier."""
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this protocol channel is actively proposing current."""
        pass

    @abstractmethod
    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute proposed residual neural current.
        
        Args:
            r: Recurrent neural rate state [N]
            microstep: Current simulation microstep index
            advance: Whether to advance clocks and state
            adapt: Whether online learning is enabled
            phase_angles: Optional theta phase angle tensor
            context: Optional contextual modulation tensor

        Returns:
            Tensor of proposed currents of shape [N]
        """
        pass

    @abstractmethod
    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Return diagnostic telemetry dictionary."""
        pass

    def export_state(self) -> Dict[str, Any]:
        """Export state for checkpoint persistence."""
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from checkpoint."""
        pass


# ==============================================================================
# 2. ADAPTERS FOR CORE PROTOCOL ENGINES
# ==============================================================================

class SANCFBusChannel(NeuralGraftProtocol):
    """Adapter wrapping SANCF NeuralCommFabric into the graft bus."""

    def __init__(self, fabric: Any, name: str = "sancf"):
        self.fabric = fabric
        self._name = name
        self._enabled = getattr(fabric, "enabled", True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and getattr(self.fabric, "enabled", False))

    def set_enabled(self, val: bool) -> None:
        self._enabled = bool(val)
        if hasattr(self.fabric, "enabled"):
            self.fabric.enabled = bool(val)

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        step_out = self.fabric.compute_current(
            r, microstep=microstep, advance=advance, adapt=adapt, phase_angles=phase_angles
        )
        return step_out.residual_current

    def get_telemetry_dict(self) -> Dict[str, Any]:
        if hasattr(self.fabric, "get_telemetry_dict"):
            return self.fabric.get_telemetry_dict()
        return {}

    def export_state(self) -> Dict[str, Any]:
        if hasattr(self.fabric, "export_state"):
            return self.fabric.export_state()
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        if hasattr(self.fabric, "import_state"):
            self.fabric.import_state(state)


class NeuroWaveX2BusChannel(NeuralGraftProtocol):
    """Adapter wrapping NeuroWaveX2Fabric into the graft bus."""

    def __init__(self, fabric: Any, name: str = "neurowave_x2"):
        self.fabric = fabric
        self._name = name
        self._enabled = getattr(fabric.config, "enabled", True) if hasattr(fabric, "config") else True

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and getattr(self.fabric.config, "enabled", False))

    def set_enabled(self, val: bool) -> None:
        self._enabled = bool(val)
        if hasattr(self.fabric, "config"):
            self.fabric.config.enabled = bool(val)

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        step_out = self.fabric.compute_current(
            r, microstep=microstep, advance=advance, adapt=adapt, phase_angles=phase_angles
        )
        return step_out.residual_current

    def get_telemetry_dict(self) -> Dict[str, Any]:
        if hasattr(self.fabric, "get_telemetry_dict"):
            return self.fabric.get_telemetry_dict()
        return {}

    def export_state(self) -> Dict[str, Any]:
        if hasattr(self.fabric, "export_state"):
            return self.fabric.export_state()
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        if hasattr(self.fabric, "import_state"):
            self.fabric.import_state(state)


class CrossGraftBusChannel(NeuralGraftProtocol):
    """Adapter wrapping CrossConnectomeGraft into the graft bus."""

    def __init__(self, engine: Any, name: str = "cross_graft"):
        self.engine = engine
        self._name = name
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and getattr(self.engine, "gamma", 0.0) != 0.0)

    def set_enabled(self, val: bool) -> None:
        self._enabled = bool(val)

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.engine.gamma == 0.0:
            return torch.zeros_like(r)
        
        # Inject context and source activations
        current, _ = self.engine(r, context=context, neuromod_da=1.0)
        
        # Match dimensions with r
        if len(current) < len(r):
            current = F.pad(current, (0, len(r) - len(current)))
        elif len(current) > len(r):
            current = current[:len(r)]
        return current

    def get_telemetry_dict(self) -> Dict[str, Any]:
        if hasattr(self.engine, "telemetry") and hasattr(self.engine.telemetry, "to_dict"):
            return self.engine.telemetry.to_dict()
        return {}

    def export_state(self) -> Dict[str, Any]:
        if hasattr(self.engine, "export_state"):
            return self.engine.export_state()
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        if hasattr(self.engine, "import_state"):
            self.engine.import_state(state)


class AetherBridgeBusChannel(NeuralGraftProtocol):
    """Adapter wrapping AetherGraftBridge into the graft bus."""

    def __init__(self, bridge: Any, name: str = "aether_bridge"):
        self.bridge = bridge
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self.bridge.enabled and self.bridge.spec.coupling_gain != 0.0)

    def set_enabled(self, val: bool) -> None:
        self.bridge.spec.enabled = bool(val)

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        
        current, _ = self.bridge(r, r, advance=advance, adapt=adapt)
        if len(current) < len(r):
            current = F.pad(current, (0, len(r) - len(current)))
        elif len(current) > len(r):
            current = current[:len(r)]
        return current

    def get_telemetry_dict(self) -> Dict[str, Any]:
        if hasattr(self.bridge, "telemetry") and hasattr(self.bridge.telemetry, "to_dict"):
            return self.bridge.telemetry.to_dict()
        return {}

    def export_state(self) -> Dict[str, Any]:
        if hasattr(self.bridge, "export_state"):
            return self.bridge.export_state()
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        if hasattr(self.bridge, "import_state"):
            self.bridge.import_state(state)


# ==============================================================================
# 3. BUS STEP OUTPUT
# ==============================================================================

@dataclass
class GraftBusStepOutput:
    """Output bundle from a Graft Communication Bus computation."""
    residual_current: torch.Tensor
    channel_currents: Dict[str, torch.Tensor] = field(default_factory=dict)
    channel_norms: Dict[str, float] = field(default_factory=dict)
    total_norm: float = 0.0
    saturation_ratio: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 4. UNIFIED GRAFT COMMUNICATION BUS
# ==============================================================================

class NeuralGraftCommunicationBus(nn.Module):
    """
    Central neural communication and grafting bus compositor.
    
    Coordinates multiple heterogeneous channels:
    - SANCF (low-rank region-to-region message passing)
    - NeuroWave-X2 (spectral modes & phase-coherent wave channels)
    - CrossConnectomeGraft (biological multi-dataset synaptic bridge)
    - AETHER GraftBridge (Laplacian modal carrier transfer)

    Formula:
        I_aux,t = clip_B ( sum_{k=1}^K gamma_k * g_k(c_t) * I_hat_{k,t} )
    """
    def __init__(
        self,
        max_aux_current: float = 0.5,
        context_dim: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.max_aux_current = float(max_aux_current)
        self.context_dim = int(context_dim)
        self.seed = int(seed)

        # Registered channel registry: name -> channel adapter
        self.channels: Dict[str, NeuralGraftProtocol] = {}
        # Channel gain multipliers gamma_k (default 1.0)
        self.channel_gains: Dict[str, float] = {}

        # Context distribution register
        self._current_context: Optional[torch.Tensor] = None

        # Context routing gates for channels: Linear(context_dim, 1) producing gate in [0, 1]
        self.context_gates = nn.ModuleDict()

        # Telemetry aggregation
        self.last_step_output: Optional[GraftBusStepOutput] = None
        self.step_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal step counters and child channel state."""
        self.step_counter = 0
        self.last_step_output = None
        self._current_context = None
        for ch in self.channels.values():
            if hasattr(ch, "reset"):
                ch.reset()
            elif hasattr(ch, "fabric") and hasattr(ch.fabric, "reset"):
                ch.fabric.reset()
            elif hasattr(ch, "engine") and hasattr(ch.engine, "reset"):
                ch.engine.reset()
            elif hasattr(ch, "bridge") and hasattr(ch.bridge, "reset"):
                ch.bridge.reset()

    @property
    def enabled(self) -> bool:
        """Bus is enabled if any registered channel is enabled and has non-zero gain."""
        for name, ch in self.channels.items():
            gain = self.channel_gains.get(name, 1.0)
            if ch.enabled and gain != 0.0:
                return True
        return False

    def register_channel(
        self,
        name: str,
        channel: NeuralGraftProtocol,
        gain: float = 1.0,
    ) -> None:
        """Register a protocol channel onto the communication bus."""
        self.channels[name] = channel
        self.channel_gains[name] = float(gain)
        if name not in self.context_gates:
            gate = nn.Linear(self.context_dim, 1)
            # Initialize to identity bias (gate ~ 1.0 initially)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 2.0)
            self.context_gates[name] = gate

    def unregister_channel(self, name: str) -> Optional[NeuralGraftProtocol]:
        """Remove a channel from the bus."""
        self.channel_gains.pop(name, None)
        return self.channels.pop(name, None)

    def get_channel(self, name: str) -> Optional[NeuralGraftProtocol]:
        """Retrieve a registered channel by name."""
        return self.channels.get(name)

    def set_channel_gain(self, name: str, gain: float) -> None:
        """Set residual gain multiplier for a specific channel."""
        if name in self.channels:
            self.channel_gains[name] = float(gain)

    def set_channel_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a specific channel."""
        ch = self.channels.get(name)
        if ch is not None and hasattr(ch, "set_enabled"):
            ch.set_enabled(enabled)

    def set_context(self, context: Optional[torch.Tensor]) -> None:
        """Distribute current context state across the bus."""
        if context is not None:
            c = context.detach().flatten()
            if len(c) < self.context_dim:
                c = F.pad(c, (0, self.context_dim - len(c)))
            elif len(c) > self.context_dim:
                c = c[:self.context_dim]
            self._current_context = c
        else:
            self._current_context = None

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
    ) -> GraftBusStepOutput:
        """
        Unified recurrent microstep computation conforming to GraftedBrain comm_fabric interface.

        Returns:
            GraftBusStepOutput with .residual_current tensor of shape matching r.
        """
        device = r.device
        dtype = r.dtype
        n_neurons = r.shape[0]

        # Zero-gain guarantee: if bus disabled, output is strictly zeros bit-for-bit
        if not self.enabled:
            zero_residual = torch.zeros(n_neurons, dtype=dtype, device=device)
            out = GraftBusStepOutput(
                residual_current=zero_residual,
                total_norm=0.0,
                saturation_ratio=0.0,
                diagnostics={"enabled": False, "num_active_channels": 0},
            )
            self.last_step_output = out
            return out

        if advance:
            self.step_counter += 1

        accumulated_current = torch.zeros(n_neurons, dtype=dtype, device=device)
        channel_currents: Dict[str, torch.Tensor] = {}
        channel_norms: Dict[str, float] = {}

        ctx = self._current_context
        if ctx is not None:
            ctx = ctx.to(device=device, dtype=dtype)

        active_count = 0

        # Iterate through all registered channels
        for name, channel in self.channels.items():
            gain = self.channel_gains.get(name, 1.0)
            if not channel.enabled or gain == 0.0:
                continue

            # Compute channel current proposal
            raw_cand = channel.compute_current(
                r,
                microstep=microstep,
                advance=advance,
                adapt=adapt,
                phase_angles=phase_angles,
                context=ctx,
            )

            # Ensure dimensional match
            if raw_cand.numel() < n_neurons:
                cand = F.pad(raw_cand, (0, n_neurons - raw_cand.numel()))
            elif raw_cand.numel() > n_neurons:
                cand = raw_cand[:n_neurons]
            else:
                cand = raw_cand
            cand = cand.to(device=device, dtype=dtype)

            # Context route gating g_k(c_t)
            if ctx is not None and name in self.context_gates:
                gate_val = float(torch.sigmoid(self.context_gates[name](ctx)).item())
            else:
                gate_val = 1.0

            scaled_current = gain * gate_val * cand
            accumulated_current = accumulated_current + scaled_current

            cur_norm = float(scaled_current.norm().item())
            channel_currents[name] = scaled_current
            channel_norms[name] = cur_norm
            active_count += 1

        # Multi-channel bounded clipping: clip_B ( sum gamma_k g_k I_k )
        B = self.max_aux_current
        final_current = torch.clamp(accumulated_current, -B, B)

        # Compute saturation ratio (fraction of units hitting clip bound)
        sat_count = (final_current.abs() >= (B - 1e-4)).sum().item()
        sat_ratio = float(sat_count) / float(max(1, n_neurons))
        tot_norm = float(final_current.norm().item())

        diagnostics = {
            "enabled": True,
            "max_aux_bound": B,
            "num_active_channels": active_count,
            "total_norm": round(tot_norm, 6),
            "saturation_ratio": round(sat_ratio, 4),
            "channel_contributions": {k: round(v, 6) for k, v in channel_norms.items()},
        }

        out = GraftBusStepOutput(
            residual_current=final_current,
            channel_currents=channel_currents,
            channel_norms=channel_norms,
            total_norm=tot_norm,
            saturation_ratio=sat_ratio,
            diagnostics=diagnostics,
        )
        self.last_step_output = out
        return out

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Aggregate telemetry from bus and all member protocol channels."""
        channels_telem = {}
        for name, ch in self.channels.items():
            channels_telem[name] = ch.get_telemetry_dict()

        bus_telem = {
            "enabled": self.enabled,
            "max_aux_current": self.max_aux_current,
            "step_counter": self.step_counter,
            "active_channels": [name for name, ch in self.channels.items() if ch.enabled],
            "channel_gains": dict(self.channel_gains),
            "last_step": self.last_step_output.diagnostics if self.last_step_output else {},
            "channels": channels_telem,
        }
        return bus_telem

    def export_state(self) -> Dict[str, Any]:
        """Export state for checkpoint persistence."""
        channels_state = {}
        for name, ch in self.channels.items():
            channels_state[name] = ch.export_state()

        gates_state = {}
        for name, g in self.context_gates.items():
            gates_state[name] = {
                "weight": g.weight.detach().cpu().numpy().tolist(),
                "bias": g.bias.detach().cpu().numpy().tolist(),
            }

        context_data = None
        if self._current_context is not None:
            context_data = self._current_context.detach().cpu().numpy().tolist()

        return {
            "max_aux_current": float(self.max_aux_current),
            "channel_gains": dict(self.channel_gains),
            "channels": channels_state,
            "context_gates": gates_state,
            "step_counter": int(self.step_counter),
            "context": context_data,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if "context" in state and state["context"] is not None:
            self._current_context = torch.tensor(state["context"], dtype=torch.float32)
        else:
            self._current_context = None
        """Import state from checkpoint."""
        if "max_aux_current" in state:
            self.max_aux_current = float(state["max_aux_current"])
        if "channel_gains" in state:
            for k, v in state["channel_gains"].items():
                if k in self.channel_gains:
                    self.channel_gains[k] = float(v)
        if "channels" in state:
            for k, ch_s in state["channels"].items():
                if k in self.channels:
                    self.channels[k].import_state(ch_s)
        if "context_gates" in state:
            for k, g_s in state["context_gates"].items():
                if k in self.context_gates and "weight" in g_s and "bias" in g_s:
                    w = torch.tensor(g_s["weight"], dtype=torch.float32)
                    b = torch.tensor(g_s["bias"], dtype=torch.float32)
                    if w.shape == self.context_gates[k].weight.shape:
                        self.context_gates[k].weight.data.copy_(w)
                    if b.shape == self.context_gates[k].bias.shape:
                        self.context_gates[k].bias.data.copy_(b)
        if "step_counter" in state:
            self.step_counter = int(state["step_counter"])
