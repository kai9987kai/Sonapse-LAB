"""Master NeuroWave-X2 Neural Communication Fabric.

Additive, default-OFF inter-region neural communication system providing:
- Zero-gain identity guarantee: gamma=0.0 produces strictly zero additive current.
- Graph-Laplacian spectral communication & hybrid learned carriers.
- Communication-Through-Coherence (CTC) phase alignment.
- Learned context-dependent communication protocol selection.
- Multi-timescale queues and event-driven triggering.
- Read-only side-effect-free evaluation when advance=False.
- Live observatory telemetry and deterministic replay serialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablations import CausalAblationController
from .compute_profiler import ComputeProfiler
from .config import NeuroWaveX2Config
from .hybrid_basis import HybridBasis
from .packets import NeuralPacket
from .phase_channel import PhaseCoherenceAdapter, WaveChannel
from .protocol_registry import PROTOCOLS, get_protocol_spec
from .protocol_selector import ProtocolSelector
from .resource_budget import ResourceBudgetController
from .spectral_basis import SpectralBasisManager
from .telemetry import NeuroWaveTelemetryAggregator
from .topology_controller import CommunicationTopologyController
from .transport import PyTorchTransport


@dataclass
class NeuroWaveStepOutput:
    """Standardized output returned to GraftedBrain recurrence."""
    residual_current: torch.Tensor
    wire_bits: int = 0
    analytical_flops: float = 0.0
    active_protocol: str = "vector"


class NeuroWaveX2Fabric(nn.Module):
    """Integrated NeuroWave-X2 neural communication fabric module."""

    def __init__(
        self,
        config: Optional[NeuroWaveX2Config] = None,
        num_regions: int = 8,
        neurons_per_region: int = 64,
        total_neurons: int = 512,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.config = config or NeuroWaveX2Config()
        self.enabled = bool(self.config.enabled)
        self.residual_gain = float(self.config.residual_gain)
        self.num_regions = max(1, int(num_regions))
        self.neurons_per_region = int(neurons_per_region)
        self.total_neurons = int(total_neurons)
        self.seed = int(seed)

        # 1. Topology & Spectral Basis Managers
        self.topology_ctrl = CommunicationTopologyController(
            num_regions=self.num_regions,
            max_in_degree=4,
            max_out_degree=4,
            allow_self_loops=False,
        )
        self.spectral_mgr = SpectralBasisManager(
            num_modes=self.config.spectral_modes,
            carrier_strategy="lowest_frequency",
        )

        # Default initial topology: ring + cross-bridges
        for i in range(self.num_regions):
            next_i = (i + 1) % self.num_regions
            self.topology_ctrl.add_route(i, next_i)
        if self.num_regions >= 4:
            self.topology_ctrl.add_route(0, 2)
            self.topology_ctrl.add_route(1, 3)

        # 2. Hybrid Basis & Wave Channel
        self.hybrid_basis = HybridBasis(
            num_graph_modes=max(1, self.config.spectral_modes // 2),
            num_learned_modes=max(1, self.config.spectral_modes - self.config.spectral_modes // 2),
            dimension=self.config.message_dimension,
            basis_type=self.config.spectral_basis,
            seed=self.seed,
        )
        # Initialize graph basis from current topology
        adj = self.topology_ctrl.get_adjacency_matrix()
        ver, evals, basis = self.spectral_mgr.get_or_compute_basis(adj)
        # Pad or truncate to message dimension
        padded_basis = np.zeros((basis.shape[0], self.config.message_dimension), dtype=np.float32)
        min_cols = min(basis.shape[1], self.config.message_dimension)
        padded_basis[:, :min_cols] = basis[:, :min_cols]
        self.hybrid_basis.set_graph_basis(padded_basis, version=ver)

        # 3. Wave Channel & Phase Adapter
        self.wave_channel = WaveChannel(
            message_dim=self.config.message_dimension,
            num_modes=self.config.spectral_modes,
            num_regions=self.num_regions,
        )

        # 4. Protocol Selector
        self.protocol_selector = ProtocolSelector(
            context_dim=32,
            hidden_dim=32,
            mode=self.config.protocol_selector,
            default_protocol=self.config.protocol if self.config.protocol != "adaptive" else "vector",
        )

        # 5. Differentiable Transport Engine
        self.transport = PyTorchTransport(
            dimension=self.config.message_dimension,
            precision_bits=self.config.precision_bits,
            max_latency=self.config.max_latency,
            packet_loss=self.config.packet_loss,
            noise_std=self.config.noise,
            max_aux_current=0.5,
            seed=self.seed,
        )

        # 6. Encoder & Decoder Banks
        # Source pooling: summary of 32 dims from region
        self.pool_dim = 32
        self.encoder = nn.Sequential(
            nn.Linear(self.pool_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.config.message_dimension),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.config.message_dimension, 64),
            nn.SiLU(),
            nn.Linear(64, self.neurons_per_region),
        )

        # 7. Controllers & Profilers
        self.budget_ctrl = ResourceBudgetController(
            max_wire_bits_per_step=self.config.max_payload_bits,
            max_flops_per_step=self.config.compute_budget,
        )
        self.profiler = ComputeProfiler()
        self.telemetry = NeuroWaveTelemetryAggregator()
        self.ablation_ctrl = CausalAblationController(ablation_type="none", seed=self.seed)

        self.step_count = 0
        self.sequence_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets dynamic state, queues, and step counters."""
        if seed is not None:
            self.seed = int(seed)
        self.transport.reset(seed=self.seed)
        self.wave_channel.phase_adapter.reset()
        self.step_count = 0
        self.sequence_counter = 0

    def set_ablation(self, ablation_type: str, target_route: Optional[Tuple[int, int]] = None) -> None:
        """Applies causal intervention during evaluation."""
        self.ablation_ctrl = CausalAblationController(
            ablation_type=ablation_type, target_route=target_route, seed=self.seed
        )

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
    ) -> NeuroWaveStepOutput:
        """Calculates bounded auxiliary residual current to add into recurrent connectome dynamics.
        
        Zero-Gain Identity Guarantee:
            If enabled is False or residual_gain == 0.0, immediately returns zero tensor with zero overhead.
        """
        zero_residual = torch.zeros(self.total_neurons, dtype=r.dtype, device=r.device)

        if not self.enabled or self.residual_gain == 0.0:
            return NeuroWaveStepOutput(residual_current=zero_residual, wire_bits=0, analytical_flops=0.0)

        # Side-effect-free evaluation guard when advance=False
        if advance:
            self.step_count += 1
            # Step phase oscillators
            self.wave_channel.phase_adapter.step(external_coherence_phases=phase_angles)

        routes = self.topology_ctrl.list_routes()
        if len(routes) == 0:
            return NeuroWaveStepOutput(residual_current=zero_residual, wire_bits=0, analytical_flops=0.0)

        # 1. Global context for protocol selector
        r_pool = F.adaptive_avg_pool1d(r.detach().reshape(1, 1, -1), 32).reshape(32)
        if self.config.protocol == "adaptive":
            sel_proto, proto_probs, proto_diag = self.protocol_selector(
                r_pool,
                max_development_level=self.config.development_level if self.config.development_mode else 6,
                deterministic=not self.training,
            )
        else:
            sel_proto = self.config.protocol
            proto_probs = torch.zeros(len(PROTOCOLS))
            proto_diag = {}

        if sel_proto == "no_comm":
            return NeuroWaveStepOutput(residual_current=zero_residual, wire_bits=0, analytical_flops=0.0, active_protocol="no_comm")

        spec = get_protocol_spec(sel_proto)
        total_step_wire_bits = 0
        total_step_flops = 0.0
        generated_packets: List[NeuralPacket] = []

        # 2. Encode and transmit packets along active routes
        for src, dst in routes:
            # Source slice
            src_start = src * self.neurons_per_region
            src_end = min(src_start + self.neurons_per_region, r.shape[0])
            r_src = r[src_start:src_end]
            if r_src.numel() == 0:
                continue

            pooled_src = F.adaptive_avg_pool1d(r_src.reshape(1, 1, -1), self.pool_dim).reshape(self.pool_dim)
            message = self.encoder(pooled_src)

            # Determine payload and phase metadata based on protocol
            if spec.uses_spectral:
                c_tuple, p_tuple = self.wave_channel.encode_wave_packet_payload(
                    message, src_region=src, band="theta"
                )
                payload = c_tuple
                phase_meta = p_tuple if spec.uses_phase else ()
            elif spec.uses_phase:
                p_val = self.wave_channel.phase_adapter.get_phase(src, band="theta")
                payload = tuple(float(m.item()) for m in message)
                phase_meta = (p_val,)
            else:
                payload = tuple(float(m.item()) for m in message)
                phase_meta = ()

            if advance:
                self.sequence_counter += 1
                seq_id = self.sequence_counter
            else:
                seq_id = 0

            pkt = NeuralPacket(
                sender_region=src,
                receiver_region=dst,
                emission_step=self.step_count,
                sequence_id=seq_id,
                topology_version=self.topology_ctrl.topology_version,
                protocol_id=sel_proto,
                payload=payload,
                precision_bits=self.config.precision_bits,
                basis_version=self.hybrid_basis.basis_version,
                phase_metadata=phase_meta,
            )
            total_step_wire_bits += pkt.total_wire_bits
            generated_packets.append(pkt)

            if advance:
                self.transport.send(pkt, current_step=self.step_count)

        # 3. Apply Causal Ablation Intervention (if active)
        if advance:
            ready_pkts = self.transport.receive(current_step=self.step_count, receiver_id=None)
        else:
            ready_pkts = list(generated_packets)

        filtered_pkts = self.ablation_ctrl.apply_ablation_to_packets(ready_pkts)

        # 4. Decode packets into destination neural targets
        residual_full = torch.zeros(self.total_neurons, dtype=r.dtype, device=r.device)
        basis_t = self.hybrid_basis.get_effective_basis()

        for dst in range(self.num_regions):
            dst_pkts = [p for p in filtered_pkts if p.receiver_region == dst]
            if not dst_pkts:
                continue

            dst_start = dst * self.neurons_per_region
            dst_end = min(dst_start + self.neurons_per_region, self.total_neurons)
            dst_len = dst_end - dst_start
            if dst_len <= 0:
                continue

            base_act = r[dst_start:dst_end].detach()
            # Decoder weight matrix: (message_dim, dst_len)
            dec_w = self.decoder[2].weight[:dst_len, :self.config.message_dimension].T

            receiver_phase = torch.tensor(
                self.wave_channel.phase_adapter.get_phase(dst, band="theta"),
                dtype=r.dtype,
                device=r.device,
            )

            h_new, raw_delta = self.transport.decode_and_inject(
                packets=dst_pkts,
                base_activation=base_act,
                decoder_weights=dec_w,
                basis=basis_t,
                receiver_phase=receiver_phase,
                gamma=self.residual_gain,
            )

            residual_full[dst_start:dst_end] = (h_new - base_act)

        # Estimate analytical FLOPs
        flop_dict = self.profiler.estimate_step_flops(
            active_routes=len(routes),
            message_dim=self.config.message_dimension,
            target_size=self.neurons_per_region,
            spectral_modes=self.config.spectral_modes if spec.uses_spectral else 0,
        )
        total_step_flops = flop_dict["analytical_flops_per_step"]

        # 5. Record telemetry if advance=True
        if advance:
            res_norm = float(residual_full.norm().item())
            phase_aligns = {
                f"{s}->{d}": self.wave_channel.phase_adapter.compute_coherence_gain(s, d)
                for s, d in routes[:4]
            }
            self.telemetry.record_step(
                active_routes=routes,
                protocol=sel_proto,
                wire_bits=total_step_wire_bits,
                flops=total_step_flops,
                residual_norm=res_norm,
                phase_alignments=phase_aligns,
                protocol_probs=proto_diag,
                dropped=self.transport.dropped_count,
                transmitted=self.transport.transmitted_count,
                topology_version=self.topology_ctrl.topology_version,
                basis_version=self.hybrid_basis.basis_version,
            )

        return NeuroWaveStepOutput(
            residual_current=residual_full,
            wire_bits=total_step_wire_bits,
            analytical_flops=total_step_flops,
            active_protocol=sel_proto,
        )

    def export_state(self) -> Dict[str, Any]:
        """Serializes complete subsystem state for deterministic checkpoints."""
        return {
            "config": self.config.to_dict(),
            "topology_routes": self.topology_ctrl.list_routes(),
            "topology_version": self.topology_ctrl.topology_version,
            "basis_version": self.hybrid_basis.basis_version,
            "step_count": self.step_count,
            "sequence_counter": self.sequence_counter,
            "telemetry": self.telemetry.export_state() if hasattr(self, "telemetry") else {},
            "budget": {
                "accumulated_wire_bits": self.budget_ctrl.accumulated_wire_bits,
                "accumulated_flops": self.budget_ctrl.accumulated_flops,
                "budget_exceeded_count": self.budget_ctrl.budget_exceeded_count,
            } if hasattr(self, "budget_ctrl") else {},
            "profiler": {
                "latency_history": list(self.profiler.latency_history),
            } if hasattr(self, "profiler") else {},
            "wave_channel": self.wave_channel.export_state() if hasattr(self, "wave_channel") else {},
            "transport": self.transport.export_state() if hasattr(self, "transport") else {},
            "model_state": {k: v.cpu() for k, v in self.state_dict().items()},
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restores state from serialized checkpoint."""
        if not state:
            return
        if "config" in state:
            self.config = NeuroWaveX2Config.from_dict(state["config"])
            self.enabled = bool(self.config.enabled)
            self.residual_gain = float(self.config.residual_gain)
        if "topology_routes" in state:
            self.topology_ctrl.routes = set(tuple(r) for r in state["topology_routes"])
            self.topology_ctrl.topology_version = int(state.get("topology_version", 1))
        if "step_count" in state:
            self.step_count = int(state["step_count"])
            self.sequence_counter = int(state.get("sequence_counter", 0))
        if "telemetry" in state and hasattr(self, "telemetry"):
            self.telemetry.import_state(state["telemetry"])
        if "budget" in state and hasattr(self, "budget_ctrl"):
            b = state["budget"]
            self.budget_ctrl.accumulated_wire_bits = int(b.get("accumulated_wire_bits", 0))
            self.budget_ctrl.accumulated_flops = float(b.get("accumulated_flops", 0.0))
            self.budget_ctrl.budget_exceeded_count = int(b.get("budget_exceeded_count", 0))
        if "profiler" in state and hasattr(self, "profiler"):
            self.profiler.latency_history = list(state["profiler"].get("latency_history", []))
        if "wave_channel" in state and hasattr(self, "wave_channel"):
            self.wave_channel.import_state(state["wave_channel"])
        if "transport" in state and hasattr(self, "transport"):
            self.transport.import_state(state["transport"])
        if "basis_version" in state:
            self.hybrid_basis.basis_version = int(state["basis_version"])
            self.spectral_mgr.basis_version = int(state["basis_version"])
        if "model_state" in state:
            self.load_state_dict(state["model_state"], strict=False)

    def get_telemetry_dict(self) -> Dict[str, Any]:
        return self.telemetry.export_observatory_payload()
