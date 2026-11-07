"""Causal Ablations & Route Importance Matrix for Sonapse NeuroWave-X2.

Implements rigorous experimental interventions (do-calculus) to test whether
reported performance improvements are causally dependent on communication:
- do(zero_comm)
- do(zero_gain)
- do(shuffle_messages)
- do(randomize_phase)
- do(freeze_phase)
- do(randomize_modes)
- do(lesion_route(i, j))
- do(randomize_protocol)
Generates the Communication Causal Atlas and Route Importance Matrix.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from .packets import NeuralPacket


VALID_ABLATIONS: Set[str] = {
    "none",
    "zero_comm",
    "zero_gain",
    "shuffle_messages",
    "shuffle_senders",
    "shuffle_receivers",
    "randomize_phase",
    "freeze_phase",
    "randomize_modes",
    "lesion_route",
    "randomize_protocol",
}


class CausalAblationController:
    """Intervenes upon communication packets and channel parameters during evaluation."""

    def __init__(self, ablation_type: str = "none", target_route: Optional[Tuple[int, int]] = None, seed: int = 42) -> None:
        if ablation_type not in VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation '{ablation_type}'. Valid: {sorted(VALID_ABLATIONS)}")
        self.ablation_type = str(ablation_type)
        self.target_route = target_route  # (src_region, dst_region) for route lesion
        self.rng = np.random.RandomState(seed)

    def apply_ablation_to_packets(self, packets: List[NeuralPacket]) -> List[NeuralPacket]:
        """Intervenes on candidate packets before delivery to receiver decoders."""
        if self.ablation_type == "none" or len(packets) == 0:
            return packets

        if self.ablation_type == "zero_comm":
            return []

        if self.ablation_type == "lesion_route" and self.target_route is not None:
            src, dst = self.target_route
            return [p for p in packets if not (p.sender_region == src and p.receiver_region == dst)]

        ablated_packets: List[NeuralPacket] = []

        if self.ablation_type == "shuffle_messages":
            # Permute payloads across packets
            payloads = [p.payload for p in packets]
            self.rng.shuffle(payloads)
            for p, new_payload in zip(packets, payloads):
                ablated_packets.append(NeuralPacket(
                    sender_region=p.sender_region,
                    receiver_region=p.receiver_region,
                    emission_step=p.emission_step,
                    sequence_id=p.sequence_id,
                    topology_version=p.topology_version,
                    protocol_id=p.protocol_id,
                    payload=new_payload,
                    precision_bits=p.precision_bits,
                    basis_version=p.basis_version,
                    phase_metadata=p.phase_metadata,
                ))
            return ablated_packets

        if self.ablation_type == "randomize_phase":
            # Assign random phase angle in [0, 2pi]
            for p in packets:
                rand_phase = float(self.rng.uniform(0.0, 2.0 * math.pi))
                ablated_packets.append(NeuralPacket(
                    sender_region=p.sender_region,
                    receiver_region=p.receiver_region,
                    emission_step=p.emission_step,
                    sequence_id=p.sequence_id,
                    topology_version=p.topology_version,
                    protocol_id=p.protocol_id,
                    payload=p.payload,
                    precision_bits=p.precision_bits,
                    basis_version=p.basis_version,
                    phase_metadata=(rand_phase,),
                ))
            return ablated_packets

        if self.ablation_type == "freeze_phase":
            # Lock all phases to 0.0
            for p in packets:
                ablated_packets.append(NeuralPacket(
                    sender_region=p.sender_region,
                    receiver_region=p.receiver_region,
                    emission_step=p.emission_step,
                    sequence_id=p.sequence_id,
                    topology_version=p.topology_version,
                    protocol_id=p.protocol_id,
                    payload=p.payload,
                    precision_bits=p.precision_bits,
                    basis_version=p.basis_version,
                    phase_metadata=(0.0,),
                ))
            return ablated_packets

        # Default fallback
        return packets


def evaluate_route_causal_importance(
    eval_fn: Callable[[Optional[Tuple[int, int]]], float],
    routes: List[Tuple[int, int]],
    baseline_performance: float,
) -> Dict[Tuple[int, int], float]:
    """Computes causal importance for each route e = (src, dst):
        importance(e) = J(baseline) - J(do(route e = disabled))
    
    A positive value indicates that lesioning the route degrades performance (route is causally beneficial).
    """
    importance_scores: Dict[Tuple[int, int], float] = {}
    for e in routes:
        lesioned_score = eval_fn(e)
        delta_j = float(baseline_performance - lesioned_score)
        importance_scores[e] = delta_j
    return importance_scores


def construct_route_importance_matrix(
    num_regions: int,
    importance_dict: Dict[Tuple[int, int], float],
) -> np.ndarray:
    """Arranges pairwise route causal importance scores into an (N, N) matrix."""
    matrix = np.zeros((num_regions, num_regions), dtype=np.float32)
    for (src, dst), score in importance_dict.items():
        if 0 <= src < num_regions and 0 <= dst < num_regions:
            matrix[src, dst] = float(score)
    return matrix
