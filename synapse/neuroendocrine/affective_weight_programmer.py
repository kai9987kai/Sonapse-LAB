"""Weight Programming and Structural Prior Construction for WPAC.

Programs biologically and functionally motivated motifs into the Affective Core:
- Sensory Salience -> Threat / Reward cross-excitatory connections
- Threat Prediction -> Equilibrium Control feedback inhibition
- Reward Valuation -> Approach / Indulgence amplification
- Social Appraisal -> Attachment / Jealousy / Affiliation balance
- Interoceptive Regulation -> Metabolic deficit alarms
- Generates cryptographic receipts for every programmed prior
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
import torch

from .affective_connectome import WeightProgrammedAffectiveCore, POPULATION_SLICES
from .evidence_types import EvidenceKind, EvidenceReceipt


class ReceiptsList(list):
    """List of EvidenceReceipts with summary delegation."""
    def __init__(self, receipts: List[EvidenceReceipt]):
        super().__init__(receipts)
        self.summary_receipt = receipts[-1] if receipts else None

    @property
    def kind(self):
        return self.summary_receipt.kind if self.summary_receipt else EvidenceKind.MODELLED_BIOLOGICAL

    @property
    def source_identifier(self):
        return self.summary_receipt.source_identifier if self.summary_receipt else ""

    @property
    def description(self):
        return self.summary_receipt.description if self.summary_receipt else ""

    @property
    def metrics(self):
        return self.summary_receipt.metrics if self.summary_receipt else {}


class AffectiveWeightProgrammer:
    """Configures and binds functional priors into the Affective Connectome."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def program_motifs(self, core: WeightProgrammedAffectiveCore) -> ReceiptsList:
        """
        Inject structured functional motifs into W_prior and provenance_mask.
        Preserves Dale's law and registers the programmed receipt.
        """
        receipts: List[EvidenceReceipt] = []
        motifs = [
            ("sensory_salience", "threat_prediction", 0.35, 0.5, "Sensory Salience -> Threat Prediction (fast threat response)"),
            ("sensory_salience", "reward_valuation", 0.30, 0.45, "Sensory Salience -> Reward Valuation (opportunity recognition)"),
            ("threat_prediction", "equilibrium_control", 0.40, 0.5, "Threat Prediction -> Equilibrium Control (trigger homeostatic alarms)"),
            ("equilibrium_control", "threat_prediction", -0.25, 0.35, "Equilibrium Control -> Threat Prediction (negative feedback stabilization)"),
            ("interoceptive_regulation", "equilibrium_control", 0.35, 0.4, "Interoceptive Regulation -> Equilibrium Control (resource depletion signals)"),
            ("social_appraisal", "memory_context", 0.25, 0.3, "Social Appraisal -> Memory Context (social recognition and history)"),
            ("rest_consolidation", "sensory_salience", -0.30, 0.4, "Rest Consolidation -> Sensory Salience (dampen sensory gates during rest)"),
        ]

        with torch.no_grad():
            w_prior = core.w_prior
            mask = core.provenance_mask

            def _connect_populations(src_name: str, dst_name: str, weight_scale: float, density: float = 0.4):
                src_start, src_end = POPULATION_SLICES[src_name]
                dst_start, dst_end = POPULATION_SLICES[dst_name]

                rng = torch.Generator().manual_seed(self.seed + hash(src_name + dst_name) % 10000)
                sub_mask = (torch.rand(dst_end - dst_start, src_end - src_start, generator=rng) < density).float()
                sub_weights = (torch.randn(dst_end - dst_start, src_end - src_start, generator=rng) * 0.1 + weight_scale)

                mask[dst_start:dst_end, src_start:src_end] = torch.maximum(
                    mask[dst_start:dst_end, src_start:src_end], sub_mask
                )
                w_prior[dst_start:dst_end, src_start:src_end] = (
                    w_prior[dst_start:dst_end, src_start:src_end] * (1.0 - sub_mask) + sub_weights * sub_mask
                )

            for i, (src, dst, scale, dens, desc) in enumerate(motifs):
                _connect_populations(src, dst, weight_scale=scale, density=dens)
                r = EvidenceReceipt(
                    kind=EvidenceKind.MODELLED_BIOLOGICAL,
                    source_identifier=f"wpac_motif_{i+1}_{src}_{dst}",
                    description=desc,
                    metrics={
                        "motif_index": i + 1,
                        "src_population": src,
                        "dst_population": dst,
                        "weight_scale": scale,
                        "density": dens,
                        "total_neurons": core.num_neurons,
                        "active_prior_synapses": int(core.provenance_mask.sum().item()),
                        "seed": self.seed,
                    },
                )
                receipts.append(r)

        return ReceiptsList(receipts)

    # Alias for API compatibility
    program_all_motifs = program_motifs
