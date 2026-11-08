"""Pretrained Donor Grafts and Latent Alignment Subsystem.

Provides:
- Multi-role frozen donor model adapter (PoseidonHubDonorAdapter, DonorRole).
- Zero-gain latent space alignment bridge and factory (PretrainedGraftBridge, PretrainedGraftBridgeFactory).
"""

from synapse.donor_grafts.poseidon_hub import (
    DonorRole,
    PoseidonHubDonorAdapter,
)
from synapse.donor_grafts.latent_alignment import (
    PretrainedGraftBridge,
    PretrainedGraftBridgeFactory,
)

__all__ = [
    "DonorRole",
    "PoseidonHubDonorAdapter",
    "PretrainedGraftBridge",
    "PretrainedGraftBridgeFactory",
]
