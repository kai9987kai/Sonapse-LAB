"""Generative Dream Replay Engine.

Synthesizes counterfactual rollouts using world models or recurrent latent dynamics
during DREAM_GENERATION sleep phases.

SCIENTIFIC HONESTY INVARIANT:
All generated dream episodes MUST be labeled as synthetic dreams
(is_synthetic_dream=True, EvidenceKind.SYNTHETIC_SIMULATION) and strictly segregated
from authentic episodic ground-truth memory to prevent confabulation and false memories.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class DreamEpisode:
    dream_id: str
    timestamp: float
    duration_steps: int
    latent_trajectory: List[List[float]]
    affective_trajectory: List[Dict[str, float]]
    counterfactual_theme: str
    is_synthetic_dream: bool = True
    evidence_kind: EvidenceKind = EvidenceKind.SYNTHETIC_SIMULATION
    receipt: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "timestamp": self.timestamp,
            "duration_steps": self.duration_steps,
            "latent_trajectory": self.latent_trajectory,
            "affective_trajectory": self.affective_trajectory,
            "counterfactual_theme": self.counterfactual_theme,
            "is_synthetic_dream": self.is_synthetic_dream,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
            "receipt": self.receipt,
        }


class DreamReplayEngine(nn.Module):
    """Generates synthetic counterfactual rollout trajectories during sleep."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Generative recurrent dream transition net
        self.dream_cell = nn.GRUCell(latent_dim, hidden_dim)
        self.proj_next_latent = nn.Linear(hidden_dim, latent_dim)
        self.proj_affective = nn.Linear(hidden_dim, 4)  # [valence, arousal, fear, curiosity]

        # Recorded dream history (strictly segregated)
        self.dream_history: List[DreamEpisode] = []
        self._dream_counter = 0

    def generate_dream(
        self,
        seed_latent: Optional[torch.Tensor] = None,
        theme: str = "counterfactual_exploration",
        steps: int = 20,
    ) -> DreamEpisode:
        """Generates a synthetic counterfactual rollout trajectory."""
        self._dream_counter += 1
        device = self.proj_next_latent.weight.device

        if seed_latent is None:
            z_t = torch.randn(1, self.latent_dim, device=device) * 0.5
        else:
            z_t = seed_latent.to(device).view(1, self.latent_dim)

        h_t = torch.zeros(1, self.hidden_dim, device=device)
        latent_traj: List[List[float]] = []
        affective_traj: List[Dict[str, float]] = []

        for s in range(steps):
            # Recurrent dream step
            h_t = self.dream_cell(z_t, h_t)
            # Add stochastic exploration noise in dream space
            noise = torch.randn_like(z_t) * 0.15
            z_t = self.proj_next_latent(h_t) + noise

            # Affective appraisal of dream step
            aff = torch.sigmoid(self.proj_affective(h_t)).squeeze(0)

            latent_traj.append(z_t.squeeze(0).tolist())
            affective_traj.append({
                "valence": float(aff[0].item()),
                "arousal": float(aff[1].item()),
                "fear": float(aff[2].item()),
                "curiosity": float(aff[3].item()),
            })

        receipt = EvidenceReceipt(
            item_id=f"dream_episode_{self._dream_counter}",
            kind=EvidenceKind.SYNTHETIC_SIMULATION,
            source="DreamReplayEngine",
            sha256_digest=compute_tensor_digest(latent_traj),
            metrics={
                "steps": steps,
                "theme": theme,
                "mean_arousal": float(np.mean([a["arousal"] for a in affective_traj])),
            },
        )

        episode = DreamEpisode(
            dream_id=f"dream_{int(time.time()*1000)}_{self._dream_counter}",
            timestamp=time.time(),
            duration_steps=steps,
            latent_trajectory=latent_traj,
            affective_trajectory=affective_traj,
            counterfactual_theme=theme,
            is_synthetic_dream=True,
            evidence_kind=EvidenceKind.SYNTHETIC_SIMULATION,
            receipt=receipt.to_dict(),
        )

        self.dream_history.append(episode)
        if len(self.dream_history) > 50:
            self.dream_history.pop(0)

        return episode

    def get_dream_history(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.dream_history]

    def clear(self) -> None:
        self.dream_history.clear()
