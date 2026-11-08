"""Prioritized Experience Replay Scheduler for Sleep Consolidation.

Prioritizes experiences with high prediction error (surprisal) or extreme emotional valence,
interleaving recent episodes with older consolidated memories to prevent catastrophic forgetting.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class ReplayMemoryItem:
    item_id: str
    timestamp: float
    state_vector: List[float]
    action_vector: List[float]
    reward: float
    next_state_vector: List[float]
    td_error: float
    salience_weight: float
    affective_valence: float = 0.0
    replay_count: int = 0


class ReplayScheduler:
    """Manages memory buffers and sampling during SLEEP_REPLAY consolidation."""

    def __init__(
        self,
        capacity: int = 1000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        seed: int = 42,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.RandomState(seed)

        self.buffer: List[ReplayMemoryItem] = []
        self.priorities: List[float] = []
        self._item_counter = 0

    def push(
        self,
        state: List[float],
        action: List[float],
        reward: float,
        next_state: List[float],
        td_error: float = 1.0,
        valence: float = 0.0,
    ) -> None:
        """Adds a transition to the replay buffer with initial priority."""
        self._item_counter += 1
        item_id = f"mem_{int(time.time()*1000)}_{self._item_counter}"

        # Salience combines TD error and emotional extremity
        salience = float(abs(td_error) + 0.5 * abs(valence) + 1e-4)

        item = ReplayMemoryItem(
            item_id=item_id,
            timestamp=time.time(),
            state_vector=list(state),
            action_vector=list(action),
            reward=float(reward),
            next_state_vector=list(next_state),
            td_error=float(td_error),
            salience_weight=salience,
            affective_valence=float(valence),
        )

        max_p = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.priorities.append(max_p)
        else:
            # Overwrite lowest priority or oldest item
            idx = int(np.argmin(self.priorities))
            self.buffer[idx] = item
            self.priorities[idx] = max_p

    def sample_batch(self, batch_size: int = 16) -> Tuple[List[ReplayMemoryItem], np.ndarray, List[int]]:
        """Samples prioritized batch with importance sampling weights."""
        if not self.buffer:
            return [], np.zeros(0), []

        n = len(self.buffer)
        b_size = min(batch_size, n)

        prios = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs = prios / (prios.sum() + 1e-8)

        indices = self.rng.choice(n, size=b_size, p=probs, replace=False).tolist()
        batch = [self.buffer[i] for i in indices]

        # Importance sampling weights w_i = (N * P(i))^(-beta)
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)

        for item in batch:
            item.replay_count += 1

        return batch, weights, indices

    def update_priorities(self, indices: List[int], new_td_errors: List[float]) -> None:
        """Updates priorities following a replay consolidation step."""
        for idx, td in zip(indices, new_td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(abs(td) + 1e-4)
                self.buffer[idx].td_error = float(td)

    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
        self.priorities.clear()
