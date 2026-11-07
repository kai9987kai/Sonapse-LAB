"""Bounded dormant-expert lifecycle with heuristic recruitment and EMA merging.

Grounding:
Fulfills the 5 November 2026 Developmental Connectome specification:
    "A candidate graft progresses through explicit stages: dormant identity,
     fitted initialization, gradual activation, contribution test,
     consolidation, and retirement."
States track recruitment and retirement. Orthogonal initialization is unfitted;
output norm is a usage heuristic, not held-out causal contribution. Consolidation
interpolates weights; it is not gradient-based knowledge distillation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DormantState(str, Enum):
    DORMANT = "dormant"
    INITIALIZING = "initializing"
    ACTIVATING = "activating"
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    RETIRED = "retired"


@dataclass
class DormantSlotRecord:
    slot_id: int
    state: str = DormantState.DORMANT.value
    age: int = 0
    activation_gain: float = 0.0
    usage_count: int = 0
    contribution_score: float = 0.0
    consolidation_progress: float = 0.0
    novelty_triggers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'slot_id': self.slot_id,
            'state': self.state,
            'age': self.age,
            'activation_gain': float(self.activation_gain),
            'usage_count': self.usage_count,
            'contribution_score': float(self.contribution_score),
            'consolidation_progress': float(self.consolidation_progress),
            'novelty_triggers': self.novelty_triggers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DormantSlotRecord:
        return cls(
            slot_id=int(data['slot_id']),
            state=str(data.get('state', DormantState.DORMANT.value)),
            age=int(data.get('age', 0)),
            activation_gain=float(data.get('activation_gain', 0.0)),
            usage_count=int(data.get('usage_count', 0)),
            contribution_score=float(data.get('contribution_score', 0.0)),
            consolidation_progress=float(data.get('consolidation_progress', 0.0)),
            novelty_triggers=int(data.get('novelty_triggers', 0)),
        )


class DormantLifecycleManager:
    """Manages explicit recruitment, activation, evaluation, consolidation and retirement."""

    def __init__(self, num_slots: int = 2, feature_dim: int = 32, seed: int = 7):
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device='cpu').manual_seed(seed)

        # PyTorch linear modules for each dormant expert slot
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.modules = nn.ModuleList([
                nn.Linear(feature_dim, feature_dim, bias=False)
                for _ in range(num_slots)
            ])
        for mod in self.modules:
            nn.init.orthogonal_(mod.weight, gain=0.01, generator=self.torch_rng)

        # Slot metadata records
        self.records: List[DormantSlotRecord] = [
            DormantSlotRecord(slot_id=i) for i in range(num_slots)
        ]
        self.consolidations_completed = 0
        self.step_counter = 0

    def step_novelty(
        self,
        novelty: float,
        preferred_slot: int,
        novelty_threshold: float = 0.02,
        learn: bool = True,
    ) -> Tuple[bool, int, float]:
        """Observes graph novelty and advances lifecycle of recruited expert slot.

        Returns:
            (recruited: bool, slot_idx: int, effective_gain: float)
        """
        slot_idx = int(preferred_slot) % self.num_slots
        rec = self.records[slot_idx]
        if not learn:
            return False, slot_idx, float(rec.activation_gain)
        self.step_counter += 1
        rec.age += 1

        recruited = bool(novelty > novelty_threshold)
        if recruited:
            rec.novelty_triggers += 1
            rec.usage_count += 1

            if rec.state == DormantState.DORMANT.value:
                # Transition DORMANT -> INITIALIZING
                rec.state = DormantState.INITIALIZING.value
                rec.activation_gain = 0.1
            elif rec.state == DormantState.INITIALIZING.value:
                # Ramping activation
                rec.activation_gain = min(1.0, rec.activation_gain + 0.15)
                if rec.activation_gain >= 0.5:
                    rec.state = DormantState.ACTIVATING.value
            elif rec.state == DormantState.ACTIVATING.value:
                rec.activation_gain = min(1.0, rec.activation_gain + 0.15)
                if rec.activation_gain >= 1.0:
                    rec.state = DormantState.ACTIVE.value
            elif rec.state == DormantState.RETIRED.value:
                # Recycle retired slot
                self.recycle_slot(slot_idx)
                rec.state = DormantState.INITIALIZING.value
                rec.activation_gain = 0.1

        gain = rec.activation_gain if rec.state in {
            DormantState.INITIALIZING.value,
            DormantState.ACTIVATING.value,
            DormantState.ACTIVE.value,
            DormantState.CONSOLIDATING.value
        } else 0.0

        return recruited, slot_idx, gain

    def evaluate_contribution(
        self,
        pooled: torch.Tensor,
        slot_idx: int
    ) -> float:
        """Tracks output magnitude; this does not measure task contribution."""
        if slot_idx < 0 or slot_idx >= self.num_slots:
            return 0.0
        rec = self.records[slot_idx]
        with torch.no_grad():
            slot_out = self.modules[slot_idx](pooled)
            out_norm = float(slot_out.norm().item())
            # Contribution EMA
            rec.contribution_score = 0.95 * rec.contribution_score + 0.05 * out_norm
            return rec.contribution_score

    def step_consolidation(
        self,
        native_linear: nn.Linear,
        recruited_slot: Optional[int] = None,
        rate: float = 0.005
    ) -> Dict[str, Any]:
        """Transfers knowledge from mature active experts into native representation."""
        events = []
        for slot_idx, rec in enumerate(self.records):
            if rec.state == DormantState.ACTIVE.value:
                # If active slot has established usage and positive contribution
                if (rec.usage_count >= 8 or recruited_slot == slot_idx) and rec.contribution_score > 0.0001:
                    rec.state = DormantState.CONSOLIDATING.value
                    events.append({'slot_id': slot_idx, 'event': 'consolidation_started'})

            if rec.state == DormantState.CONSOLIDATING.value:
                with torch.no_grad():
                    # Heuristic EMA merge, with no outcome or retention claim.
                    native_linear.weight.mul_(1.0 - rate).add_(self.modules[slot_idx].weight * rate)
                    rec.consolidation_progress = min(1.0, rec.consolidation_progress + rate * 4.0)

                if rec.consolidation_progress >= 1.0:
                    rec.state = DormantState.RETIRED.value
                    self.consolidations_completed += 1
                    events.append({'slot_id': slot_idx, 'event': 'slot_retired'})
                    # Immediately recycle slot to DORMANT for next exploration
                    self.recycle_slot(slot_idx)

        return {
            'consolidations_completed': self.consolidations_completed,
            'events': events
        }

    def recycle_slot(self, slot_idx: int):
        """Cleans and re-initializes a retired slot back to DORMANT."""
        rec = self.records[slot_idx]
        with torch.no_grad():
            nn.init.orthogonal_(self.modules[slot_idx].weight, gain=0.01, generator=self.torch_rng)
        rec.state = DormantState.DORMANT.value
        rec.age = 0
        rec.activation_gain = 0.0
        rec.usage_count = 0
        rec.contribution_score = 0.0
        rec.consolidation_progress = 0.0

    def export_state(self) -> Dict[str, Any]:
        return {
            'schema': 1,
            'num_slots': self.num_slots,
            'feature_dim': self.feature_dim,
            'consolidations_completed': self.consolidations_completed,
            'step_counter': self.step_counter,
            'rng': deepcopy(self.rng.bit_generator.state),
            'torch_rng': self.torch_rng.get_state().clone(),
            'records': [r.to_dict() for r in self.records],
            'weights': [m.weight.detach().cpu().numpy() for m in self.modules],
        }

    def import_state(self, state: Dict[str, Any]):
        if not isinstance(state, dict) or state.get('schema') != 1:
            raise ValueError('invalid DormantLifecycle state schema')
        self.num_slots = int(state['num_slots'])
        self.feature_dim = int(state['feature_dim'])
        self.consolidations_completed = int(state.get('consolidations_completed', 0))
        self.step_counter = int(state.get('step_counter', 0))
        if 'rng' in state:
            self.rng.bit_generator.state = deepcopy(state['rng'])
        if 'torch_rng' in state:
            self.torch_rng.set_state(state['torch_rng'].clone())
        self.records = [DormantSlotRecord.from_dict(r) for r in state['records']]
        for i, w in enumerate(state['weights']):
            with torch.no_grad():
                self.modules[i].weight.copy_(torch.from_numpy(np.asarray(w, dtype=np.float32)))

    def summary(self) -> Dict[str, Any]:
        return {
            'num_slots': self.num_slots,
            'consolidations_completed': self.consolidations_completed,
            'initialization': 'unfitted_orthogonal',
            'contribution_metric': 'output_norm_ema',
            'consolidation_method': 'weight_ema',
            'outcome_contribution_verified': False,
            'slots': [r.to_dict() for r in self.records],
        }
