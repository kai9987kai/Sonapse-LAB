"""Uncertainty-aware neural modification and weight modification guard for SYNAPSE.

Assesses candidate weight-space edits before application using:
1. Ensemble disagreement across weight generators
2. Out-of-Distribution (OOD) distance from safe experience buffer
3. Optional measured validation on an inactive cloned module.

Structural approval alone is not evidence of task improvement or a safety proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .stateio import digest


@dataclass
class ModificationProposal:
    """A proposed weight modification patch."""
    target_layer: str
    delta_weights: torch.Tensor
    source_context: np.ndarray
    expected_gain: float


@dataclass
class GuardAssessment:
    """Comprehensive safety and reliability assessment of a proposed weight update."""
    approved: bool
    ensemble_disagreement: float
    ood_distance: float
    validation_performance_delta: Optional[float]
    rejection_reason: Optional[str] = None
    ood_reference_count: int = 0
    target_digest: Optional[str] = None
    delta_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'approved': self.approved,
            'ensemble_disagreement': float(self.ensemble_disagreement),
            'ood_distance': float(self.ood_distance),
            'validation_performance_delta': self.validation_performance_delta,
            'rejection_reason': self.rejection_reason,
            'ood_reference_count': self.ood_reference_count,
            'ood_supported': self.ood_reference_count >= 5,
            'target_digest': self.target_digest,
            'delta_digest': self.delta_digest,
            'scope': 'measured inactive-candidate screening' if self.validation_performance_delta is not None else 'structural screening; task performance unmeasured'
        }


class UncertaintyGuard:
    """Heuristic ensemble screening with optional real task evaluation."""
    def __init__(self, ensemble_size: int = 3, ood_threshold: float = 3.5, disagreement_threshold: float = 0.45):
        self.ensemble_size = ensemble_size
        self.ood_threshold = ood_threshold
        self.disagreement_threshold = disagreement_threshold
        
        # Reference buffer of safe in-distribution activation representations
        self.reference_embeddings: List[np.ndarray] = []
        self.history: List[GuardAssessment] = []

    def register_safe_experience(self, context_embedding: np.ndarray):
        """Adds validated in-distribution state embeddings to the reference baseline."""
        emb = np.asarray(context_embedding, dtype=np.float32).flatten()
        if not len(emb) or not np.isfinite(emb).all():
            raise ValueError('reference context must be finite and nonempty')
        if self.reference_embeddings and emb.shape != self.reference_embeddings[0].shape:
            raise ValueError('reference context shape mismatch')
        self.reference_embeddings.append(emb.copy())
        if len(self.reference_embeddings) > 200:
            self.reference_embeddings.pop(0)

    def compute_ood_score(self, context_embedding: np.ndarray) -> float:
        """Computes Mahalanobis / Euclidean distance to reference safe manifold."""
        context_embedding = np.asarray(context_embedding, dtype=np.float64)
        if context_embedding.ndim != 1 or not len(context_embedding) or not np.isfinite(context_embedding).all():
            raise ValueError('context must be a finite nonempty vector')
        if self.reference_embeddings and context_embedding.shape != self.reference_embeddings[0].shape:
            raise ValueError('context shape mismatch')
        if len(self.reference_embeddings) < 5:
            return 0.0 # Insufficient baseline data
            
        ref = np.array(self.reference_embeddings)
        mean = np.mean(ref, axis=0)
        std = np.std(ref, axis=0) + 1e-4
        
        z = (context_embedding - mean) / std
        return float(np.linalg.norm(z) / np.sqrt(len(context_embedding)))

    def assess_ensemble_proposals(self, proposals: List[ModificationProposal], target_module: nn.Module,
                                  evaluator=None, min_validation_delta=0.0) -> GuardAssessment:
        """Screen structure, then optionally measure a paired candidate score.

        ``evaluator(module)`` returns a finite scalar where higher is better.
        It must supply the same support/validation examples for both cloned
        modules. No fabricated gain is inferred from finite weights.
        """
        def reject(reason):
            assessment = GuardAssessment(False, 0.0, 0.0, None, reason, len(self.reference_embeddings))
            self.history.append(assessment)
            return assessment
        if len(proposals) < 2:
            return reject('At least two ensemble proposals are required')
        if not isinstance(target_module, nn.Module) or not np.isfinite(min_validation_delta):
            raise ValueError('valid target module and finite validation threshold required')
        target = getattr(target_module, 'weight', None)
        if not isinstance(target, torch.Tensor):
            return reject('Target module must expose a weight tensor')
        if len({p.target_layer for p in proposals}) != 1:
            return reject('Proposals must refer to the same target layer')
        if any(not isinstance(p.delta_weights, torch.Tensor) or p.delta_weights.shape != target.shape
               or not bool(torch.isfinite(p.delta_weights).all()) for p in proposals):
            return reject('Non-finite or shape-mismatched proposed weight tensor')
            
        # 1. Compute ensemble variance / disagreement across proposed weight matrices
        weights = [p.delta_weights.detach().to(device='cpu', dtype=torch.float64) for p in proposals]
        stacked = torch.stack(weights, dim=0) # (E, ...)
        var = torch.var(stacked, dim=0, unbiased=False).mean()
        norm = stacked.abs().mean() + 1e-6
        disagreement = float(var / norm)
        
        # 2. Compute OOD distance of the proposed contexts
        try:
            ood_scores = [self.compute_ood_score(p.source_context) for p in proposals]
        except ValueError as exc:
            return reject(str(exc))
        mean_ood = float(np.mean(ood_scores))
        
        # 3. Dry-run validation check: verify that applying average delta does not cause NaN or divergence
        avg_delta = torch.stack([p.delta_weights.detach().to(target) for p in proposals]).mean(dim=0)
        is_finite = bool(torch.isfinite(avg_delta).all()) and np.isfinite(disagreement)
        
        rejection_reason = None
        if not is_finite:
            rejection_reason = "Non-finite values detected in proposed weight tensor"
        elif disagreement > self.disagreement_threshold:
            rejection_reason = f"Ensemble disagreement {disagreement:.3f} exceeds safety threshold {self.disagreement_threshold:.3f}"
        elif mean_ood > self.ood_threshold:
            rejection_reason = f"Context OOD score {mean_ood:.3f} exceeds threshold {self.ood_threshold:.3f}"
            
        performance_delta = None
        if rejection_reason is None and evaluator is not None:
            baseline, candidate = deepcopy(target_module), deepcopy(target_module)
            with torch.no_grad():
                candidate.weight.add_(avg_delta.to(candidate.weight))
                if not bool(torch.isfinite(candidate.weight).all()):
                    return reject('Candidate weights overflow after update')
                before = float(evaluator(baseline))
                after = float(evaluator(candidate))
            if not np.isfinite([before, after]).all():
                return reject('Validation evaluator returned non-finite scores')
            performance_delta = after - before
            if performance_delta < min_validation_delta:
                rejection_reason = 'Measured validation regression or insufficient gain'
        approved = rejection_reason is None
        
        assessment = GuardAssessment(
            approved=approved,
            ensemble_disagreement=disagreement,
            ood_distance=mean_ood,
            validation_performance_delta=performance_delta,
            rejection_reason=rejection_reason,
            ood_reference_count=len(self.reference_embeddings),
            target_digest=digest(target.detach().cpu()),
            delta_digest=digest(avg_delta.detach().cpu())
        )
        self.history.append(assessment)
        return assessment


def apply_safe_modification(target_param: nn.Parameter, assessment: GuardAssessment, avg_delta: torch.Tensor) -> bool:
    """Applies weight modification if and only if approved by the uncertainty guard."""
    if not assessment.approved:
        return False
    with torch.no_grad():
        if not isinstance(avg_delta, torch.Tensor) or avg_delta.shape != target_param.shape or not bool(torch.isfinite(avg_delta).all()):
            raise ValueError('delta must be finite and match target shape')
        delta = avg_delta.to(target_param)
        if assessment.delta_digest is None or digest(delta.detach().cpu()) != assessment.delta_digest:
            raise ValueError('delta differs from the assessed modification')
        if assessment.target_digest is None or digest(target_param.detach().cpu()) != assessment.target_digest:
            raise ValueError('target changed after assessment')
        proposed = target_param + delta
        if not bool(torch.isfinite(proposed).all()):
            raise ValueError('modification would produce non-finite weights')
        target_param.copy_(proposed)
    return True
