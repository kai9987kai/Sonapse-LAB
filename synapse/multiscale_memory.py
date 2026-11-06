"""Multi-timescale working memory, automatic rule-change detection, and tripartite memory for SYNAPSE.

Implements:
1. Multi-timescale working memory combining millisecond sensory traces, second task contexts,
   and slow behavioral state.
2. Sequential change-point detection (Page-Hinkley / CUSUM) detecting environmental rule shifts.
3. Tripartite memory architecture cleanly separating episodic, semantic, and procedural stores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn


class MultiTimescaleWorkingMemory:
    """Three-tier temporal hierarchy fusing fast, medium, and slow contextual representations."""
    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim
        # Decay rates corresponding to:
        # Fast (sensory): decay = 0.5 (~few ticks / millisecond scale)
        # Medium (context): decay = 0.9 (~seconds scale)
        # Slow (behavior): decay = 0.99 (~minutes scale)
        self.fast_trace = np.zeros(feature_dim, dtype=np.float32)
        self.med_context = np.zeros(feature_dim, dtype=np.float32)
        self.slow_state = np.zeros(feature_dim, dtype=np.float32)

    def step(self, current_features: np.ndarray) -> np.ndarray:
        """Updates all three tiers and returns unified fused representation."""
        x = np.asarray(current_features[:self.feature_dim], dtype=np.float32)
        if len(x) < self.feature_dim:
            x = np.pad(x, (0, self.feature_dim - len(x)))
            
        self.fast_trace = 0.5 * self.fast_trace + 0.5 * x
        self.med_context = 0.9 * self.med_context + 0.1 * x
        self.slow_state = 0.99 * self.slow_state + 0.01 * x
        
        # Fused multi-timescale embedding: [Fast(32), Medium(32), Slow(32)] -> 96-d or pooled 32-d
        fused = (self.fast_trace * 0.4 + self.med_context * 0.4 + self.slow_state * 0.2)
        return fused

    def get_tier_states(self) -> Dict[str, np.ndarray]:
        return {
            'fast_sensory': self.fast_trace.copy(),
            'medium_context': self.med_context.copy(),
            'slow_behavior': self.slow_state.copy()
        }


class RuleChangeDetector:
    """Detects shifts in environmental reward contingencies, physics, or transitions using CUSUM."""
    def __init__(self, delta: float = 0.05, threshold: float = 8.0):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.mean_error = 0.0
        self.count = 0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.rule_shift_detected = False
        self.history_errors: List[float] = []

    def update(self, prediction_error: float) -> Tuple[bool, float]:
        """Processes prediction error / surprisal metric.
        
        Returns:
            Tuple (detected_shift, cusum_statistic)
        """
        err = float(prediction_error)
        self.history_errors.append(err)
        self.count += 1
        
        # Update running baseline
        if self.count <= 20:
            self.mean_error += (err - self.mean_error) / self.count
            return False, 0.0
            
        # Two-sided CUSUM test
        s = err - self.mean_error
        self.cusum_pos = max(0.0, self.cusum_pos + s - self.delta)
        self.cusum_neg = max(0.0, self.cusum_neg - s - self.delta)
        
        cusum_stat = max(self.cusum_pos, self.cusum_neg)
        shift_detected = bool(cusum_stat > self.threshold)
        
        if shift_detected:
            self.rule_shift_detected = True
            # Reset CUSUM to resume monitoring in new regime
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self.mean_error = err
            
        return shift_detected, cusum_stat


@dataclass
class EpisodicRecord:
    episode_id: int
    step: int
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    valence: float


@dataclass
class SemanticConcept:
    entity: str
    relation: str
    target: str
    confidence: float


@dataclass
class ProceduralSkill:
    skill_tag: str
    policy_weights: Dict[str, np.ndarray]
    preconditions: str
    expected_duration: int


class TripartiteMemorySystem:
    """Dedicated storage and retrieval across Episodic, Semantic, and Procedural memory."""
    def __init__(self, episodic_capacity: int = 500):
        self.episodic_capacity = episodic_capacity
        self.episodic_store: List[EpisodicRecord] = []
        self.semantic_store: Dict[str, SemanticConcept] = {}
        self.procedural_store: Dict[str, ProceduralSkill] = {}

    def store_episode_step(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray):
        rec = EpisodicRecord(
            episode_id=len(self.episodic_store) // 100,
            step=len(self.episodic_store),
            observation=obs.copy(),
            action=action.copy(),
            reward=float(reward),
            next_observation=next_obs.copy(),
            valence=abs(float(reward))
        )
        self.episodic_store.append(rec)
        if len(self.episodic_store) > self.episodic_capacity:
            self.episodic_store.pop(0)

    def retrieve_similar_episodes(self, query_obs: np.ndarray, top_k: int = 3) -> List[EpisodicRecord]:
        """Retrieves top-k closest past episodes by sensory cosine similarity."""
        if not self.episodic_store:
            return []
        scores = []
        q_norm = np.linalg.norm(query_obs) + 1e-6
        for rec in self.episodic_store:
            r_norm = np.linalg.norm(rec.observation) + 1e-6
            sim = np.dot(query_obs, rec.observation) / (q_norm * r_norm)
            scores.append((float(sim), rec))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scores[:top_k]]

    def add_semantic_fact(self, entity: str, relation: str, target: str, confidence: float = 1.0):
        key = f"{entity}:{relation}:{target}"
        self.semantic_store[key] = SemanticConcept(entity, relation, target, confidence)

    def query_semantic(self, entity: str, relation: Optional[str] = None) -> List[SemanticConcept]:
        results = []
        for fact in self.semantic_store.values():
            if fact.entity == entity:
                if relation is None or fact.relation == relation:
                    results.append(fact)
        return results

    def register_procedural_skill(self, skill_tag: str, weights: Dict[str, np.ndarray], preconditions: str = "always", duration: int = 20):
        self.procedural_store[skill_tag] = ProceduralSkill(skill_tag, weights, preconditions, duration)

    def retrieve_procedural_skill(self, skill_tag: str) -> Optional[ProceduralSkill]:
        return self.procedural_store.get(skill_tag)
