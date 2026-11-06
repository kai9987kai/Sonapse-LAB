"""Object-centric understanding and relational dynamics for SYNAPSE.

Implements Slot Attention to extract discrete entity representations from spatial features,
tracks persistent object identity under movement, and models relational physical interactions
(e.g., predicting how pushing one object impacts neighbouring entities).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ObjectSlot:
    """Represents a discrete entity in the environment."""
    track_id: int
    category: str                   # 'agent', 'box', 'obstacle', 'goal', 'hazard', 'unknown'
    position: np.ndarray            # [x, y, z] or [x, y]
    velocity: np.ndarray            # [vx, vy]
    appearance: np.ndarray          # feature embedding vector
    confidence: float               # detection confidence [0, 1]
    is_pushable: bool = False
    mass: float = 1.0
    history: List[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'track_id': self.track_id,
            'category': self.category,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'confidence': float(self.confidence),
            'is_pushable': self.is_pushable,
            'mass': float(self.mass)
        }


class SlotAttention(nn.Module):
    """Slot Attention module (Locatello et al., 2020) adapted for real-time biological cognition.
    
    Transforms spatial visual feature maps into K discrete permutation-symmetric object slots.
    """
    def __init__(self, num_slots: int = 6, slot_dim: int = 32, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5
        
        # Learnable Gaussian slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs iterative slot attention.
        
        Args:
            inputs: Tensor of shape (B, N, slot_dim) containing spatial tokens
            
        Returns:
            slots: (B, num_slots, slot_dim)
            attn_weights: (B, num_slots, N)
        """
        b, n, d = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)
        
        # Initialize slots from Gaussian prior
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)
            
            # Dot-product attention: (B, num_slots, N)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = F.softmax(dots, dim=1) + self.eps
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)
            
            # Weighted average over input tokens
            updates = torch.einsum('bij,bjd->bid', attn_weights, v)
            
            # GRU update
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots, attn


class RelationalDynamicsPredictor(nn.Module):
    """Predicts physical entity interactions and momentum transfer between object slots."""
    def __init__(self, slot_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.slot_dim = slot_dim
        # Edge function: takes pair of object slots and optional push force
        self.edge_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        # Node update function: aggregates edge messages to predict delta position & delta velocity
        self.node_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # (dx, dy, dvx, dvy)
        )

    def forward(self, slots: torch.Tensor, applied_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predicts delta state for each slot given pairwise relations.
        
        Args:
            slots: (num_slots, slot_dim)
            applied_action: (2,) push/move force on active agent slot
        """
        k, d = slots.shape
        if applied_action is None:
            applied_action = torch.zeros(2, device=slots.device)
            
        # Broadcast all pairs (i, j)
        slot_i = slots.unsqueeze(1).expand(k, k, d)
        slot_j = slots.unsqueeze(0).expand(k, k, d)
        act_expanded = applied_action.view(1, 1, 2).expand(k, k, 2)
        
        pair_features = torch.cat([slot_i, slot_j, act_expanded], dim=-1)
        messages = self.edge_mlp(pair_features) # (k, k, d)
        
        # Zero out self-interaction
        diag_mask = (1.0 - torch.eye(k, device=slots.device)).unsqueeze(-1)
        messages = messages * diag_mask
        
        # Aggregate messages incoming to node i
        aggregated = messages.sum(dim=1) # (k, d)
        
        node_input = torch.cat([slots, aggregated], dim=-1)
        deltas = self.node_mlp(node_input) # (k, 4) -> (dx, dy, dvx, dvy)
        return deltas


class ObjectCentricMemory:
    """High-level object-centric understanding and persistent tracking system."""
    def __init__(self, num_slots: int = 6, slot_dim: int = 32, match_threshold: float = 0.6):
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.match_threshold = match_threshold
        
        self.slot_attention = SlotAttention(num_slots=num_slots, slot_dim=slot_dim)
        self.dynamics = RelationalDynamicsPredictor(slot_dim=slot_dim)
        
        # Attribute readouts from slot vectors
        self.pos_head = nn.Linear(slot_dim, 2)
        self.vel_head = nn.Linear(slot_dim, 2)
        self.affordance_head = nn.Linear(slot_dim, 3) # [pushable_prob, static_prob, goal_prob]
        
        self.active_tracks: Dict[int, ObjectSlot] = {}
        self.next_track_id = 1

    def process_spatial_features(self, spatial_map: torch.Tensor, applied_action: Optional[np.ndarray] = None) -> List[ObjectSlot]:
        """Extracts and tracks object entities from convolutional feature maps.
        
        Args:
            spatial_map: (C, H, W) or (1, C, H, W)
            applied_action: Optional 2D push/velocity vector
            
        Returns:
            List of detected ObjectSlot entities with persistent track IDs.
        """
        if spatial_map.ndim == 3:
            c, h, w = spatial_map.shape
            tokens = spatial_map.reshape(1, c, h * w).permute(0, 2, 1) # (1, N, C)
        elif spatial_map.ndim == 4:
            b, c, h, w = spatial_map.shape
            tokens = spatial_map.reshape(b, c, h * w).permute(0, 2, 1)
        else:
            raise ValueError(f"Invalid spatial_map shape: {spatial_map.shape}")
            
        # Project token dimension to slot_dim if necessary
        if tokens.shape[-1] != self.slot_dim:
            if not hasattr(self, '_token_proj') or self._token_proj.in_features != tokens.shape[-1]:
                self._token_proj = nn.Linear(tokens.shape[-1], self.slot_dim)
            tokens = self._token_proj(tokens)
            
        slots, _ = self.slot_attention(tokens)
        slots_flat = slots.squeeze(0) # (num_slots, slot_dim)
        
        positions = self.pos_head(slots_flat).detach().numpy()
        velocities = self.vel_head(slots_flat).detach().numpy()
        affordances = F.softmax(self.affordance_head(slots_flat), dim=-1).detach().numpy()
        
        new_slots: List[ObjectSlot] = []
        for i in range(self.num_slots):
            pos = positions[i]
            vel = velocities[i]
            app = slots_flat[i].detach().numpy()
            aff = affordances[i]
            
            # Classify category based on affordance scores
            if aff[0] > 0.5:
                category = 'box'
                pushable = True
            elif aff[2] > 0.4:
                category = 'goal'
                pushable = False
            else:
                category = 'obstacle'
                pushable = False
                
            slot = ObjectSlot(
                track_id=-1,
                category=category,
                position=pos,
                velocity=vel,
                appearance=app,
                confidence=float(max(aff)),
                is_pushable=pushable
            )
            new_slots.append(slot)
            
        # Persistent re-identification across timesteps
        self._match_and_update_tracks(new_slots)
        return list(self.active_tracks.values())

    def _match_and_update_tracks(self, candidate_slots: List[ObjectSlot]):
        """Associates new detections with existing tracks using appearance and spatial continuity."""
        if not self.active_tracks:
            for s in candidate_slots:
                s.track_id = self.next_track_id
                s.history.append(s.position.copy())
                self.active_tracks[self.next_track_id] = s
                self.next_track_id += 1
            return
            
        existing_ids = list(self.active_tracks.keys())
        matched_tracks = set()
        
        for cand in candidate_slots:
            best_id = None
            best_score = -1.0
            
            for tid in existing_ids:
                if tid in matched_tracks:
                    continue
                prev = self.active_tracks[tid]
                # Cosine similarity in appearance space
                norm_a = np.linalg.norm(cand.appearance) + 1e-8
                norm_b = np.linalg.norm(prev.appearance) + 1e-8
                app_sim = float(np.dot(cand.appearance, prev.appearance) / (norm_a * norm_b))
                
                # Spatial distance penalty (closer is higher score)
                dist = float(np.linalg.norm(cand.position - (prev.position + prev.velocity)))
                dist_score = 1.0 / (1.0 + dist)
                
                total_score = 0.7 * app_sim + 0.3 * dist_score
                if total_score > best_score and total_score > self.match_threshold:
                    best_score = total_score
                    best_id = tid
                    
            if best_id is not None:
                cand.track_id = best_id
                # Update velocity from position delta
                cand.velocity = cand.position - self.active_tracks[best_id].position
                cand.history = self.active_tracks[best_id].history + [cand.position.copy()]
                self.active_tracks[best_id] = cand
                matched_tracks.add(best_id)
            else:
                # Spawn new track
                cand.track_id = self.next_track_id
                cand.history.append(cand.position.copy())
                self.active_tracks[self.next_track_id] = cand
                self.next_track_id += 1

    def predict_push_effect(self, target_track_id: int, push_force: np.ndarray) -> Dict[str, Any]:
        """Predicts the relational physical outcome when pushing an object.
        
        Demonstrates that pushing a movable box shifts its location and propagates force to neighbours.
        """
        if target_track_id not in self.active_tracks:
            return {'error': 'Track not found'}
            
        slots_list = list(self.active_tracks.values())
        k = len(slots_list)
        app_tensors = torch.tensor(np.array([s.appearance for s in slots_list]), dtype=torch.float32)
        force_tensor = torch.tensor(push_force[:2], dtype=torch.float32)
        
        deltas = self.dynamics(app_tensors, force_tensor).detach().numpy()
        
        predictions = {}
        for i, s in enumerate(slots_list):
            d_pos = deltas[i, :2]
            d_vel = deltas[i, 2:]
            predicted_new_pos = s.position + d_pos
            predictions[s.track_id] = {
                'category': s.category,
                'initial_position': s.position.tolist(),
                'predicted_position': predicted_new_pos.tolist(),
                'delta_pos': d_pos.tolist(),
                'is_target': (s.track_id == target_track_id)
            }
            
        return {
            'target_track_id': target_track_id,
            'push_force': push_force.tolist(),
            'predictions': predictions
        }
