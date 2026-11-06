"""Trainable visual perception encoder and multimodal fusion for SYNAPSE.

Consumes genuine 2D/3D RGB camera frames (C, H, W) or (B, C, H, W),
extracts spatial/object visual latents, and fuses them with proprioception
and connectome activity.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VisualFeatureExtractor(nn.Module):
    """Convolutional feature extractor with spatial adaptive pooling.
    
    Accepts arbitrary image dimensions (e.g. 3x64x64 or 3x32x32) and produces
    both a dense spatial feature map and a compact visual embedding.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2) # -> H/2, W/2
        self.ln1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # -> H/4, W/4
        self.ln2 = nn.GroupNorm(4, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # -> H/8, W/8
        self.ln3 = nn.GroupNorm(8, 64)
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(64 * 4 * 4, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        
        # Auxiliary object detection/localization head: predicts normalized (x, y, r, g, b, scale)
        self.object_probe = nn.Linear(latent_dim, 6)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Tuple of:
                latent: (B, latent_dim) compact representation
                spatial_map: (B, 64, 4, 4) spatial feature grid
        """
        single = (x.ndim == 3)
        if single:
            x = x.unsqueeze(0)
            
        h = F.relu(self.ln1(self.conv1(x)))
        h = F.relu(self.ln2(self.conv2(h)))
        h = F.relu(self.ln3(self.conv3(h)))
        
        spatial_map = self.spatial_pool(h)
        flat = spatial_map.flatten(start_dim=1)
        latent = self.norm(F.relu(self.proj(flat)))
        
        if single:
            return latent.squeeze(0), spatial_map.squeeze(0)
        return latent, spatial_map


class VisualProprioceptiveFusion(nn.Module):
    """Multimodal fusion layer combining visual latents, proprioception, and connectome state."""
    def __init__(self, visual_dim: int = 32, proprio_dim: int = 155, connectome_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim
        self.connectome_dim = connectome_dim
        self.out_dim = out_dim
        
        self.vis_proj = nn.Linear(visual_dim, out_dim)
        self.proprio_proj = nn.Linear(proprio_dim, out_dim)
        self.conn_proj = nn.Linear(connectome_dim, out_dim)
        
        # Cross-modal gating: allows visual attention to dynamically modulate proprioception
        self.gate = nn.Linear(visual_dim + proprio_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, visual_latent: torch.Tensor, proprio: torch.Tensor, connectome: torch.Tensor) -> torch.Tensor:
        """Fuses modalities into unified multimodal representation."""
        v = F.relu(self.vis_proj(visual_latent))
        p = F.relu(self.proprio_proj(proprio))
        c = F.relu(self.conn_proj(connectome))
        
        # Gating signal
        vp_cat = torch.cat([visual_latent, proprio], dim=-1)
        alpha = torch.sigmoid(self.gate(vp_cat))
        
        fused = alpha * v + (1.0 - alpha) * p + 0.2 * c
        return self.layer_norm(fused)


class TrainableVisualEncoder(nn.Module):
    """Complete visual encoder module for SYNAPSE.
    
    Includes feature extractor, spatial patch representation, object prediction head,
    and multimodal integration.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 32, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.extractor = VisualFeatureExtractor(in_channels, latent_dim)
        self.fusion = VisualProprioceptiveFusion(visual_dim=latent_dim, out_dim=latent_dim)
        self.view_invariant_head = nn.Linear(latent_dim, 16)
        
    def encode_frame(self, frame: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Encodes a single RGB camera frame into visual features.
        
        Args:
            frame: numpy array or torch Tensor of shape (C, H, W) or (H, W, C) in [0, 1] or [0, 255].
            
        Returns:
            Dictionary containing:
                latent: (latent_dim,) torch tensor
                spatial_map: (64, 4, 4) spatial tensor
                object_prediction: (6,) normalized entity vector (x, y, r, g, b, size)
                invariant_code: (16,) view-invariant descriptor
        """
        if isinstance(frame, np.ndarray):
            t = torch.from_numpy(frame.astype(np.float32))
        else:
            t = frame.float()
            
        # Standardize to (C, H, W)
        if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        if t.shape[0] == 4:
            t = t[:3]
        elif t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] != 3:
            raise ValueError(f"Expected 3 color channels, got {t.shape[0]}")
            
        # Normalize to [0, 1] if values exceed 1.0
        if float(t.max()) > 1.0:
            t = t / 255.0
            
        latent, spatial = self.extractor(t)
        obj_pred = self.extractor.object_probe(latent)
        inv_code = self.view_invariant_head(latent)
        
        return {
            'latent': latent,
            'spatial_map': spatial,
            'object_prediction': obj_pred,
            'invariant_code': inv_code,
            'mean_brightness': float(t.mean()),
            'contrast': float(t.std())
        }

    def fuse(self, visual_latent: torch.Tensor, proprio: torch.Tensor, connectome: torch.Tensor) -> torch.Tensor:
        """Fuses visual latent with proprioception and connectome state."""
        return self.fusion(visual_latent, proprio, connectome)


def evaluate_viewpoint_generalization(encoder: TrainableVisualEncoder, base_frame: np.ndarray) -> Dict[str, float]:
    """Evaluates encoder robustness and feature consistency across simulated viewpoint/lighting shifts."""
    res1 = encoder.encode_frame(base_frame)
    
    # 1. Viewpoint shift (translation / crop)
    shifted = np.roll(base_frame, shift=4, axis=1)
    res_shifted = encoder.encode_frame(shifted)
    
    # 2. Lighting shift (illumination scaling)
    lit = np.clip(base_frame * 1.3 + 0.05, 0.0, 1.0)
    res_lit = encoder.encode_frame(lit)
    
    # Cosine similarities
    sim_shift = float(F.cosine_similarity(res1['invariant_code'].detach().unsqueeze(0), res_shifted['invariant_code'].detach().unsqueeze(0)))
    sim_lit = float(F.cosine_similarity(res1['invariant_code'].detach().unsqueeze(0), res_lit['invariant_code'].detach().unsqueeze(0)))
    
    return {
        'shift_invariance': sim_shift,
        'lighting_invariance': sim_lit,
        'feature_stability': float((sim_shift + sim_lit) / 2.0)
    }
