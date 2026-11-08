"""Modulatory Compositor.

Authoritative bridge uniting NeuromodulatoryEngine, HomeostaticEngine,
SyntheticGlandBank, and SyntheticReceptorBank.

Governed by the non-negotiable zero-gain identity contract:
When enabled=False or alpha_endocrine=0.0:
composite_modulations(...) == legacy_neuromodulator_state bit-for-bit (Delta = 0.0).

Non-competing authority:
Does NOT mutate legacy internal engine state directly. Composites output vectors
using controlled bounded perturbations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from synapse.neuromodulation import NeuromodulatorState
from synapse.neuroendocrine.synthetic_glands import GLAND_NAMES
from synapse.neuroendocrine.receptor_fields import SyntheticReceptorBank, TARGET_PROPERTIES
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class ModulatoryCompositor(nn.Module):
    """Authoritative non-competing compositor for neurochemical and endocrine signals."""

    def __init__(
        self,
        enabled: bool = False,
        alpha_endocrine: float = 0.35,
        receptor_bank: Optional[SyntheticReceptorBank] = None,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.enabled = enabled
        self.alpha_endocrine = alpha_endocrine
        self.receptors = receptor_bank if receptor_bank is not None else SyntheticReceptorBank(seed=seed)

        # Telemetry storage
        self.last_legacy_dict: Dict[str, float] = {}
        self.last_composite_dict: Dict[str, float] = {}
        self.last_target_gains: Dict[str, float] = {prop: 1.0 for prop in TARGET_PROPERTIES}

    def composite(
        self,
        legacy_state: Union[NeuromodulatorState, Dict[str, float], np.ndarray],
        gland_deltas: Optional[torch.Tensor] = None,
        gland_levels: Optional[torch.Tensor] = None,
        enabled_override: Optional[bool] = None,
    ) -> Tuple[NeuromodulatorState, Dict[str, float]]:
        """Composites legacy neuromodulatory state with endocrine signals.

        Returns:
            composite_state: NeuromodulatorState with blended chemical concentrations.
            target_gains: Dict[str, float] with receptor-derived multipliers (excitability, plasticity, etc.).
        """
        is_enabled = enabled_override if enabled_override is not None else self.enabled

        # Extract legacy values
        if isinstance(legacy_state, NeuromodulatorState):
            da_leg = float(legacy_state.dopamine)
            sht_leg = float(legacy_state.serotonin)
            oa_leg = float(legacy_state.octopamine)
            ach_leg = float(legacy_state.acetylcholine)
        elif isinstance(legacy_state, dict):
            da_leg = float(legacy_state.get("dopamine", 0.0))
            sht_leg = float(legacy_state.get("serotonin", 0.2))
            oa_leg = float(legacy_state.get("octopamine", 0.2))
            ach_leg = float(legacy_state.get("acetylcholine", 0.5))
        elif isinstance(legacy_state, (np.ndarray, torch.Tensor)):
            arr = np.asarray(legacy_state).flatten()
            da_leg = float(arr[0])
            sht_leg = float(arr[1])
            oa_leg = float(arr[2])
            ach_leg = float(arr[3])
        else:
            da_leg, sht_leg, oa_leg, ach_leg = 0.0, 0.2, 0.2, 0.5

        self.last_legacy_dict = {
            "dopamine": da_leg,
            "serotonin": sht_leg,
            "octopamine": oa_leg,
            "acetylcholine": ach_leg,
        }

        # 1. Zero-gain identity contract: if disabled or no deltas, return legacy exact
        if not is_enabled or gland_deltas is None or self.alpha_endocrine <= 0.0:
            neutral_gains = {prop: 1.0 for prop in TARGET_PROPERTIES}
            self.last_composite_dict = dict(self.last_legacy_dict)
            self.last_target_gains = neutral_gains
            return (
                NeuromodulatorState(
                    dopamine=da_leg,
                    serotonin=sht_leg,
                    octopamine=oa_leg,
                    acetylcholine=ach_leg,
                ),
                neutral_gains,
            )

        # 2. Endocrine perturbation
        alpha = float(self.alpha_endocrine)
        g_deltas = gland_deltas.detach().cpu().flatten()

        idx = {name: i for i, name in enumerate(GLAND_NAMES)}
        da_delta = float(g_deltas[idx["dopamine"]].item()) if "dopamine" in idx else 0.0
        sht_delta = float(g_deltas[idx["serotonin"]].item()) if "serotonin" in idx else 0.0
        oa_delta = float(g_deltas[idx["octopamine"]].item()) if "octopamine" in idx else 0.0
        ach_delta = float(g_deltas[idx["acetylcholine"]].item()) if "acetylcholine" in idx else 0.0
        adren_delta = float(g_deltas[idx["adrenaline"]].item()) if "adrenaline" in idx else 0.0
        norad_delta = float(g_deltas[idx["noradrenaline"]].item()) if "noradrenaline" in idx else 0.0

        # Blended neuromodulators with biological bounds
        # Octopamine in insects parallels noradrenaline/adrenaline in vertebrates
        oa_combined_delta = oa_delta + 0.4 * adren_delta + 0.4 * norad_delta

        da_comp = float(np.clip(da_leg + alpha * da_delta, -1.0, 1.0))
        sht_comp = float(np.clip(sht_leg + alpha * sht_delta, 0.0, 1.0))
        oa_comp = float(np.clip(oa_leg + alpha * oa_combined_delta, 0.0, 1.0))
        ach_comp = float(np.clip(ach_leg + alpha * ach_delta, 0.0, 1.0))

        composite_state = NeuromodulatorState(
            dopamine=da_comp,
            serotonin=sht_comp,
            octopamine=oa_comp,
            acetylcholine=ach_comp,
        )

        # 3. Compute recipient target gains from receptor bank
        target_gains = self.receptors.compute_gains(g_deltas)

        self.last_composite_dict = composite_state.to_dict()
        self.last_target_gains = target_gains

        return composite_state, target_gains

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry comparing legacy and composite modulations."""
        return {
            "enabled": self.enabled,
            "alpha_endocrine": self.alpha_endocrine,
            "legacy": self.last_legacy_dict,
            "composite": self.last_composite_dict,
            "target_gains": self.last_target_gains,
        }
