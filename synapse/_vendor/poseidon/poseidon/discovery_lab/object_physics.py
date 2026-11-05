"""Project PROMETHEUS Discovery Lab: Physical Object Properties & Buoyancy Engine.

Defines rigid-body physical properties, fluid-solid interactions, density-based
Archimedean buoyancy calculations, structural connectivity for multi-part assemblages
(e.g., rafts), and load-bearing support capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple


# Standard physical constants in Discovery Lab units (SI Metric derived)
GRAVITY = 9.81  # m/s^2
FLUID_DENSITY_WATER = 1.00  # g/cm^3 or 1000 kg/m^3 (relative = 1.0)
FLUID_DENSITY_SALINE = 1.03
AIR_DENSITY = 0.0012
AGENT_MASS = 70.0  # kg / standard mass units


class MaterialType:
    GRANITE_STONE = "granite_stone"
    BASALT_STONE = "basalt_stone"
    PINE_LOG = "pine_log"
    OAK_LOG = "oak_log"
    BALSA_LOG = "balsa_log"
    VINE_CORD = "vine_cord"
    ENZYMATIC_RESIN = "enzymatic_resin"
    REED_BUNDLE = "reed_bundle"


# Material density in relative g/cm^3 (water = 1.0)
MATERIAL_DENSITIES: Dict[str, float] = {
    MaterialType.GRANITE_STONE: 2.70,   # Sinks rapidly
    MaterialType.BASALT_STONE: 2.90,    # Sinks rapidly
    MaterialType.PINE_LOG: 0.52,        # Floats well (52% submerged)
    MaterialType.OAK_LOG: 0.75,         # Floats (75% submerged)
    MaterialType.BALSA_LOG: 0.16,       # Super-buoyant (16% submerged)
    MaterialType.VINE_CORD: 0.90,       # Near-neutral
    MaterialType.ENZYMATIC_RESIN: 1.08, # Dense bio-crosslinked polymer
    MaterialType.REED_BUNDLE: 0.30,     # Highly buoyant
}


@dataclass
class PhysicalObject:
    """Rigid or composite physical entity in the River World."""
    object_id: str
    name: str
    material: str
    position: List[float]  # [x, y, z]
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # [dx, dy, dz] in meters
    orientation_yaw: float = 0.0  # radians
    is_static: bool = False
    is_held: bool = False
    holder_id: Optional[str] = None
    connected_object_ids: List[str] = field(default_factory=list)
    bond_strength: float = 1.0  # 0.0 to 1.0 (reinforced by enzymatic resin)
    submerged_fraction: float = 0.0

    @property
    def volume(self) -> float:
        """Volume in cubic meters (dx * dy * dz)."""
        return max(0.001, float(self.dimensions[0] * self.dimensions[1] * self.dimensions[2]))

    @property
    def density(self) -> float:
        """Relative material density (water = 1.0)."""
        return MATERIAL_DENSITIES.get(self.material, 1.0)

    @property
    def mass(self) -> float:
        """Mass in arbitrary mass units (volume * density * 100)."""
        return float(self.volume * self.density * 100.0)

    @property
    def is_buoyant(self) -> bool:
        """True if material density is strictly less than water density (1.0)."""
        return self.density < FLUID_DENSITY_WATER

    @property
    def net_buoyant_force(self) -> float:
        """Archimedean buoyant force minus gravitational weight when fully submerged."""
        # F_net = (rho_fluid - rho_obj) * V * g
        return float((FLUID_DENSITY_WATER - self.density) * self.volume * GRAVITY * 100.0)

    @property
    def max_support_payload(self) -> float:
        """Maximum extra payload mass (e.g. agent) this object can support while remaining afloat."""
        if not self.is_buoyant:
            return 0.0
        # Displaced fluid mass minus own mass
        displaced_fluid_mass = self.volume * FLUID_DENSITY_WATER * 100.0
        return max(0.0, displaced_fluid_mass - self.mass)

    def compute_assemblage_buoyancy(self, all_objects: Dict[str, PhysicalObject]) -> Tuple[float, float, bool]:
        """Calculates total mass, total buoyant force, and floatability for connected cluster."""
        cluster_ids = set(self.connected_object_ids) | {self.object_id}
        total_mass = sum(all_objects[oid].mass for oid in cluster_ids if oid in all_objects)
        total_displaced_mass = sum(
            all_objects[oid].volume * FLUID_DENSITY_WATER * 100.0
            for oid in cluster_ids if oid in all_objects
        )
        total_support_payload = max(0.0, total_displaced_mass - total_mass)
        can_float = total_displaced_mass > total_mass
        return total_mass, total_support_payload, can_float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "material": self.material,
            "position": [round(float(p), 3) for p in self.position],
            "velocity": [round(float(v), 3) for v in self.velocity],
            "dimensions": [round(float(d), 3) for d in self.dimensions],
            "orientation_yaw": round(float(self.orientation_yaw), 3),
            "mass": round(float(self.mass), 2),
            "density": round(float(self.density), 2),
            "is_buoyant": self.is_buoyant,
            "net_buoyant_force": round(float(self.net_buoyant_force), 2),
            "max_support_payload": round(float(self.max_support_payload), 2),
            "is_held": self.is_held,
            "connected_object_ids": list(self.connected_object_ids),
            "bond_strength": round(float(self.bond_strength), 3),
            "submerged_fraction": round(float(self.submerged_fraction), 3),
        }
