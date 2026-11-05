"""Project PROMETHEUS: Autonomous Cognitive Discovery Engine & Discovery Lab.

A separate, opt-in subsystem for exploring procedurally generated 3D environments,
discovering how objects work, learning from interventions, acquiring reusable skills,
synthesizing multimodal catalytic enzymes, and developing new solutions when existing strategies fail.
"""
from __future__ import annotations

from .object_physics import PhysicalObject, MaterialType, MATERIAL_DENSITIES, AGENT_MASS
from .manipulation_actions import ManipulationActionType, ManipulationCommand, ACTION_NAMES
from .river_world import RiverWorld, AgentState
from .observation_encoder import DiscoveryObservationEncoder
from .curiosity import DiscoveryCuriosityEngine, CuriosityDecomposition
from .learned_dynamics import ActionConditionedDynamicsModel
from .causal_experiments import DiscoveryCausalEngine, CausalHypothesis
from .skill_library import DiscoverySkillLibrary, DiscoveredSkill, SkillValidationStats
from .skill_composer import DiscoverySkillComposer, CompositionPlan, StrategyType
from .enzyme_generator import MultimodalEnzymeGenerator, SynthesizedEnzyme, CatalyticActiveSite
from .curriculum import DiscoveryCurriculumRunner, CurriculumPhaseResult
from .evaluator import DiscoveryEvaluator
from .biomechanics import (
    BiomechanicalGaitModel,
    NavalStabilityModel,
    StrideAssessment,
    NavalStabilityAssessment,
)

__all__ = [
    "PhysicalObject",
    "MaterialType",
    "MATERIAL_DENSITIES",
    "AGENT_MASS",
    "ManipulationActionType",
    "ManipulationCommand",
    "ACTION_NAMES",
    "RiverWorld",
    "AgentState",
    "DiscoveryObservationEncoder",
    "DiscoveryCuriosityEngine",
    "CuriosityDecomposition",
    "ActionConditionedDynamicsModel",
    "DiscoveryCausalEngine",
    "CausalHypothesis",
    "DiscoverySkillLibrary",
    "DiscoveredSkill",
    "SkillValidationStats",
    "DiscoverySkillComposer",
    "CompositionPlan",
    "StrategyType",
    "MultimodalEnzymeGenerator",
    "SynthesizedEnzyme",
    "CatalyticActiveSite",
    "DiscoveryCurriculumRunner",
    "CurriculumPhaseResult",
    "DiscoveryEvaluator",
    "BiomechanicalGaitModel",
    "NavalStabilityModel",
    "StrideAssessment",
    "NavalStabilityAssessment",
]
