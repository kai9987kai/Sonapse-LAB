"""Tests for Functional Memory, Skill Transplantation, and Causal Ablation."""
import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.transfer import SkillTransplantationEngine, NavigationModule, WorkingMemoryModule


def test_modular_skills_and_training():
    engine = SkillTransplantationEngine()
    nav_loss = engine.train_donor_skill('navigation', steps=10)
    assert isinstance(nav_loss, float)
    assert nav_loss >= 0.0

    wm_loss = engine.train_donor_skill('working_memory', steps=10)
    assert isinstance(wm_loss, float)
    assert wm_loss >= 0.0


def test_skill_transplantation_and_ablation_assay():
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))
    engine = SkillTransplantationEngine()

    engine.train_donor_skill('navigation', steps=10)
    receipt = engine.transplant_to_agent('navigation', agent)
    assert receipt['skill_name'] == 'navigation'
    assert receipt['resulting_digest'] != receipt['parent_digest']
    assert receipt['manifold_gromov_wasserstein_discrepancy'] >= 0.0

    # Run causal ablation assay
    assay = engine.run_causal_ablation_assay(agent, 'navigation', episodes=2)
    assert 'intact_score' in assay
    assert 'ablated_score' in assay
    assert 'causal_performance_drop' in assay
    assert assay['causal_necessity_ratio'] >= 0.0


def test_negative_control_transfer():
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))
    engine = SkillTransplantationEngine()
    res = engine.run_negative_control(agent, 'navigation')
    assert res['control_type'] == 'scrambled_untrained_donor'
    assert 'scrambled_score' in res
