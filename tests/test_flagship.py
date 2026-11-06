"""Tests for Flagship Dual-Task Skill Transfer and Recursive Learning Evolution."""
import pytest
from synapse.flagship import FlagshipExperimentRunner


def test_flagship_full_experiment():
    runner = FlagshipExperimentRunner(seed=42)
    report = runner.run_full_experiment(
        training_steps=5,
        evolution_generations=2,
        eval_episodes=2,
    )

    assert report['status'] == 'success'
    assert 'stage1_donor_losses' in report
    assert 'stage2_dual_skill_retention' in report
    assert 'stage3_double_dissociation_ablation' in report
    assert 'stage4_recursive_evolution' in report
    assert 'stage5_comparative_control_benchmark' in report

    # Double dissociation verification
    assert report['stage3_double_dissociation_ablation']['double_dissociation_verified'] is True

    # Comparative benchmark vs controls
    bench = report['stage5_comparative_control_benchmark']
    assert 'arm1_evolved_recipient' in bench
    assert 'arm2_tabula_rasa_scratch' in bench
    assert 'arm3_static_recipient' in bench
    assert 'arm4_scrambled_control' in bench
    assert bench['statistically_superior'] is True
