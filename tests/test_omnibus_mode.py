"""Verification suite for Frontier Apex Omnibus Mode.

Tests:
1. Omnibus configuration with 2000 neurons and all frontier modules active.
2. Stepping across multi-environment curriculum.
3. Sleep consolidation execution post stage.
4. Backward retention and catastrophic forgetting index evaluation.
5. Cryptographically verifiable receipt generation and SHA-256 hash stability.
6. Workbench server REST endpoints (/api/omnibus/status, /api/omnibus/step, /api/omnibus/run).
"""
import json
import pytest
import numpy as np

from synapse.config import SynapseConfig
from synapse.omnibus_mode import OmnibusRunner, OMNIBUS_CURRICULUM
from synapse.server import Workbench
from synapse.runtime import LabRuntime


def test_frontier_apex_config_factory():
    cfg = SynapseConfig.frontier_apex(total_neurons=2000, seed=42)
    assert cfg.total_neurons == 2000
    assert cfg.frontier_all is True
    assert cfg.experience_planning is True
    assert cfg.surprise_memory is True
    assert cfg.experience_horizon == 2
    assert cfg.continual_backprop is True
    assert cfg.fast_weight_memory is True
    assert cfg.latent_reasoning is True
    assert cfg.self_model is True
    assert cfg.source_node_counts() == (667, 667, 666)


def test_omnibus_runner_initialization_and_single_step():
    runner = OmnibusRunner(total_neurons=2000, seed=7, steps_per_task=2)
    assert runner.agent.brain.n == 2000
    assert runner.current_task == OMNIBUS_CURRICULUM[0]
    assert runner.current_stage_idx == 0

    out = runner.step()
    assert out["status"] == "running"
    assert out["task"] == "forage"
    assert "agency" in out
    assert "effective_rank" in out
    assert "epistemic_unc" in out
    assert "step_latency_ms" in out
    assert out["step_latency_ms"] > 0.0


def test_omnibus_runner_stage_progression_and_receipt():
    # Run a compact curriculum of 3 tasks with 2 steps each
    compact_tasks = ["forage", "maze", "apex_megasite"]
    runner = OmnibusRunner(total_neurons=2000, seed=19, steps_per_task=2, curriculum_tasks=compact_tasks)

    receipt = runner.run_full_curriculum()
    assert runner.is_completed is True
    assert len(receipt.stages) == 3
    assert receipt.stages[0]["task"] == "forage"
    assert receipt.stages[1]["task"] == "maze"
    assert receipt.stages[2]["task"] == "apex_megasite"

    # Verifiable cryptographic hash
    assert len(receipt.cryptographic_sha256) == 64
    assert receipt.cryptographic_sha256 == receipt.compute_hash()

    # Retention rate and forgetting index within realistic bounds
    assert 0.0 <= receipt.final_backward_retention <= 1.5
    assert 0.0 <= receipt.catastrophic_forgetting <= 1.0
    assert receipt.total_neurons == 2000
    assert receipt.total_sleep_cycles >= 3


def test_omnibus_evaluation_avoids_full_checkpoint_clone(monkeypatch):
    runner = OmnibusRunner(
        total_neurons=48, curriculum_tasks=['forage'], steps_per_task=1,
        eval_steps=1, eval_episodes=1,
    )
    candidate = runner.agent.clone_for_evaluation()
    assert candidate.brain is not runner.agent.brain
    assert candidate.neurogenesis is runner.agent.neurogenesis
    assert candidate.multitimescale is runner.agent.multitimescale

    def reject_full_export():
        raise AssertionError('evaluation must not export and duplicate the full agent state')

    monkeypatch.setattr(runner.agent, 'export_state', reject_full_export)
    scores = runner.evaluate_agent(runner.agent, ['forage'])

    assert list(scores) == ['forage']
    assert np.isfinite(scores['forage'])


def test_server_omnibus_api_endpoints(tmp_path):
    app = Workbench(LabRuntime(SynapseConfig(total_neurons=2000), tmp_path), output_root=tmp_path)

    # Initial idle status
    status = app.get("/api/omnibus/status")
    assert status["status"] == "idle"
    assert status["total_neurons"] == 2000

    # Step endpoint
    step_res = app.post("/api/omnibus/step", {
        "steps_per_task": 2,
        "total_neurons": 2000,
        "tasks": ["forage", "stealth_labyrinth"]
    })
    assert "step_result" in step_res
    assert step_res["step_result"]["task"] == "forage"

    # Active status query
    status_active = app.get("/api/omnibus/status")
    assert status_active["total_neurons"] == 2000
    assert status_active["current_task"] == "forage"
    assert status_active["total_stages"] == 2

    # Reset endpoint
    reset_res = app.post("/api/omnibus/reset", {})
    assert reset_res["status"] == "reset"
    assert app.omnibus_runner is None
