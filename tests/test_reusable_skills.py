"""Unit tests for Reusable Macro-Action Skills Library."""
import pytest
import numpy as np
from synapse.reusable_skills import MacroSkill, ReusableSkillLibrary


def test_macro_skill_properties_and_precondition():
    skill = MacroSkill(
        skill_id="test_01",
        name="TurnAndMove",
        precondition_centroid=[0.5, 0.5, 0.0, 0.0],
        precondition_radius=0.5,
        action_sequence=[[0.2, 0.1], [0.3, 0.0]],
        expected_duration=2,
        expected_reward=1.5,
        success_count=3,
        attempt_count=4
    )
    assert skill.success_rate == 0.75

    # In-precondition
    obs_near = np.array([0.6, 0.5, 0.0, 0.0, 0.9, 0.8], dtype=np.float32)
    assert skill.matches_precondition(obs_near)

    # Out-of-precondition
    obs_far = np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)
    assert not skill.matches_precondition(obs_far)

    d = skill.to_dict()
    assert d['skill_id'] == "test_01"
    assert d['success_rate'] == 0.75


def test_skill_discovery_and_query():
    lib = ReusableSkillLibrary(max_skills=5, discovery_window=50, min_skill_length=3, max_skill_length=6, seed=42)
    rng = np.random.default_rng(42)

    # Record some transitions with a reward spike
    for t in range(20):
        obs = rng.normal(0.0, 0.1, size=155).astype(np.float32)
        act = rng.uniform(-0.5, 0.5, size=5).astype(np.float32)
        next_obs = obs + 0.05
        rew = 1.0 if t in (8, 18) else 0.0
        lib.record_transition(obs, act, rew, next_obs, step=t)

    discovered = lib.discover_skills(reward_threshold=0.5)
    assert len(discovered) > 0
    assert lib.total_discoveries > 0

    # Query matching skill
    test_obs = np.zeros(155, dtype=np.float32)
    matched = lib.query_skill(test_obs)
    assert matched is not None
    assert matched.skill_id in lib.skills

    # Record execution
    lib.record_skill_execution(matched.skill_id, success=True)
    assert matched.attempt_count == 1
    assert matched.success_count == 1


def test_skill_pruning_and_state_roundtrip():
    lib = ReusableSkillLibrary(max_skills=3, seed=42)
    for i in range(4):
        s = MacroSkill(
            skill_id=f"skill_{i}",
            name=f"Skill {i}",
            precondition_centroid=[float(i), 0.0],
            precondition_radius=1.0,
            action_sequence=[[0.1, 0.2]],
            expected_duration=1,
            expected_reward=float(i + 1),
            success_count=i,
            attempt_count=i + 1
        )
        lib.skills[s.skill_id] = s
        if len(lib.skills) > lib.max_skills:
            lib._prune_lowest_utility_skill()

    assert len(lib.skills) <= 3
    assert "skill_0" not in lib.skills  # Lowest utility should be pruned

    # State roundtrip
    state = lib.export_state()
    lib2 = ReusableSkillLibrary(max_skills=3, seed=42)
    lib2.import_state(state)
    assert len(lib2.skills) == len(lib.skills)
    assert set(lib2.skills.keys()) == set(lib.skills.keys())
