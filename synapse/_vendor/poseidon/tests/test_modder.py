"""Tests for Universal Modder Dynamic Bytecode & Hook Interceptor Engine."""
from poseidon.modder import UniversalModderEngine, HookPoint


def test_modder_initialization_and_manifest():
    engine = UniversalModderEngine()
    manifest = engine.manifest()

    assert manifest["total_hooks"] >= 2
    assert "emergency_stamina_guard" in [h["description"] for h in manifest["hooks"]] or len(manifest["hooks"]) >= 2
    assert len(manifest["manifest_digest"]) == 16


def test_modder_emergency_stamina_guard():
    engine = UniversalModderEngine()
    # Obs with near-zero stamina (obs[3] = 0.05)
    exhausted_obs = [0.8, 0.7, 0.6, 0.05, 0.1, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.1]

    # Proposed action 4 (explore - costly movement)
    final_action, reason = engine.apply_pre_action_trampolines(proposed_action=4, observation=exhausted_obs, context={})
    assert final_action == 0  # Trampolined to rest
    assert reason is not None
    assert "emergency_stamina_depletion_override" in reason


def test_modder_severe_storm_shelter_guard():
    engine = UniversalModderEngine()
    # Severe storm obs: severity = 0.95 (obs[9]), exposure = 0.85 (obs[4]), shelter = 0.80 (obs[8])
    storm_obs = [0.8, 0.7, 0.6, 0.5, 0.85, 0.0, 0.5, 0.5, 0.80, 0.95, 0.5, 0.5, 0.0, 0.5, 0.0, 0.1]

    # Proposed action 1 (forage outdoors)
    final_action, reason = engine.apply_pre_action_trampolines(proposed_action=1, observation=storm_obs, context={})
    assert final_action == 3  # Trampolined to shelter
    assert reason is not None
    assert "severe_storm_shelter_override" in reason


def test_modder_custom_hook_registration():
    engine = UniversalModderEngine()

    def custom_guard(action, obs, ctx):
        if action == 5:
            return (0, "custom_freeze_override")
        return (action, None)

    hook_id = engine.register_hook(HookPoint.PRE_ACTION, "custom_freeze", custom_guard, priority=5)
    assert len(hook_id) == 12

    # Test override
    final_action, reason = engine.apply_pre_action_trampolines(proposed_action=5, observation=[0.5] * 16, context={})
    assert final_action == 0
    assert reason == "custom_freeze_override"

    # Unregister
    removed = engine.unregister_hook(hook_id)
    assert removed is True
