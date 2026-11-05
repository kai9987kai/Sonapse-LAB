import pytest
from poseidon.ghost_monologue import GhostIntrospectionEngine
from poseidon.phase_plane import PhasePlaneEngine


def test_ghost_latent_drives_and_monologue():
    ghost = GhostIntrospectionEngine()
    phase = PhasePlaneEngine()

    # Thirsty organism in exposed terrain
    obs = [1.0, 0.80, 0.20, 0.80, 0.70, 0.1, 0.5, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase_audit = phase.analyze_phase_space(obs)

    drives = ghost.compute_latent_drives(obs, surprise=0.1, separatrix_dist=phase_audit.separatrix_distance)
    assert drives.dominant_drive == "thirst"
    assert drives.thirst > 0.7

    audit = ghost.synthesize_monologue(
        obs=obs,
        phase_audit=phase_audit,
        quantum_coherence=0.92,
        quantum_gain=0.45,
        causal_effect=0.35,
        active_neurons=64,
        morph_state="stable",
        chosen_action=2,  # Drink
        base_action=2,
        override=False,
        guardrail_veto=False,
        surprise=0.05,
    )
    assert "DRINK" in audit.monologue
    assert "THIRST" in audit.monologue
    assert audit.agency_score > 0.5
    assert len(audit.planes_consulted) >= 5
