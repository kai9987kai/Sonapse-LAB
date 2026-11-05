import pytest
import math
from poseidon.phase_plane import PhasePlaneEngine, PhaseStateAudit


def test_phase_plane_attractor_and_lyapunov():
    engine = PhasePlaneEngine()
    # At exact attractor coordinates: L(x*) should be 0
    l_zero = engine.compute_lyapunov_function(
        e=engine.ATTRACTOR_ENERGY,
        h=engine.ATTRACTOR_HYDRATION,
        s=engine.ATTRACTOR_STAMINA,
        v=engine.ATTRACTOR_HEALTH,
    )
    assert abs(l_zero) < 1e-6

    # Deviating from attractor increases Lyapunov energy L(x) > 0
    l_dev = engine.compute_lyapunov_function(e=0.3, h=0.3, s=0.3, v=0.5)
    assert l_dev > 0.5


def test_phase_plane_separatrix_distance():
    engine = PhasePlaneEngine()
    # High vital reserves
    dist_high = engine.compute_separatrix_distance(e=0.9, h=0.9, v=1.0)
    assert dist_high > 0.8

    # Near starvation / dehydration
    dist_low = engine.compute_separatrix_distance(e=0.13, h=0.13, v=0.16)
    assert dist_low < 0.05


def test_phase_plane_analysis_and_vector_field():
    engine = PhasePlaneEngine()
    # Organism in depleted state with local food available
    obs = [1.0, 0.25, 0.80, 0.85, 0.05, 0.0, 0.90, 0.50, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    audit = engine.analyze_phase_space(obs)
    assert isinstance(audit, PhaseStateAudit)
    assert len(audit.action_phase_drifts) == 6
    # With low energy and high local food, foraging (action 1) should reduce Lyapunov drift towards attractor
    assert audit.action_phase_drifts[1] < audit.action_phase_drifts[4]
