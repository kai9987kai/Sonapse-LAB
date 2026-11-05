import math
import pytest

from poseidon.nexusflow import NexusFlowNetwork, FlowWaypoint


def test_nexusflow_registration_and_flux():
    flow = NexusFlowNetwork(decay_tau=2.0)
    w1 = flow.register_waypoint("n1", 0.0, 0.0, replenishment=0.8, depletion=0.1)
    w2 = flow.register_waypoint("n2", 1.0, 1.0, replenishment=1.0, depletion=0.0)

    assert w1.potential > 0.0
    assert w2.potential > w1.potential

    # Flux from lower to higher potential
    flux = flow.compute_causal_flux("n1", "n2")
    assert flux > 0.0

    # Reverse flux should be 0 because potential decreases
    reverse_flux = flow.compute_causal_flux("n2", "n1")
    assert reverse_flux == 0.0


def test_nexusflow_superposition_interference():
    flow = NexusFlowNetwork()
    
    # Constructive alignment: angles aligned (0 vs 0)
    interf_aligned, conflict_aligned = flow.evaluate_superposition_interference(
        value_a=1.0, value_b=1.0, heading_angle_a=0.0, heading_angle_b=0.0
    )
    assert interf_aligned > 0.0
    assert not conflict_aligned

    # Destructive conflict: opposite directions (0 vs pi)
    interf_opp, conflict_opp = flow.evaluate_superposition_interference(
        value_a=1.0, value_b=1.0, heading_angle_a=0.0, heading_angle_b=math.pi
    )
    assert interf_opp < 0.0
    assert conflict_opp
    assert flow.interference_events == 1


def test_nexusflow_action_recommendation():
    flow = NexusFlowNetwork()
    flow.register_waypoint("origin", 0.0, 0.0, replenishment=0.2)
    flow.register_waypoint("target_right", 2.0, 0.0, replenishment=0.9)

    action, info = flow.recommend_action_from_flow(current_x=0.0, current_y=0.0, compass_heading=0.0)
    assert action == 4  # RIGHT (+x)
    assert info["source"] == "nexus_gradient"
    assert info["target_node"] == "target_right"
