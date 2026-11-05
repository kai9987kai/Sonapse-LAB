import math
import numpy as np
import pytest

from poseidon.cx_ring import CXRingCompass


def test_cx_ring_bump_formation_and_pva_alignment():
    compass = CXRingCompass(num_wedges=16, initial_heading=math.pi / 2)
    pva, coh = compass.readout()
    # Heading should align near pi/2 (1.5708)
    assert abs(pva - math.pi / 2) < 0.1
    # Coherence of a fresh Gaussian bump should be high
    assert coh > 0.6
    assert not compass.is_disoriented()


def test_cx_ring_angular_velocity_shifts_bump():
    compass = CXRingCompass(num_wedges=16, initial_heading=1.0)
    initial_pva, _ = compass.readout()
    
    # Positive angular velocity (turning counter-clockwise) for multiple steps
    for _ in range(5):
        compass.step(angular_velocity=1.0, dt=0.2)
        
    final_pva, _ = compass.readout()
    # Circular difference should be positive (counter-clockwise)
    circ_diff = math.atan2(math.sin(final_pva - initial_pva), math.cos(final_pva - initial_pva))
    assert circ_diff > 0.0


def test_cx_ring_optomotor_reflex_stabilization():
    compass = CXRingCompass(num_wedges=16)
    # Manually disperse activations to create disorientation (flat noise)
    compass.state = np.ones(16) / 16.0
    _, coh = compass.readout()
    assert coh < 0.1
    assert compass.is_disoriented()
    
    # Reflex should steer left or right to reorient
    action = compass.optomotor_reflex_action(current_action=0, orientation_error=0.8)
    assert action == 4  # Turn right
    
    action_left = compass.optomotor_reflex_action(current_action=0, orientation_error=-0.8)
    assert action_left == 3  # Turn left
