"""Tests for Allocentric Cognitive Mapping, Probabilistic Belief Tracking, Multi-timescale Memory, Rule Change Detection, and Tripartite Memory (Items 26-30).
"""
import numpy as np
from synapse.spatial_map import AllocentricCognitiveMap, ParticleFilterBeliefTracker
from synapse.multiscale_memory import (
    MultiTimescaleWorkingMemory,
    RuleChangeDetector,
    TripartiteMemorySystem
)


def test_allocentric_cognitive_map_and_path_planning():
    c_map = AllocentricCognitiveMap(grid_size=30, resolution=0.5)
    
    # 1. Integrate odometry
    c_map.integrate_odometry(forward_velocity=1.0, angular_velocity=0.0, dt=1.0)
    assert c_map.pose[0] > 0.5
    
    # 2. Register landmark
    c_map.register_landmark("water_source", np.array([2.0, 1.0]), np.ones(8))
    assert "water_source" in c_map.landmarks
    assert c_map.landmarks["water_source"].position[0] > 1.0
    
    # 3. Ray observation
    rays = np.ones(33) * 0.9 # mostly clear
    rays[16] = 0.2 # close obstacle ahead
    c_map.observe_rays(rays)
    
    # 4. Plan path even when cues are missing
    path = c_map.plan_path_to_target(np.array([4.0, 4.0]))
    assert len(path) >= 2
    assert isinstance(path[0], np.ndarray)


def test_particle_filter_belief_tracker():
    pf = ParticleFilterBeliefTracker(num_particles=200, state_dim=2, seed=42)
    initial_trace = np.trace(pf.estimated_covariance)
    for _ in range(8):
        pf.predict(np.zeros(2), dt=1.0)
        noisy_measurement = np.array([1.0, 2.0]) + np.random.normal(0, 0.05, size=2)
        pf.update(noisy_measurement, meas_std=0.2)
    est_mean = pf.estimated_mean
    est_cov = pf.estimated_covariance
    assert np.linalg.norm(est_mean - np.array([1.0, 2.0])) < 0.6
    assert np.trace(est_cov) < initial_trace
def test_multiscale_working_memory():
    wm = MultiTimescaleWorkingMemory(feature_dim=16)
    
    # Step with steady stimulus
    stimulus = np.ones(16, dtype=np.float32) * 2.0
    for _ in range(5):
        fused = wm.step(stimulus)
        
    tiers = wm.get_tier_states()
    fast = tiers['fast_sensory']
    med = tiers['medium_context']
    slow = tiers['slow_behavior']
    
    # Fast trace responds quickest; slow responds gradual
    assert fast.mean() > med.mean() > slow.mean()
    assert fused.shape == (16,)
    
    # Now step with zero stimulus
    zero_stim = np.zeros(16, dtype=np.float32)
    wm.step(zero_stim)
    tiers_after = wm.get_tier_states()
    
    # Fast drops faster than slow
    fast_drop = fast.mean() - tiers_after['fast_sensory'].mean()
    slow_drop = slow.mean() - tiers_after['slow_behavior'].mean()
    assert fast_drop > slow_drop


def test_rule_change_detector():
    detector = RuleChangeDetector(delta=0.05, threshold=6.0)
    
    # Stationary phase (small errors around 0.1)
    for _ in range(30):
        shift, stat = detector.update(0.1 + np.random.uniform(-0.02, 0.02))
        assert shift is False
        
    # Sudden rule change: errors spike to 2.0+
    detected = False
    for _ in range(10):
        shift, stat = detector.update(2.5)
        if shift:
            detected = True
            break
            
    assert detected is True
    assert detector.rule_shift_detected is True


def test_tripartite_memory_stores():
    tms = TripartiteMemorySystem(episodic_capacity=50)
    
    # 1. Episodic store
    obs1 = np.ones(10, dtype=np.float32)
    obs2 = -np.ones(10, dtype=np.float32)
    tms.store_episode_step(obs1, np.zeros(5), reward=1.0, next_obs=obs1 * 1.1)
    tms.store_episode_step(obs2, np.zeros(5), reward=-1.0, next_obs=obs2 * 1.1)
    
    retrieved = tms.retrieve_similar_episodes(np.ones(10) * 0.9, top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].reward == 1.0 # Cosine match to obs1
    
    # 2. Semantic store
    tms.add_semantic_fact("apple", "is_a", "food", confidence=1.0)
    tms.add_semantic_fact("apple", "has_color", "red", confidence=0.9)
    query_res = tms.query_semantic("apple")
    assert len(query_res) == 2
    assert {q.relation for q in query_res} == {"is_a", "has_color"}
    
    # 3. Procedural store
    tms.register_procedural_skill("grasp_object", {"w": np.ones((2, 2))}, preconditions="in_reach", duration=15)
    skill = tms.retrieve_procedural_skill("grasp_object")
    assert skill is not None
    assert skill.preconditions == "in_reach"
    assert skill.expected_duration == 15
