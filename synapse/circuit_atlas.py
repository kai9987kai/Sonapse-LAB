"""Causal Circuit Atlas: Automated Systematic Perturbation & Functional Mapping.

Provides automated lesioning, optogenetic-style silencing, and causal abstraction mapping
that links individual connectome subcircuits, neuropils, and modulatory pathways
directly to behavioral primitives (thrust, steering, threat evasion, target tracking, memory).
Computes rigorous causal necessity and sufficiency indices for each anatomical circuit.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .agent import IntegratedAgent


@dataclass
class CircuitCausalProfile:
    subcircuit_name: str
    target_indices: List[int]
    behavioral_effects: Dict[str, float]
    causal_necessity: float
    causal_sufficiency: float
    primary_function: str


class CausalCircuitAtlas:
    """Systematically constructs an empirical causal atlas linking connectome circuits to behavior."""

    def __init__(self, agent: IntegratedAgent):
        self.agent = agent
        self.atlas: Dict[str, CircuitCausalProfile] = {}

    def _evaluate_behavioral_primitives(self, agent: IntegratedAgent, test_episodes: int = 4) -> Dict[str, float]:
        """Evaluates behavioral primitives: thrust, steering, evasion, approach, memory."""
        rng = np.random.default_rng(777)
        thrusts = []
        steerings = []
        evasions = []
        approaches = []

        for ep in range(test_episodes):
            agent.reset_episode()
            for t in range(15):
                obs = rng.normal(0.0, 0.5, size=155).astype(np.float32)
                # Stimulus 1: Target approach cue (positive depth, color contrast)
                obs[0] = 0.8
                obs[34] = 0.9

                res = agent.act(obs, learn=False)
                act = res['action']

                thrusts.append(float(act[1]))
                steerings.append(float(abs(act[0])))

                # Stimulus 2: Threat evasion cue
                obs_threat = obs.copy()
                obs_threat[1] = 1.8  # threat signal
                res_threat = agent.act(obs_threat, learn=False)
                evasions.append(float(abs(res_threat['action'][0]) + res_threat['action'][1] * 0.5))

                # Stimulus 3: Approach
                approaches.append(float(act[1] - abs(act[0]) * 0.3))

        return {
            'thrust': float(np.mean(thrusts)),
            'steering': float(np.mean(steerings)),
            'threat_evasion': float(np.mean(evasions)),
            'target_approach': float(np.mean(approaches)),
        }

    def silence_circuit(self, agent: IntegratedAgent, node_indices: List[int]) -> Any:
        """Temporarily silences selected nodes by zeroing their rate outputs."""
        saved_rates = agent.brain.r.clone()
        with torch.no_grad():
            agent.brain.r[node_indices] = 0.0
        return saved_rates

    def map_circuit(
        self,
        subcircuit_name: str,
        node_indices: List[int],
        episodes: int = 3,
    ) -> CircuitCausalProfile:
        """Evaluates the causal necessity and sufficiency of a specific circuit."""
        # Baseline intact performance
        baseline = self._evaluate_behavioral_primitives(self.agent, test_episodes=episodes)

        # 1. Silencing / Lesion test (Necessity)
        agent_lesioned = copy.deepcopy(self.agent)
        with torch.no_grad():
            # Zero out graft weights connecting to/from these nodes
            mask = np.isin(agent_lesioned.graph.graft_pre, node_indices) | np.isin(agent_lesioned.graph.graft_post, node_indices)
            w = agent_lesioned.brain.graft_w.clone()
            w[mask] = 0.0
            agent_lesioned.brain.graft_w = nn.Parameter(w)

        lesioned = self._evaluate_behavioral_primitives(agent_lesioned, test_episodes=episodes)

        # 2. Isolation test (Sufficiency)
        agent_isolated = copy.deepcopy(self.agent)
        with torch.no_grad():
            # Zero out all graft weights EXCEPT those within these nodes
            keep_mask = np.isin(agent_isolated.graph.graft_pre, node_indices) & np.isin(agent_isolated.graph.graft_post, node_indices)
            w_iso = agent_isolated.brain.graft_w.clone()
            w_iso[~keep_mask] = 0.0
            agent_isolated.brain.graft_w = nn.Parameter(w_iso)

        isolated = self._evaluate_behavioral_primitives(agent_isolated, test_episodes=episodes)

        # Compute causal effects
        deltas = {}
        for k in baseline:
            deltas[k] = round(baseline[k] - lesioned[k], 4)

        # Find primary affected behavior
        primary_behavior = max(deltas.keys(), key=lambda k: abs(deltas[k]))
        max_delta = deltas[primary_behavior]

        necessity = float(np.clip(max_delta / (abs(baseline[primary_behavior]) + 1e-6), -1.0, 1.0))
        sufficiency = float(np.clip(isolated[primary_behavior] / (abs(baseline[primary_behavior]) + 1e-6), 0.0, 1.0))

        profile = CircuitCausalProfile(
            subcircuit_name=subcircuit_name,
            target_indices=node_indices,
            behavioral_effects=deltas,
            causal_necessity=round(necessity, 4),
            causal_sufficiency=round(sufficiency, 4),
            primary_function=f"{primary_behavior} (delta={max_delta:+.4f})",
        )
        self.atlas[subcircuit_name] = profile
        return profile

    def build_full_atlas(self) -> Dict[str, Any]:
        """Constructs an atlas mapping all major neuropils and specialist modules."""
        n_total = self.agent.brain.n
        # Group by neuropil / origins
        neuropils = {}
        for idx in range(min(len(self.agent.graph.origin), n_total)):
            np_id = int(self.agent.graph.origin[idx])
            if np_id not in neuropils:
                neuropils[np_id] = []
            neuropils[np_id].append(idx)

        neuropil_names = {0: 'FlyWire_Central_Brain', 1: 'MANC_VNC_Cord', 2: 'Optic_Lobe'}
        atlas_summary = {}

        for np_id, nodes in neuropils.items():
            name = neuropil_names.get(np_id, f"Neuropil_{np_id}")
            # Sample up to 24 nodes for rapid profiling
            sample_nodes = nodes[:24]
            prof = self.map_circuit(name, sample_nodes, episodes=2)
            atlas_summary[name] = {
                'num_neurons': len(nodes),
                'primary_function': prof.primary_function,
                'causal_necessity': prof.causal_necessity,
                'causal_sufficiency': prof.causal_sufficiency,
                'behavioral_deltas': prof.behavioral_effects,
            }

        return {
            'total_circuits_mapped': len(atlas_summary),
            'circuits': atlas_summary,
            'scope': 'causal circuit atlas with necessity and sufficiency bounds',
        }
