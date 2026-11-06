"""Counterfactual Neural Twin Engine with Live Neural Intervention Hooks.

Runs two synchronized, independently simulated neural twins experiencing
identical sensory observation streams. The counterfactual twin receives precise
neural intervention hooks after selected internal updates:
1. 'pre_graph': Intervene on encoded sensory/intent current.
2. 'intra_graph': Silence, clamp, or inject current into designated neurons/circuits.
3. 'neuromod': Clamp or perturb dynamic neurochemical fields (DA, 5-HT, OA, ACh).
4. 'post_graph': Intervene on bridge representations or motor decoders.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import torch
from .agent import IntegratedAgent
from .config import SynapseConfig

@dataclass
class NeuralIntervention:
    stage: str  # 'pre_graph', 'intra_graph', 'neuromod', 'post_graph'
    target: str # e.g. 'silence_neuropil', 'clamp_dopamine', 'bias_motor', 'custom'
    value: float = 0.0
    neuron_mask: Optional[list[int]] = None
    custom_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def apply(self, stage_name: str, tensor: torch.Tensor, agent: IntegratedAgent) -> torch.Tensor:
        if stage_name != self.stage:
            return tensor
        if self.custom_fn is not None:
            return self.custom_fn(tensor)

        if self.stage == 'intra_graph' and self.neuron_mask:
            modified = tensor.clone()
            for idx in self.neuron_mask:
                if 0 <= idx < len(modified):
                    modified[idx] = self.value
            return modified

        if self.stage == 'neuromod':
            if hasattr(agent, 'neuromod'):
                if self.target == 'dopamine':
                    agent.neuromod.state.dopamine = self.value
                elif self.target == 'serotonin':
                    agent.neuromod.state.serotonin = self.value
                elif self.target == 'octopamine':
                    agent.neuromod.state.octopamine = self.value
                elif self.target == 'acetylcholine':
                    agent.neuromod.state.acetylcholine = self.value
            return tensor

        if self.stage == 'pre_graph':
            return tensor + self.value

        if self.stage == 'post_graph':
            return tensor + self.value

        return tensor

class CounterfactualNeuralTwinRunner:
    """Manages twin simulation and computes causal divergence profiles."""

    def __init__(self, config: SynapseConfig):
        self.config = config
        self.factual_agent = IntegratedAgent(config)
        self.counterfactual_agent = IntegratedAgent(config)
        # Ensure identical initial states
        initial_state = copy.deepcopy(self.factual_agent.export_state())
        self.counterfactual_agent.import_state(initial_state)

    def run_paired_episode(
        self,
        observations: list[np.ndarray],
        interventions: list[NeuralIntervention]
    ) -> dict:
        """Run synchronized factual vs counterfactual rollout over identical observations."""
        factual_actions = []
        cf_actions = []
        action_divergences = []
        rate_divergences = []
        free_energies_factual = []
        free_energies_cf = []

        self.factual_agent.reset_episode()
        self.counterfactual_agent.reset_episode()

        for t, obs in enumerate(observations):
            # 1. Factual step
            res_fact = self.factual_agent.act(obs, learn=False)
            a_fact = res_fact['action']
            r_fact = self.factual_agent.brain.r.detach().cpu().numpy()

            # 2. Counterfactual step with hooks
            # Apply pre-step neuromod interventions
            for hook in interventions:
                if hook.stage == 'neuromod':
                    hook.apply('neuromod', torch.empty(0), self.counterfactual_agent)

            # Custom stepping for counterfactual agent with stage interception
            cf_o = torch.from_numpy(obs.copy())
            with torch.no_grad():
                cf_compass = self.counterfactual_agent.compass.step(float(obs[137]) * 2.5, obs[:132])
                cf_mbon = self.counterfactual_agent.mb(cf_o[149:153], adapt=False)

                if self.counterfactual_agent.ticks % self.counterfactual_agent.config.cognition_interval == 0:
                    ctx = self.counterfactual_agent._pooled(self.counterfactual_agent.brain.r)
                    frame = (self.counterfactual_agent.semantic_frame(obs) + 0.05 * torch.tanh(self.counterfactual_agent.graph_to_resource(ctx))).clamp(0, 1)
                    out = self.counterfactual_agent.tidal(torch.zeros(1, self.counterfactual_agent.tidal.config.hash_buckets), frame[None], memory=self.counterfactual_agent.hidden)
                    self.counterfactual_agent.hidden = out['memory'].detach()
                    self.counterfactual_agent.intent = out['action'][0].softmax(-1).detach()

                current = 2 * torch.tanh(self.counterfactual_agent.encoder(torch.cat([cf_o, cf_compass, cf_mbon])) + 0.3 * self.counterfactual_agent.donor_to_graph(torch.cat([self.counterfactual_agent.hidden[0], self.counterfactual_agent.intent])))

                # Pre-graph intervention hook
                for hook in interventions:
                    current = hook.apply('pre_graph', current, self.counterfactual_agent)

                gates = self.counterfactual_agent.plasticity.gates(cf_o)
                effective = self.counterfactual_agent.plasticity.effective_weights(self.counterfactual_agent.brain.graft_w, gates)
                if hasattr(self.counterfactual_agent, 'astrocytes'):
                    effective = self.counterfactual_agent.astrocytes.modulate_graft_weights(effective, torch.as_tensor(self.counterfactual_agent.graph.graft_pre, dtype=torch.long), self.counterfactual_agent.brain.n)

                r_cf_tensor = self.counterfactual_agent.brain(current, effective, adapt=False)

                # Intra-graph intervention hook
                for hook in interventions:
                    r_cf_tensor = hook.apply('intra_graph', r_cf_tensor, self.counterfactual_agent)

                pooled_cf = self.counterfactual_agent._pooled(r_cf_tensor)

                # Post-graph intervention hook
                for hook in interventions:
                    pooled_cf = hook.apply('post_graph', pooled_cf, self.counterfactual_agent)

                features_cf = torch.cat([pooled_cf, self.counterfactual_agent.hidden[0, :16], cf_mbon, self.counterfactual_agent.intent, cf_compass[:8]])
                gains_cf = self.counterfactual_agent.neuromod.compute_gains() if hasattr(self.counterfactual_agent, 'neuromod') else {'motor_vigor': 1.0}
                a_cf_tensor = torch.tanh(self.counterfactual_agent.motor(features_cf)) * torch.sigmoid(pooled_cf.mean() * 4) * gains_cf['motor_vigor']
                a_cf = a_cf_tensor.numpy()
                r_cf = r_cf_tensor.detach().cpu().numpy()

            # Record metrics
            factual_actions.append(a_fact.tolist())
            cf_actions.append(a_cf.tolist())
            action_diff = float(np.linalg.norm(a_fact - a_cf))
            rate_diff = float(np.linalg.norm(r_fact - r_cf))
            action_divergences.append(action_diff)
            rate_divergences.append(rate_diff)

        mean_action_div = float(np.mean(action_divergences)) if action_divergences else 0.0
        max_action_div = float(np.max(action_divergences)) if action_divergences else 0.0
        mean_rate_div = float(np.mean(rate_divergences)) if rate_divergences else 0.0

        return {
            'status': 'ok',
            'timesteps': len(observations),
            'interventions_count': len(interventions),
            'mean_action_divergence': mean_action_div,
            'max_action_divergence': max_action_div,
            'mean_rate_divergence': mean_rate_div,
            'causal_intervention_active': mean_action_div > 1e-6,
            'action_divergence_series': action_divergences,
            'rate_divergence_series': rate_divergences,
            'scope': 'paired counterfactual neural twin simulation with causal intervention hooks'
        }
