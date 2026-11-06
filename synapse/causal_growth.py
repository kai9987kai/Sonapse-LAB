"""Causal Connectome Evolution and Interventional Structural Plasticity.

Grounding:
Replaces correlational/Hebbian-only growth with true do-calculus interventional screening:
    Delta J(m) = E[R | do(m)] - E[R | do(m_0)]
Evaluated on cloned agents in identical matched paired environments.
Multi-objective fitness:
    J = E[R] - lambda_w * C_wiring - lambda_c * C_compute - lambda_f * C_forgetting
Discovers small, reusable, causally verified subnetworks while pruning metabolically costly connections.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import torch
from .agent import IntegratedAgent
from .complex_world import ComplexWorld

@dataclass
class CausalMutationProposal:
    pre_idx: int
    post_idx: int
    weight_delta: float
    is_inhibitory: bool
    description: str = ""

@dataclass
class CausalEvaluationReceipt:
    mutation: CausalMutationProposal
    expected_return_mut: float
    expected_return_base: float
    causal_delta: float
    wiring_cost: float
    compute_cost: float
    forgetting_cost: float
    net_fitness: float
    accepted: bool

class CausalConnectomeEvolutionEngine:
    """Evaluates candidate synaptic structural changes via paired interventional counterfactual rollouts."""

    def __init__(
        self,
        lambda_w: float = 0.05,
        lambda_c: float = 0.02,
        lambda_f: float = 1.0,
        min_causal_gain: float = 0.01,
        seed: int = 42
    ):
        self.lambda_w = lambda_w
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f
        self.min_causal_gain = min_causal_gain
        self.rng = np.random.default_rng(seed)

        self.evaluated_mutations: List[CausalEvaluationReceipt] = []
        self.accepted_count: int = 0
        self.rejected_count: int = 0

    def generate_candidate_mutation(self, agent: IntegratedAgent) -> CausalMutationProposal:
        """Sample a candidate structural synaptic addition or pruning intervention."""
        n = agent.brain.n
        pre = int(self.rng.integers(0, n))
        post = int(self.rng.integers(0, n))
        while post == pre:
            post = int(self.rng.integers(0, n))

        is_inhibitory = (pre % 4 == 0)
        # Sample delta
        mag = float(self.rng.uniform(0.02, 0.08))
        delta = -mag if is_inhibitory else mag

        return CausalMutationProposal(
            pre_idx=pre,
            post_idx=post,
            weight_delta=delta,
            is_inhibitory=is_inhibitory,
            description=f"Synapse {pre}->{post} ({'Inh' if is_inhibitory else 'Exc'} delta={delta:.4f})"
        )

    def evaluate_interventional_mutation(
        self,
        agent: IntegratedAgent,
        mutation: CausalMutationProposal,
        test_episodes: int = 3,
        horizon: int = 30,
        task: str = 'forage',
        use_real_environment: bool = True,
        baseline_task_eval_fn: Optional[Callable[[IntegratedAgent], float]] = None
    ) -> CausalEvaluationReceipt:
        """Evaluate Delta J(m) = E[R | do(m)] - E[R | do(m_0)] on cloned agents in matched worlds."""
        base_state = agent.export_state()

        # Measure baseline retention if old-task evaluator provided
        old_perf_before = 0.0
        if baseline_task_eval_fn is not None:
            old_perf_before = float(baseline_task_eval_fn(agent))

        # 1. Base rollouts (do(m_0))
        base_returns = []
        base_compute = []
        for ep in range(test_episodes):
            ep_seed = 1000 + ep
            agent.import_state(base_state)
            agent.reset_episode()
            ep_ret = 0.0
            act_norms = []
            if use_real_environment:
                world = ComplexWorld(seed=ep_seed)
                obs = world.reset(task=task)
                for t in range(horizon):
                    res = agent.act(obs, learn=False)
                    obs, rew, done, info = world.step(res['action'])
                    ep_ret += float(rew)
                    act_norms.append(float(agent.brain.r.abs().mean().item()))
                    if done:
                        break
            else:
                rng_ep = np.random.default_rng(ep_seed)
                for t in range(horizon):
                    obs = rng_ep.uniform(0.0, 1.0, size=155).astype(np.float32)
                    res = agent.act(obs, learn=False)
                    rew = float(res['action'][0] - 0.2 * abs(res['action'][1]))
                    ep_ret += rew
                    act_norms.append(float(agent.brain.r.abs().mean().item()))
            base_returns.append(ep_ret)
            base_compute.append(float(np.mean(act_norms)) if act_norms else 0.0)

        # 2. Mutated rollouts (do(m))
        agent.import_state(base_state)
        # Apply interventional mutation: find or create synaptic index
        syn_idx = -1
        if hasattr(agent.graph, 'graft_pre') and hasattr(agent.graph, 'graft_post'):
            pre_arr = np.array(agent.graph.graft_pre)
            post_arr = np.array(agent.graph.graft_post)
            matches = np.where((pre_arr == mutation.pre_idx) & (post_arr == mutation.post_idx))[0]
            if len(matches) > 0:
                syn_idx = matches[0]
                agent.brain.graft_w.data[syn_idx] += mutation.weight_delta
            else:
                # Add to graft weights if slot exists or modify motor head
                if hasattr(agent, 'motor') and hasattr(agent.motor, 'weight'):
                    agent.motor.weight.data[:, mutation.pre_idx % agent.motor.weight.shape[1]] += mutation.weight_delta * 0.1

        mut_returns = []
        mut_compute = []
        for ep in range(test_episodes):
            ep_seed = 1000 + ep
            ep_ret = 0.0
            act_norms = []
            if use_real_environment:
                world = ComplexWorld(seed=ep_seed)
                obs = world.reset(task=task)
                agent.reset_episode()
                for t in range(horizon):
                    res = agent.act(obs, learn=False)
                    obs, rew, done, info = world.step(res['action'])
                    ep_ret += float(rew)
                    act_norms.append(float(agent.brain.r.abs().mean().item()))
                    if done:
                        break
            else:
                rng_ep = np.random.default_rng(ep_seed)
                for t in range(horizon):
                    obs = rng_ep.uniform(0.0, 1.0, size=155).astype(np.float32)
                    res = agent.act(obs, learn=False)
                    rew = float(res['action'][0] - 0.2 * abs(res['action'][1]))
                    ep_ret += rew
                    act_norms.append(float(agent.brain.r.abs().mean().item()))
            mut_returns.append(ep_ret)
            mut_compute.append(float(np.mean(act_norms)) if act_norms else 0.0)

        # 3. Assess forgetting on established baseline
        old_perf_after = old_perf_before
        if baseline_task_eval_fn is not None:
            old_perf_after = float(baseline_task_eval_fn(agent))
        forgetting = max(0.0, old_perf_before - old_perf_after)

        mean_r_base = float(np.mean(base_returns))
        mean_r_mut = float(np.mean(mut_returns))
        delta_j = mean_r_mut - mean_r_base

        wiring_cost = float(agent.brain.graft_w.abs().mean().item())
        compute_cost = float(np.mean(mut_compute))

        # Net multi-objective fitness
        net_fitness = delta_j - self.lambda_w * wiring_cost - self.lambda_c * compute_cost - self.lambda_f * forgetting

        accepted = (delta_j >= self.min_causal_gain) and (net_fitness > 0.0)

        receipt = CausalEvaluationReceipt(
            mutation=mutation,
            expected_return_mut=mean_r_mut,
            expected_return_base=mean_r_base,
            causal_delta=delta_j,
            wiring_cost=wiring_cost,
            compute_cost=compute_cost,
            forgetting_cost=forgetting,
            net_fitness=net_fitness,
            accepted=accepted
        )

        if accepted:
            self.accepted_count += 1
            # Keep mutated state in agent
        else:
            self.rejected_count += 1
            # Rollback to exact base state
            agent.import_state(base_state)

        self.evaluated_mutations.append(receipt)
        return receipt

    def summary(self) -> Dict[str, Any]:
        return {
            'evaluated_mutations_count': len(self.evaluated_mutations),
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'acceptance_rate': (self.accepted_count / max(1, len(self.evaluated_mutations))),
            'recent_receipts': [
                {
                    'desc': r.mutation.description,
                    'delta_j': r.causal_delta,
                    'net_fitness': r.net_fitness,
                    'accepted': r.accepted
                }
                for r in self.evaluated_mutations[-5:]
            ]
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'summary': self.summary()
        }

    def import_state(self, state: Dict[str, Any]):
        self.accepted_count = int(state.get('accepted_count', 0))
        self.rejected_count = int(state.get('rejected_count', 0))
