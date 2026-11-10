"""Regression contract: fast candidate ranking == legacy full-argsort visit order."""
import numpy as np
import pytest

from synapse.neurogenesis import ConnectomeNeurogenesisEngine


def _legacy_sequence(cov, threshold=0.01):
    n = cov.shape[0]
    c = cov.copy()
    np.fill_diagonal(c, -1.0)
    out = []
    for idx in np.argsort(c.ravel())[::-1]:
        if c.ravel()[idx] < threshold:
            break
        out.append(int(idx))
    return out


def _assert_equivalent(fast, legacy, flat):
    """Same candidates, same descending value sequence; index order may differ
    only among exactly-tied values (legacy argsort is unstable, so its tie
    order was itself arbitrary)."""
    assert len(fast) == len(legacy)
    assert set(fast) == set(legacy)
    assert np.array_equal(flat[fast], flat[legacy])
    for a, b in zip(fast, legacy):
        if a != b:
            assert flat[a] == flat[b]


@pytest.mark.parametrize("n,scale,seed", [(40, 0.02, 0), (300, 0.01, 1), (600, 0.004, 2), (50, 0.0001, 3)])
def test_ranked_candidates_matches_legacy_order(n, scale, seed):
    rng = np.random.default_rng(seed)
    cov = (rng.standard_normal((n, n)) * scale).astype(np.float32)
    np.fill_diagonal(cov, 5.0)  # large diagonal must still be excluded
    fast = list(ConnectomeNeurogenesisEngine._ranked_candidates(cov, n, threshold=0.01))
    _assert_equivalent(fast, _legacy_sequence(cov), cov.ravel())


def test_ranked_candidates_on_noncontiguous_view():
    rng = np.random.default_rng(7)
    big = (rng.standard_normal((80, 80)) * 0.02).astype(np.float32)
    view = big[:50, :50]
    fast = list(ConnectomeNeurogenesisEngine._ranked_candidates(view, 50, threshold=0.01))
    dense = np.ascontiguousarray(view)
    _assert_equivalent(fast, _legacy_sequence(dense), dense.ravel())


def test_structural_remodeling_end_to_end_unchanged():
    """Paired run: legacy algorithm vs current produce identical graft edits."""
    import copy
    import torch
    from synapse.config import SynapseConfig
    from synapse.agent import IntegratedAgent

    torch.manual_seed(5)
    agent = IntegratedAgent(SynapseConfig(seed=5, nodes_per_source=16))
    eng = agent.neurogenesis
    n = eng.num_neurons
    rng = np.random.default_rng(11)
    for _ in range(20):
        eng.record_activity(rng.random(n).astype(np.float32))

    agent_legacy = copy.deepcopy(agent)
    eng_legacy = agent_legacy.neurogenesis

    def legacy_rank(cov_mat, n_, threshold=0.01, head=256):
        yield from _legacy_sequence(np.ascontiguousarray(cov_mat), threshold)

    eng_legacy._ranked_candidates = legacy_rank

    r_new = eng.evaluate_structural_remodeling(agent, dopamine=1.0, astrocyte_calcium=1.0, threat=0.0)
    r_old = eng_legacy.evaluate_structural_remodeling(agent_legacy, dopamine=1.0, astrocyte_calcium=1.0, threat=0.0)
    assert r_new['sprouted'] == r_old['sprouted']
    assert r_new['pruned'] == r_old['pruned']
    assert torch.equal(agent.brain.graft_w, agent_legacy.brain.graft_w)
    assert list(agent.graph.graft_pre) == list(agent_legacy.graph.graft_pre)
    assert list(agent.graph.graft_post) == list(agent_legacy.graph.graft_post)
