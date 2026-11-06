import numpy as np
from synapse.manifold_alignment import (
    compute_graph_distance_matrix,
    gromov_wasserstein_discrepancy,
    align_subcircuit_gromov_wasserstein,
)

def test_graph_distance_matrix():
    adj = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    D = compute_graph_distance_matrix(adj)
    assert D.shape == (3, 3)
    assert D[0, 0] == 0.0
    assert D[0, 1] < D[0, 2]

def test_gromov_wasserstein_alignment():
    # Donor 4-node ring
    donor = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
    ])
    # Host 4-node ring
    host = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
    ])
    res = align_subcircuit_gromov_wasserstein(donor, host)
    assert res['status'] == 'ok'
    assert res['donor_nodes'] == 4
    assert res['host_nodes'] == 4
    assert res['gromov_wasserstein_distortion'] >= 0.0
    assert len(res['coupling']) == 4
    assert len(res['coupling'][0]) == 4
    # Coupling should sum close to 1
    total_transport = sum(sum(row) for row in res['coupling'])
    assert abs(total_transport - 1.0) < 0.05
