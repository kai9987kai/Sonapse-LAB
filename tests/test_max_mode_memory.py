"""Regression coverage for large Max Mode allocations."""

import numpy as np
import pytest
from scipy import sparse

from synapse.aether_graft_bridge import AetherGraftBridge, GraftBridgeSpec, compute_graph_modes
from synapse.adaptation_coordinator import AdaptationCoordinator
from synapse.neurogenesis import ConnectomeNeurogenesisEngine
from synapse.spectral import compute_connectome_harmonics
from synapse.tandem_max import _ensure_max_checkpoint_space, _max_checkpoint_reserve_bytes


def test_max_mode_checkpoint_preflight_reports_low_disk_space():
    required = _max_checkpoint_reserve_bytes(9000)

    with pytest.raises(RuntimeError, match='checkpoint requires'):
        _ensure_max_checkpoint_space('unused-test-path', 9000, free_bytes=required - 1)

    result = _ensure_max_checkpoint_space('unused-test-path', 9000, free_bytes=required)
    assert result['required_free_bytes'] == required
    assert result['available_free_bytes'] == required


def test_neurogenesis_covariance_update_avoids_full_outer_temporary(monkeypatch):
    engine = ConnectomeNeurogenesisEngine(num_neurons=19, seed=13)
    engine.rate_mean[:] = np.linspace(0.0, 0.2, 19, dtype=np.float32)
    engine.rate_cov[:] = np.arange(19 * 19, dtype=np.float32).reshape(19, 19) / 1000
    rates = np.linspace(0.3, 0.9, 19, dtype=np.float32)

    previous_mean = engine.rate_mean.copy()
    previous_cov = engine.rate_cov.copy()
    delta = rates - previous_mean
    expected_mean = previous_mean + 0.05 * delta
    expected_cov = engine.covariance_decay * previous_cov + (1.0 - engine.covariance_decay) * np.outer(delta, delta)

    def reject_dense_outer(*_args, **_kwargs):
        raise AssertionError("large covariance update must use bounded row blocks")

    monkeypatch.setattr(np, "outer", reject_dense_outer)
    engine.record_activity(rates)

    np.testing.assert_array_equal(engine.rate_mean, expected_mean)
    np.testing.assert_allclose(engine.rate_cov, expected_cov, rtol=0, atol=1e-7)


def test_adaptation_noise_is_generated_in_bounded_blocks_without_rng_drift():
    neuron_count = 512
    seed = 29
    novelty = 0.37
    coordinator = AdaptationCoordinator(num_neurons=neuron_count, seed=seed)
    coordinator.fast_weights[:] = np.linspace(-0.1, 0.1, neuron_count * neuron_count, dtype=np.float32).reshape(
        neuron_count, neuron_count
    )

    class BoundedRng:
        def __init__(self):
            self.inner = np.random.default_rng(seed)
            self.shapes = []

        def standard_normal(self, shape):
            shape = tuple(shape)
            self.shapes.append(shape)
            assert shape != (neuron_count, neuron_count), "must not allocate a full-size normal matrix"
            return self.inner.standard_normal(shape)

    bounded_rng = BoundedRng()
    coordinator.rng = bounded_rng

    reference_rng = np.random.default_rng(seed)
    expected = coordinator.fast_weights.copy() * 0.95
    expected += novelty * 0.01 * reference_rng.standard_normal((neuron_count, neuron_count)).astype(np.float32)

    result = coordinator.step(novelty=novelty)

    assert len(bounded_rng.shapes) > 1
    assert max(rows for rows, _ in bounded_rng.shapes) < neuron_count
    np.testing.assert_array_equal(coordinator.fast_weights, expected)
    assert result["fast_weight_norm"] == float(np.linalg.norm(expected))
    assert bounded_rng.inner.bit_generator.state == reference_rng.bit_generator.state


def test_large_connectome_harmonics_use_sparse_eigensolver(monkeypatch):
    neuron_count = 3000
    edges = np.arange(neuron_count - 1, dtype=np.int32)

    class ChainGraph:
        n = neuron_count
        base_pre = edges
        base_post = edges + 1
        base_w = np.ones(neuron_count - 1, dtype=np.float32)
        graft_pre = np.empty(0, dtype=np.int32)
        graft_post = np.empty(0, dtype=np.int32)

    def reject_dense_eigh(*_args, **_kwargs):
        raise AssertionError("large connectome harmonics must not use a dense eigendecomposition")

    monkeypatch.setattr(np.linalg, "eigh", reject_dense_eigh)
    harmonics = compute_connectome_harmonics(ChainGraph(), num_modes=8)

    assert harmonics["num_neurons"] == neuron_count
    assert harmonics["num_modes"] == 8
    assert harmonics["eigenvectors"].shape == (neuron_count, 8)
    assert harmonics["eigenvalues"] == sorted(harmonics["eigenvalues"])


def test_large_aether_recipient_graph_is_generated_sparse():
    bridge = AetherGraftBridge(spec=GraftBridgeSpec(enabled=True), donor_dim=32, recipient_dim=32)
    graph = bridge._generate_biological_connectome_graph(3000)

    assert sparse.issparse(graph)
    assert graph.shape == (3000, 3000)
    assert graph.nnz < 3000 * 128
    assert (graph != graph.T).nnz == 0


def test_aether_graph_modes_use_sparse_low_mode_solver(monkeypatch):
    neuron_count = 3000
    adjacency = sparse.diags(
        [np.ones(neuron_count - 1), np.ones(neuron_count - 1)],
        offsets=[-1, 1],
        shape=(neuron_count, neuron_count),
        format="csr",
    )

    def reject_dense_eigh(*_args, **_kwargs):
        raise AssertionError("large AETHER modal bases must not use dense eigendecomposition")

    monkeypatch.setattr(np.linalg, "eigh", reject_dense_eigh)
    eigenvalues, eigenvectors = compute_graph_modes(adjacency, mode_count=8)

    assert eigenvalues.shape == (8,)
    assert eigenvectors.shape == (neuron_count, 8)
    assert np.all(np.diff(eigenvalues) >= -1e-10)
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(8), atol=1e-6)
