import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from synapse.neurowave_x2.topology_controller import CommunicationTopologyController
from synapse.neurowave_x2.spectral_basis import SpectralBasisManager
from synapse.neurowave_x2.topology_transaction import TopologyTransaction, TopologyTransactionError


def test_topology_transaction_commit():
    controller = CommunicationTopologyController(num_regions=4)
    controller.add_route(0, 1)
    spec_mgr = SpectralBasisManager(num_modes=2)
    model = nn.Linear(4, 4)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Initial dummy step to populate optimizer state
    loss = model(torch.randn(1, 4)).sum()
    loss.backward()
    opt.step()

    tx = TopologyTransaction(controller, spec_mgr, model, opt)
    tx.mutate_add_route(1, 2)
    assert (1, 2) in controller.routes

    tx.validate_and_reconcile()
    tx.commit()

    assert tx.committed is True
    assert (1, 2) in controller.routes

    # Commit after commit should be no-op or error
    tx.commit()
    with pytest.raises(TopologyTransactionError):
        tx.rollback()


def test_topology_transaction_rollback():
    controller = CommunicationTopologyController(num_regions=4)
    controller.add_route(0, 1)
    orig_routes = set(controller.routes)
    orig_version = controller.topology_version

    spec_mgr = SpectralBasisManager(num_modes=2)
    adj = controller.get_adjacency_matrix()
    spec_mgr.get_or_compute_basis(adj)
    orig_spec_version = spec_mgr.basis_version

    model = nn.Linear(4, 4)
    orig_weight = model.weight.data.clone()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    tx = TopologyTransaction(controller, spec_mgr, model, opt)
    tx.mutate_add_route(1, 2)
    # Alter model weights during mutation attempt
    model.weight.data.add_(1.0)

    # Roll back
    tx.rollback(reason="Simulated validation failure")

    assert tx.rolled_back is True
    assert controller.routes == orig_routes
    assert controller.topology_version == orig_version
    assert spec_mgr.basis_version == orig_spec_version
    assert torch.allclose(model.weight.data, orig_weight)

    with pytest.raises(TopologyTransactionError):
        tx.commit()
