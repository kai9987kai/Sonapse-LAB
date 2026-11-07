import pytest
import numpy as np

from synapse.neurowave_x2.topology_controller import CommunicationTopologyController, TopologyInvariantError


def test_topology_controller_initialization():
    controller = CommunicationTopologyController(num_regions=4, max_in_degree=2, max_out_degree=2)
    assert controller.num_regions == 4
    assert controller.topology_version == 1
    assert len(controller.routes) == 0


def test_topology_controller_add_remove_route():
    controller = CommunicationTopologyController(num_regions=4, max_in_degree=2, max_out_degree=2)
    assert controller.add_route(0, 1) is True
    assert (0, 1) in controller.routes
    assert controller.topology_version == 2

    # Duplicate add
    assert controller.add_route(0, 1) is False

    # Remove
    assert controller.remove_route(0, 1) is True
    assert (0, 1) not in controller.routes
    assert controller.topology_version == 3


def test_topology_controller_degree_limits():
    controller = CommunicationTopologyController(num_regions=4, max_in_degree=1, max_out_degree=1)
    controller.add_route(0, 1)

    # Exceeds max_out_degree for node 0
    with pytest.raises(TopologyInvariantError, match="exceeds max_out_degree"):
        controller.add_route(0, 2)

    # Exceeds max_in_degree for node 1
    with pytest.raises(TopologyInvariantError, match="exceeds max_in_degree"):
        controller.add_route(2, 1)


def test_topology_controller_self_loops():
    controller = CommunicationTopologyController(num_regions=4, allow_self_loops=False)
    with pytest.raises(TopologyInvariantError, match="Self-loop"):
        controller.add_route(1, 1)


def test_topology_controller_bounds():
    controller = CommunicationTopologyController(num_regions=4)
    with pytest.raises(TopologyInvariantError, match="out of bounds"):
        controller.add_route(-1, 2)
    with pytest.raises(TopologyInvariantError, match="out of bounds"):
        controller.add_route(0, 5)


def test_topology_controller_adjacency_and_clone():
    controller = CommunicationTopologyController(num_regions=3)
    controller.add_route(0, 1)
    controller.add_route(1, 2)
    adj = controller.get_adjacency_matrix()
    expected = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    assert np.allclose(adj, expected)

    # Clone
    clone = controller.clone()
    assert clone.routes == controller.routes
    assert clone.topology_version == controller.topology_version
    clone.add_route(2, 0)
    assert (2, 0) not in controller.routes
