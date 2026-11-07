import math
import torch
import numpy as np
import pytest

from synapse.neurowave_x2.protocol_selector import ProtocolSelector
from synapse.neurowave_x2.protocol_registry import PROTOCOLS, list_protocols_up_to_level


def test_protocol_selector_initialization():
    selector = ProtocolSelector(context_dim=16, hidden_dim=16, default_protocol="vector")
    assert selector.context_dim == 16
    assert selector.default_protocol == "vector"
    assert len(selector.protocol_names) == len(PROTOCOLS)


def test_protocol_selector_forward_deterministic():
    selector = ProtocolSelector(context_dim=8, hidden_dim=16, default_protocol="vector")
    ctx = torch.randn(8)
    proto, probs, diag = selector(ctx, max_development_level=6, deterministic=True)
    assert proto in selector.protocol_names
    assert math.isclose(float(torch.sum(probs).item()), 1.0, rel_tol=1e-5)
    assert "protocol_entropy" in diag
    assert "effective_protocols" in diag
    assert diag["protocol_entropy"] >= 0.0


def test_protocol_masking_development_level():
    selector = ProtocolSelector(context_dim=8, hidden_dim=16)
    ctx = torch.randn(8)

    # At level 0, only 'no_comm' and 'scalar' should be possible
    proto, probs, diag = selector(ctx, max_development_level=0, deterministic=True)
    level_0_allowed = set(list_protocols_up_to_level(0))
    assert proto in level_0_allowed

    # Verify disallowed protocol probabilities are zeroed out (under softmax threshold)
    for i, p_name in enumerate(selector.protocol_names):
        if p_name not in level_0_allowed:
            assert probs[i].item() < 1e-6


def test_protocol_selector_gumbel_mode():
    selector = ProtocolSelector(context_dim=8, hidden_dim=16, mode="gumbel")
    selector.train()
    ctx = torch.randn(8)
    proto, probs, diag = selector(ctx, max_development_level=6, deterministic=False)
    assert proto in selector.protocol_names
    assert len(probs) == len(PROTOCOLS)
