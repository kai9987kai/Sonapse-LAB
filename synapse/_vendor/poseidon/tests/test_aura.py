import pytest
import numpy as np

from poseidon.aura import AuraController
from poseidon.tessera import TesseraMacroCommons


class DummyCore:
    def act(self, obs):
        return 1  # Always UP


def test_aura_controller_decision_trace():
    core = DummyCore()
    aura = AuraController(core=core)
    aura.reset(scarcity=2.0)
    
    # Standard observation
    obs = [0.8, 0.8, 0.8, 0.5, 0.1, 0.2, 0.3, 0.0, 0.8, 0.0, 0.9, 0.0, 2, 0.2, 0.5, 1]
    decision = aura.plan(obs)
    
    assert "action" in decision
    assert 0 <= decision["action"] <= 5
    assert "pva_heading" in decision
    assert "pva_coherence" in decision
    assert decision["winning_channel"] in ("harvester", "sentinel", "escaper")
    assert decision["confidence"] > 0.0
    assert not decision["disoriented"]


def test_aura_controller_executes_ratified_tessera_opcode():
    core = DummyCore()
    tessera = TesseraMacroCommons()
    # Ratify an opcode with high delta_r
    tessera.propose_candidate("l1", [4, 5], 2.5)
    op = tessera.propose_candidate("l2", [4, 5], 2.5)
    assert op is not None
    
    aura = AuraController(core=core, tessera=tessera)
    aura.reset(scarcity=2.5)
    
    # Safe observation with high stamina
    obs = [0.9, 0.9, 0.9, 0.2, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.9, 0.0, 1, 0.1, 0.8, 1]
    d1 = aura.plan(obs)
    assert d1["action_source"] == "tessera_macro"
    assert d1["action"] == 4  # First action of [4, 5]
    
    d2 = aura.plan(obs)
    assert d2["action_source"] == "tessera_macro"
    assert d2["action"] == 5  # Second action of [4, 5]
