import pytest

from poseidon.tessera import TesseraMacroCommons


def test_tessera_quarantine_and_ratification():
    commons = TesseraMacroCommons(current_epoch=1)
    seq = [0, 1, 4, 5]  # Rest, Up, Right, Forage
    
    # 1. First proposal from lineage A -> enters quarantine
    res_a = commons.propose_candidate(lineage_id="lineage_seed_101", actions=seq, delta_r=1.2)
    assert res_a is None
    assert len(commons.quarantine) == 1
    assert len(commons.ratified) == 0
    
    # 2. Second proposal of same sequence from SAME lineage -> stays in quarantine
    res_a_dup = commons.propose_candidate(lineage_id="lineage_seed_101", actions=seq, delta_r=1.5)
    assert res_a_dup is None
    assert len(commons.ratified) == 0
    
    # 3. Third proposal from INDEPENDENT lineage B with positive delta -> RATIFIED!
    res_b = commons.propose_candidate(lineage_id="lineage_seed_202", actions=seq, delta_r=0.9)
    assert res_b is not None
    assert res_b.startswith("OP_")
    assert len(commons.ratified) == 1
    assert len(commons.quarantine) == 0
    assert commons.ratified[res_b]["ratifiers"] == ["lineage_seed_101", "lineage_seed_202"]


def test_tessera_finite_domain_safety():
    commons = TesseraMacroCommons()
    # High-cost movement sequence
    seq = [2, 2, 2, 5, 2]
    
    # Low stamina (0.1) should fail safety check
    assert not commons.validate_safety(seq, current_stamina=0.10, current_depth=0.5)
    
    # High depth (0.85) moving down should fail
    assert not commons.validate_safety(seq, current_stamina=0.90, current_depth=0.85)
    
    # Safe condition
    assert commons.validate_safety([0, 1, 0], current_stamina=0.50, current_depth=0.3)


def test_tessera_opcode_expiration():
    commons = TesseraMacroCommons(current_epoch=1)
    commons.propose_candidate("l1", [1, 2], 1.0)
    op = commons.propose_candidate("l2", [1, 2], 1.0)
    assert op in commons.ratified
    
    # Advance epochs beyond expiration limit (3 epochs)
    commons.advance_epoch()  # epoch 2
    commons.advance_epoch()  # epoch 3
    commons.advance_epoch()  # epoch 4
    expired = commons.advance_epoch()  # epoch 5 (gap = 4 > 3)
    assert op in expired
    assert op not in commons.ratified
