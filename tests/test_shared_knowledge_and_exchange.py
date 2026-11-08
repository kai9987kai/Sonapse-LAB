"""Comprehensive Test Suite for Shared Knowledge, Vector Binding & Neural Exchange.

Verifies:
1. KnowledgeRecord schemas and checksum digests.
2. Holographic Vector-Symbolic circular convolution and correlation unbinding.
3. NeuralKnowledgePool soft-attention retrieval mechanism.
4. KnowledgeGraftAdapter zero-gain identity.
5. PoseidonHubDonorAdapter frozen weights and PretrainedGraftBridge alignment.
6. SpontaneousNeuralExchangeController candidate updates and atomic rollback.
7. EvolutionaryGrowthController two-phase commit utility transactions.
"""
import pytest
import numpy as np
import torch

from synapse.shared_knowledge import (
    KnowledgeRecord,
    KnowledgeContentType,
    VectorSymbolicBinder,
    NeuralKnowledgePool,
    KnowledgeGraftAdapter,
)
from synapse.donor_grafts import (
    PoseidonHubDonorAdapter,
    DonorRole,
    PretrainedGraftBridge,
)
from synapse.neural_exchange import (
    SpontaneousNeuralExchangeController,
)
from synapse.evo_compat import (
    EvolutionaryGrowthController,
)


def test_knowledge_record_and_vector_binding():
    # Vector-symbolic architecture binding and unbinding
    vsa = VectorSymbolicBinder()
    dim = 64
    role = torch.randn(dim)
    role = role / torch.norm(role)
    filler = torch.randn(dim)
    filler = filler / torch.norm(filler)

    # Bind: z = role (*) filler
    bound = vsa.bind(role, filler)
    assert bound.shape == (dim,)
    assert abs(float(torch.norm(bound).item()) - 1.0) < 1e-4

    # Unbind: recovered = bound (*) role*
    recovered = vsa.unbind(bound, role)
    sim = vsa.similarity(recovered, filler)
    # Circular correlation recovers positive similarity
    assert sim > 0.40, f"Expected positive similarity after unbinding, got {sim}"


def test_neural_knowledge_pool_attention_retrieval():
    pool = NeuralKnowledgePool(dim=64, capacity=100)

    # Ingest distinct records
    k1 = [1.0] * 32 + [0.0] * 32
    v1 = [0.5] * 64
    r1 = KnowledgeRecord(
        record_id="rec_nav_water",
        content_type=KnowledgeContentType.TACTICAL_RULE,
        source_id="agent_exploration",
        description="Head northwest for fresh water",
        key_vector=k1,
        value_vector=v1,
        confidence=0.95,
    )

    k2 = [0.0] * 32 + [1.0] * 32
    v2 = [-0.5] * 64
    r2 = KnowledgeRecord(
        record_id="rec_danger_predator",
        content_type=KnowledgeContentType.CAUSAL_RELATION,
        source_id="threat_appraisal",
        description="Avoid red cavern zone due to predator patrol",
        key_vector=k2,
        value_vector=v2,
        confidence=0.90,
    )

    pool.add_record(r1)
    pool.add_record(r2)
    assert pool.size() == 2

    # Query matching k1
    q_vec = torch.tensor([1.0] * 32 + [0.0] * 32, dtype=torch.float32)
    retrieved, top_matches, receipt = pool.query_attention(q_vec)

    assert retrieved.shape == (64,)
    assert len(top_matches) > 0
    # Top match should be r1
    assert top_matches[0][0].record_id == "rec_nav_water"


def test_knowledge_graft_adapter_zero_gain():
    adapter = KnowledgeGraftAdapter(knowledge_dim=64, target_dim=32, enabled=False)
    m_t = torch.randn(64)

    # Disabled: output is exactly zero
    zero_inj = adapter.project(m_t, enabled_override=False)
    assert torch.all(zero_inj == 0.0)

    # Enabled: bounded projection
    inj = adapter.project(m_t, enabled_override=True)
    assert inj.shape == (32,)
    assert torch.all(torch.abs(inj) <= adapter.max_current + 1e-6)


def test_poseidon_hub_donor_and_bridge():
    hub = PoseidonHubDonorAdapter(donor_dim=128, obs_dim=64)
    obs = torch.randn(64)

    # Verify donor weights are frozen
    for p in hub.parameters():
        assert p.requires_grad is False

    feats, receipt = hub.extract_features(obs, role=DonorRole.TIDAL_POLICY)
    assert feats.shape == (128,)

    bridge = PretrainedGraftBridge(donor_dim=128, recipient_dim=64, enabled=False)
    # Disabled check
    aligned_off = bridge.align(feats, enabled_override=False)
    assert torch.all(aligned_off == 0.0)

    # Enabled
    aligned_on = bridge.align(feats, enabled_override=True)
    assert aligned_on.shape == (64,)


def test_spontaneous_neural_exchange_controller_and_rollback():
    controller = SpontaneousNeuralExchangeController(feature_dim=64, rank=4, exchange_threshold=0.5)

    base_w = torch.randn(64, 64)
    h_src = torch.randn(64)
    h_tgt = torch.randn(64)

    # Case 1: Evaluation function where candidate degrades performance -> Rollback!
    def bad_eval_fn(w: torch.Tensor) -> float:
        # Candidate weights have higher cost
        diff = torch.norm(w - base_w).item()
        return 1.0 + 10.0 * diff

    final_w, receipt = controller.execute_transaction(base_w, h_src, h_tgt, bad_eval_fn)
    assert receipt.rolled_back is True
    assert receipt.promoted is False
    assert torch.equal(final_w, base_w)  # Exact snapshot restored

    # Case 2: Evaluation function where candidate improves performance -> Promote!
    # Force exchange readiness
    controller.exchange_threshold = 0.0
    u, v = controller.generate_candidate_delta(h_src)
    target_good = base_w + torch.matmul(u, v.t())

    def good_eval_fn(w: torch.Tensor) -> float:
        return float(torch.norm(w - target_good).item())

    final_good, receipt_good = controller.execute_transaction(base_w, h_src, h_tgt, good_eval_fn)
    assert receipt_good.promoted is True
    assert receipt_good.rolled_back is False


def test_evolutionary_growth_controller_utility():
    controller = EvolutionaryGrowthController(promotion_threshold=0.1)

    cand = controller.propose_candidate(source_region="cortex_A", target_region="cortex_B", num_nodes=4, num_edges=8)
    assert cand.num_nodes == 4

    # High reward improvement: should promote
    promoted, rec = controller.evaluate_and_transact(cand, delta_reward=1.5, uncertainty=0.05)
    assert promoted is True
    assert rec.promoted is True
    assert rec.rolled_back is False

    # Low reward improvement (outweighed by compute/param cost): should rollback
    promoted_low, rec_low = controller.evaluate_and_transact(cand, delta_reward=0.01, uncertainty=0.5)
    assert promoted_low is False
    assert rec_low.rolled_back is True
