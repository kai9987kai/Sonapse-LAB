"""Unit tests for the Memory Carrier Observatory (MCO)."""
import copy
import json
import pytest

from poseidon.mco import (
    CARRIERS,
    CONDITIONS,
    STANDARD_CORPUS,
    STANDARD_TASKS,
    SUBSETS,
    build_standard_bank,
    evaluate_task_trial,
    run_mco_experiment,
    verify_mco_receipt,
)
from poseidon.memory import MemoryBank


def test_standard_corpus_and_tasks_integrity():
    assert len(STANDARD_CORPUS) == 32
    assert len(STANDARD_TASKS) == 16
    assert len(SUBSETS) == 16
    assert set(CONDITIONS) == {"intact", "shuffled", "irrelevant", "erased"}

    # Each carrier has exactly 8 facts and 4 tasks
    for carrier in CARRIERS:
        facts = [f for f in STANDARD_CORPUS if f["carrier"] == carrier]
        assert len(facts) == 8
        tasks = [t for t in STANDARD_TASKS if t["carrier"] == carrier]
        assert len(tasks) == 4


def test_standard_bank_indexing_and_retrieval():
    bank = build_standard_bank()
    assert len(bank) == 32

    # Query for tidal surge fact in episodic carrier
    hits = bank.retrieve("What occurred at tick 42 at the northern reef?", top_k=3, mode="hybrid")
    assert len(hits) > 0
    assert hits[0]["metadata"]["fact_id"] == "fact-ep-01"


def test_evaluation_trial_controls():
    bank = build_standard_bank()
    task = STANDARD_TASKS[0]  # episodic task

    # 1. Intact with episodic carrier active -> high hit and grounded score
    res_intact = evaluate_task_trial(bank, task, subset=("episodic",), condition="intact")
    assert res_intact["hit_at_1"] == 1
    assert res_intact["grounded_score"] > 0.5
    assert res_intact["target_retrieved"] is True

    # 2. Intact with episodic carrier disabled -> 0 hit and empty context
    res_disabled = evaluate_task_trial(bank, task, subset=("body", "habitat"), condition="intact")
    assert res_disabled["hit_at_1"] == 0
    assert res_disabled["hit_at_3"] == 0
    assert res_disabled["grounded_score"] == 0.0

    # 3. Erased condition -> 0 hits
    res_erased = evaluate_task_trial(bank, task, subset=("episodic", "body"), condition="erased")
    assert res_erased["retrieved_count"] == 0
    assert res_erased["hit_at_3"] == 0

    # 4. Irrelevant condition -> distractors retrieved, 0 target hit
    res_irrel = evaluate_task_trial(bank, task, subset=("episodic",), condition="irrelevant")
    assert res_irrel["retrieved_count"] > 0
    assert res_irrel["hit_at_3"] == 0

    # 5. Shuffled condition -> swapped texts retrieved
    res_shuff = evaluate_task_trial(bank, task, subset=("episodic",), condition="shuffled")
    assert res_shuff["retrieved_count"] > 0


def test_micro_mco_experiment_execution_and_verification():
    # Run a micro benchmark over 2 tasks across all 4 conditions and all 16 subsets
    micro_tasks = STANDARD_TASKS[:2]
    progress_calls = []

    receipt = run_mco_experiment(
        tasks=micro_tasks,
        conditions=CONDITIONS,
        subsets=SUBSETS,
        progress=lambda p: progress_calls.append(p),
    )

    assert receipt["schema"] == "poseidon-mco-experiment-v1"
    assert receipt["status"] == "completed"
    assert receipt["task_count"] == 2
    assert receipt["total_evaluations"] == 2 * 4 * 16  # 128 trials
    assert len(receipt["trials"]) == 128
    assert "receipt_sha256" in receipt

    # Condition summaries exist
    for cond in CONDITIONS:
        assert cond in receipt["condition_summary"]
        assert receipt["condition_summary"][cond]["trials"] == 2 * 16

    # Verify receipt offline
    verification = verify_mco_receipt(receipt)
    assert verification["verified"] is True
    assert verification["trials_verified"] == 128

    # Tampered receipt is rejected
    tampered = copy.deepcopy(receipt)
    tampered["condition_summary"]["intact"]["mean_hit_at_1"] += 0.5
    with pytest.raises(ValueError, match="Receipt checksum mismatch"):
        verify_mco_receipt(tampered)
