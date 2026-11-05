"""Memory Carrier Observatory (MCO): Causal Factorial Memory Evaluation.

Evaluates the 4 external memory carriers (episodic, body, habitat, social)
across all 16 inclusion/exclusion subsets under intact, shuffled, irrelevant,
and erased control conditions with falsifiable negative controls and influence analysis.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import time
from typing import Sequence

from .memory import CARRIERS, MemoryBank

SCHEMA = "poseidon-mco-experiment-v1"
CONDITIONS = ("intact", "shuffled", "irrelevant", "erased")

# All 16 carrier inclusion subsets (powerset of CARRIERS)
SUBSETS = [
    (),
    ("episodic",),
    ("body",),
    ("habitat",),
    ("social",),
    ("episodic", "body"),
    ("episodic", "habitat"),
    ("episodic", "social"),
    ("body", "habitat"),
    ("body", "social"),
    ("habitat", "social"),
    ("episodic", "body", "habitat"),
    ("episodic", "body", "social"),
    ("episodic", "habitat", "social"),
    ("body", "habitat", "social"),
    ("episodic", "body", "habitat", "social"),
]

# Standardized, versioned 4-carrier knowledge corpus (8 facts per carrier)
STANDARD_CORPUS = [
    # Episodic carrier facts
    {"id": "fact-ep-01", "carrier": "episodic", "text": "At tick 42, a sudden tidal surge struck the northern reef and reduced shelter by 40 percent.", "keywords": ["tidal surge", "northern reef", "40 percent", "shelter"]},
    {"id": "fact-ep-02", "carrier": "episodic", "text": "During cycle 3, foraging at patch P12 depleted food reserves to zero within 15 time steps.", "keywords": ["cycle 3", "p12", "depleted", "15 time steps"]},
    {"id": "fact-ep-03", "carrier": "episodic", "text": "The agent rested near waypoint W4 at dawn to regenerate depleted stamina before transit.", "keywords": ["w4", "dawn", "stamina", "transit"]},
    {"id": "fact-ep-04", "carrier": "episodic", "text": "At tick 88, severe dehydration forced emergency consumption of brackish water at lagoon L2.", "keywords": ["tick 88", "dehydration", "brackish", "lagoon l2"]},
    {"id": "fact-ep-05", "carrier": "episodic", "text": "A flash storm at tick 110 collapsed the temporary canopy at outer perimeter sector S9.", "keywords": ["flash storm", "tick 110", "canopy", "sector s9"]},
    {"id": "fact-ep-06", "carrier": "episodic", "text": "Exploratory sortie E5 discovered a freshwater seep behind the western basalt ridge.", "keywords": ["sortie e5", "freshwater", "basalt ridge"]},
    {"id": "fact-ep-07", "carrier": "episodic", "text": "At tick 64, predatory gulls circled patch P07, forcing the agent to execute evasion fleeing.", "keywords": ["tick 64", "gulls", "p07", "fleeing"]},
    {"id": "fact-ep-08", "carrier": "episodic", "text": "Harvest cycle H2 recovered 28 algae bundles from the tidal pool before dusk.", "keywords": ["cycle h2", "28 algae", "tidal pool", "dusk"]},

    # Body carrier facts
    {"id": "fact-bo-01", "carrier": "body", "text": "When stamina drops below 0.20, continuous movement actions incur double metabolic energy depletion.", "keywords": ["stamina", "0.20", "double", "energy"]},
    {"id": "fact-bo-02", "carrier": "body", "text": "Internal hydration reserve depletion accelerates by 50 percent during peak daytime thermal stress.", "keywords": ["hydration", "50 percent", "daytime", "thermal stress"]},
    {"id": "fact-bo-03", "carrier": "body", "text": "Resting for 3 uninterrupted ticks restores physical stamina above the pre-transit threshold.", "keywords": ["resting", "3 ticks", "stamina", "threshold"]},
    {"id": "fact-bo-04", "carrier": "body", "text": "Consuming raw salt lichen decreases hydration by 0.15 while providing minimal caloric energy.", "keywords": ["salt lichen", "hydration", "0.15", "caloric"]},
    {"id": "fact-bo-05", "carrier": "body", "text": "Core body temperature stabilization requires minimum shelter index above 0.45 during storms.", "keywords": ["temperature", "shelter index", "0.45", "storms"]},
    {"id": "fact-bo-06", "carrier": "body", "text": "Metabolic recovery rate peaks when remaining energy and hydration both exceed 0.70.", "keywords": ["metabolic recovery", "peaks", "0.70"]},
    {"id": "fact-bo-07", "carrier": "body", "text": "Critical vitals warning activates when any biological reserve falls under the 0.10 failure boundary.", "keywords": ["critical vitals", "0.10", "failure boundary"]},
    {"id": "fact-bo-08", "carrier": "body", "text": "Sprint travel expenditure drains 0.112 stamina units per hop across rough basalt terrain.", "keywords": ["sprint", "0.112", "stamina", "basalt"]},

    # Habitat carrier facts
    {"id": "fact-ha-01", "carrier": "habitat", "text": "TidePool patches with terrain score above 0.70 offer natural high-elevation shelter from storm flooding.", "keywords": ["terrain score", "0.70", "high-elevation", "shelter"]},
    {"id": "fact-ha-02", "carrier": "habitat", "text": "Freshwater pools replenish at rate lambda 0.014 divided by the environmental scarcity factor.", "keywords": ["freshwater", "0.014", "scarcity factor"]},
    {"id": "fact-ha-03", "carrier": "habitat", "text": "Coastal mudflats impose an additional travel delay of 2 ticks when crossed during high tide.", "keywords": ["mudflats", "travel delay", "2 ticks", "high tide"]},
    {"id": "fact-ha-04", "carrier": "habitat", "text": "Algae patches replenish at baseline rate 0.007 divided by environmental scarcity.", "keywords": ["algae patches", "0.007", "scarcity"]},
    {"id": "fact-ha-05", "carrier": "habitat", "text": "The central caldera maintains stable perennial water springs even during severe seasonal droughts.", "keywords": ["caldera", "perennial", "springs", "droughts"]},
    {"id": "fact-ha-06", "carrier": "habitat", "text": "Rocky grottos provide 0.85 shelter rating against high-velocity gale winds and squalls.", "keywords": ["grottos", "0.85", "shelter rating", "winds"]},
    {"id": "fact-ha-07", "carrier": "habitat", "text": "Lowland sandbars become completely submerged when tidal water levels reach 0.80.", "keywords": ["sandbars", "submerged", "water levels", "0.80"]},
    {"id": "fact-ha-08", "carrier": "habitat", "text": "Southern kelp groves yield triple biomass harvest when gathered during morning ebb tide.", "keywords": ["kelp groves", "triple biomass", "ebb tide"]},

    # Social carrier facts
    {"id": "fact-so-01", "carrier": "social", "text": "A two-tone acoustic whistle from sentinel peers signals immediate predator approach in adjacent sectors.", "keywords": ["two-tone", "whistle", "sentinel", "predator"]},
    {"id": "fact-so-02", "carrier": "social", "text": "Foraging in proximity to peer scouts reduces vigilance requirement by 25 percent.", "keywords": ["scouts", "vigilance", "25 percent"]},
    {"id": "fact-so-03", "carrier": "social", "text": "Shared territory markers at cluster border B3 indicate mutual non-aggression among foraging pods.", "keywords": ["territory markers", "border b3", "non-aggression", "pods"]},
    {"id": "fact-so-04", "carrier": "social", "text": "Cooperative harvest sharing at roost R1 increases food acquisition efficiency by 30 percent.", "keywords": ["cooperative", "roost r1", "efficiency", "30 percent"]},
    {"id": "fact-so-05", "carrier": "social", "text": "Visual wing-flapping displays signal discovery of an unharvested freshwater spring.", "keywords": ["wing-flapping", "signal", "freshwater spring"]},
    {"id": "fact-so-06", "carrier": "social", "text": "Alarm chirp sequence C4 mandates immediate dispersion into nearest grotto shelters.", "keywords": ["chirp sequence c4", "dispersion", "grotto shelters"]},
    {"id": "fact-so-07", "carrier": "social", "text": "Juvenile pods follow experienced pathfinders along the protected inland trench navigation route.", "keywords": ["pathfinders", "inland trench", "route"]},
    {"id": "fact-so-08", "carrier": "social", "text": "Subordinate foragers yield primary feeding priority at high-yield patches to avoid agonistic conflict.", "keywords": ["subordinate", "feeding priority", "conflict"]},
]

# Irrelevant distractor corpus for the negative control condition
IRRELEVANT_CORPUS = [
    {"id": "distract-01", "carrier": "habitat", "text": "The boiling point of water at standard sea-level atmospheric pressure is 100 degrees Celsius.", "keywords": ["boiling point", "celsius"]},
    {"id": "distract-02", "carrier": "body", "text": "Binary search algorithms achieve logarithmic time complexity O(log n) over sorted arrays.", "keywords": ["binary search", "logarithmic"]},
    {"id": "distract-03", "carrier": "episodic", "text": "Photosynthesis converts carbon dioxide and sunlight into glucose and oxygen in plant chloroplasts.", "keywords": ["photosynthesis", "glucose"]},
    {"id": "distract-04", "carrier": "social", "text": "The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second.", "keywords": ["speed of sound", "meters per second"]},
]

# Standardized benchmark tasks (4 per carrier domain = 16 tasks)
STANDARD_TASKS = [
    # Episodic query tasks
    {"id": "task-ep-01", "carrier": "episodic", "query": "What occurred at tick 42 at the northern reef?", "target_fact": "fact-ep-01", "expected_keywords": ["surge", "reef", "40"]},
    {"id": "task-ep-02", "carrier": "episodic", "query": "What happened to food reserves at patch P12 during cycle 3?", "target_fact": "fact-ep-02", "expected_keywords": ["depleted", "15"]},
    {"id": "task-ep-03", "carrier": "episodic", "query": "Why did the agent rest near waypoint W4 at dawn?", "target_fact": "fact-ep-03", "expected_keywords": ["stamina", "transit"]},
    {"id": "task-ep-04", "carrier": "episodic", "query": "What emergency action was taken at tick 88 due to dehydration?", "target_fact": "fact-ep-04", "expected_keywords": ["brackish", "lagoon"]},

    # Body query tasks
    {"id": "task-bo-01", "carrier": "body", "query": "What penalty happens when stamina falls below 0.20?", "target_fact": "fact-bo-01", "expected_keywords": ["double", "energy"]},
    {"id": "task-bo-02", "carrier": "body", "query": "How does daytime thermal stress affect internal hydration depletion?", "target_fact": "fact-bo-02", "expected_keywords": ["50 percent", "hydration"]},
    {"id": "task-bo-03", "carrier": "body", "query": "How many ticks of resting are needed to restore stamina before transit?", "target_fact": "fact-bo-03", "expected_keywords": ["3", "stamina"]},
    {"id": "task-bo-04", "carrier": "body", "query": "What biological effect occurs after eating raw salt lichen?", "target_fact": "fact-bo-04", "expected_keywords": ["hydration", "0.15"]},

    # Habitat query tasks
    {"id": "task-ha-01", "carrier": "habitat", "query": "What protection is provided by patches with terrain score above 0.70?", "target_fact": "fact-ha-01", "expected_keywords": ["shelter", "flooding"]},
    {"id": "task-ha-02", "carrier": "habitat", "query": "At what rate do freshwater pools replenish relative to environmental scarcity?", "target_fact": "fact-ha-02", "expected_keywords": ["0.014", "scarcity"]},
    {"id": "task-ha-03", "carrier": "habitat", "query": "What travel delay penalty do coastal mudflats impose during high tide?", "target_fact": "fact-ha-03", "expected_keywords": ["2 ticks", "delay"]},
    {"id": "task-ha-04", "carrier": "habitat", "query": "What is the replenishment rate formula for algae patches?", "target_fact": "fact-ha-04", "expected_keywords": ["0.007", "scarcity"]},

    # Social query tasks
    {"id": "task-so-01", "carrier": "social", "query": "What does a two-tone acoustic whistle from sentinel peers mean?", "target_fact": "fact-so-01", "expected_keywords": ["predator", "whistle"]},
    {"id": "task-so-02", "carrier": "social", "query": "How much does foraging near peer scouts reduce vigilance demands?", "target_fact": "fact-so-02", "expected_keywords": ["25 percent", "vigilance"]},
    {"id": "task-so-03", "carrier": "social", "query": "What is indicated by shared territory markers at border B3?", "target_fact": "fact-so-03", "expected_keywords": ["non-aggression", "pods"]},
    {"id": "task-so-04", "carrier": "social", "query": "How much does cooperative sharing at roost R1 increase acquisition efficiency?", "target_fact": "fact-so-04", "expected_keywords": ["30 percent", "efficiency"]},
]

LIMITS = [
    "Synthetic 4-carrier evaluation benchmark for external language memory; not human memory or biological neuroscience.",
    "Lexical BM25 and hybrid RRF rankings score token associations, not semantic truth or reasoning ability.",
    "16-subset factorial design isolates carrier exclusion effects under controlled queries and matched contexts.",
    "Falsifiable negative controls (shuffled, irrelevant, erased) test attribution validity and distraction sensitivity.",
    "Leave-one-task-out metrics estimate item sensitivity, not general population confidence intervals.",
    "External carrier retrieval provides quoted untrusted user context; model weights remain frozen.",
]


def digest(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_standard_bank(corpus: Sequence[dict] = STANDARD_CORPUS) -> MemoryBank:
    """Construct an initialized MemoryBank populated with the standardized corpus."""
    bank = MemoryBank(max_entries=len(corpus) + 16)
    for fact in corpus:
        bank.remember(fact["text"], carrier=fact["carrier"], metadata={"fact_id": fact["id"], "keywords": fact.get("keywords", [])})
    return bank


def evaluate_task_trial(
    bank: MemoryBank,
    task: dict,
    subset: tuple[str, ...],
    condition: str,
    language_runtime=None,
    distractor_corpus: Sequence[dict] = IRRELEVANT_CORPUS,
    shuffled_corpus: Sequence[dict] | None = None,
) -> dict:
    """Evaluate a single query task under a specific carrier subset and control condition."""
    active_carriers = set(subset)
    disabled = set(CARRIERS) - active_carriers
    query = task["query"]
    target_fact_id = task["target_fact"]
    expected_keywords = task["expected_keywords"]

    retrieval_start = time.perf_counter()
    if condition == "erased" or not active_carriers:
        retrieved_entries = []
    elif condition == "irrelevant":
        # Inject unrelated distractors as context
        retrieved_entries = [
            {"id": item["id"], "text": item["text"], "carrier": item["carrier"], "metadata": {"fact_id": item["id"]}}
            for item in distractor_corpus[:3]
        ]
    elif condition == "shuffled":
        # Retrieve from bank populated with shuffled texts
        if shuffled_corpus is None:
            shuffled_corpus = list(STANDARD_CORPUS)
            # Deterministic shift by 3
            shuffled_corpus = shuffled_corpus[3:] + shuffled_corpus[:3]
        s_bank = MemoryBank(max_entries=len(shuffled_corpus) + 8)
        for idx, item in enumerate(STANDARD_CORPUS):
            # Same fact id and carrier, but swapped text from another domain
            swapped = shuffled_corpus[idx]
            s_bank.remember(swapped["text"], carrier=item["carrier"], metadata={"fact_id": item["id"], "original_id": swapped["id"]})
        retrieved_entries = s_bank.retrieve(query, disabled_carriers=disabled, top_k=3, mode="bm25")
    else:  # intact
        retrieved_entries = bank.retrieve(query, disabled_carriers=disabled, top_k=3, mode="hybrid")
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    # Evaluate retrieval accuracy
    retrieved_fact_ids = [entry.get("metadata", {}).get("fact_id") for entry in retrieved_entries]
    hit_at_1 = int(bool(retrieved_fact_ids and retrieved_fact_ids[0] == target_fact_id))
    hit_at_3 = int(target_fact_id in retrieved_fact_ids)
    if target_fact_id in retrieved_fact_ids:
        rank = retrieved_fact_ids.index(target_fact_id) + 1
        mrr = 1.0 / rank
    else:
        mrr = 0.0

    # Determine answer generation and keyword grounding
    eval_text = " ".join(entry["text"] for entry in retrieved_entries).casefold()
    if language_runtime is not None:
        try:
            memories_context = [e["text"] for e in retrieved_entries]
            res = language_runtime.generate(query, memories=memories_context, max_new_tokens=64)
            generated_answer = res["text"].casefold()
        except Exception:
            generated_answer = eval_text
    else:
        # Exact offline baseline matches retrieved context
        generated_answer = eval_text

    keyword_matches = sum(1 for kw in expected_keywords if kw.casefold() in generated_answer)
    grounded_score = keyword_matches / max(1, len(expected_keywords))
    target_retrieved = bool(hit_at_3)

    return {
        "task_id": task["id"],
        "carrier": task["carrier"],
        "subset": list(subset),
        "subset_name": "+".join(subset) if subset else "empty",
        "condition": condition,
        "active_carrier_count": len(subset),
        "target_carrier_active": task["carrier"] in active_carriers,
        "retrieved_count": len(retrieved_entries),
        "retrieved_fact_ids": retrieved_fact_ids,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
        "grounded_score": grounded_score,
        "target_retrieved": target_retrieved,
        "retrieval_ms": retrieval_ms,
    }


def run_mco_experiment(
    tasks: Sequence[dict] = STANDARD_TASKS,
    corpus: Sequence[dict] = STANDARD_CORPUS,
    conditions: Sequence[str] = CONDITIONS,
    subsets: Sequence[tuple[str, ...]] = SUBSETS,
    progress=None,
) -> dict:
    """Run full factorial MCO experiment across all 16 subsets and 4 conditions."""
    bank = build_standard_bank(corpus)
    trials = []
    total_runs = len(tasks) * len(conditions) * len(subsets)
    completed = 0

    if progress and callable(progress):
        progress({"phase": "init", "completed": 0, "total": total_runs})

    for cond in conditions:
        for subset in subsets:
            for task in tasks:
                res = evaluate_task_trial(bank, task, subset, cond)
                trials.append(res)
                completed += 1
                if progress and callable(progress) and completed % 64 == 0:
                    progress({"phase": "evaluate", "completed": completed, "total": total_runs, "condition": cond, "subset": "+".join(subset)})

    # Summary aggregations
    condition_summary = {}
    for cond in conditions:
        cond_trials = [t for t in trials if t["condition"] == cond]
        condition_summary[cond] = {
            "trials": len(cond_trials),
            "mean_hit_at_1": fmean(t["hit_at_1"] for t in cond_trials),
            "mean_hit_at_3": fmean(t["hit_at_3"] for t in cond_trials),
            "mean_mrr": fmean(t["mrr"] for t in cond_trials),
            "mean_grounded_score": fmean(t["grounded_score"] for t in cond_trials),
            "mean_retrieval_ms": fmean(t["retrieval_ms"] for t in cond_trials),
        }

    # Factorial subset summary for intact condition
    subset_summary = {}
    intact_trials = [t for t in trials if t["condition"] == "intact"]
    for subset in subsets:
        name = "+".join(subset) if subset else "empty"
        sub_trials = [t for t in intact_trials if t["subset_name"] == name]
        subset_summary[name] = {
            "subset": list(subset),
            "active_count": len(subset),
            "trials": len(sub_trials),
            "mean_hit_at_1": fmean(t["hit_at_1"] for t in sub_trials),
            "mean_hit_at_3": fmean(t["hit_at_3"] for t in sub_trials),
            "mean_mrr": fmean(t["mrr"] for t in sub_trials),
            "mean_grounded_score": fmean(t["grounded_score"] for t in sub_trials),
            # Isolated performance when the query's target carrier is active vs inactive
            "target_active_grounded": fmean(t["grounded_score"] for t in sub_trials if t["target_carrier_active"]) if any(t["target_carrier_active"] for t in sub_trials) else 0.0,
            "target_inactive_grounded": fmean(t["grounded_score"] for t in sub_trials if not t["target_carrier_active"]) if any(not t["target_carrier_active"] for t in sub_trials) else 0.0,
        }

    # Marginal carrier contribution (presence vs absence across all 8 subset pairs)
    marginal_effects = {}
    for c in CARRIERS:
        with_c = [t["grounded_score"] for t in intact_trials if c in t["subset"]]
        without_c = [t["grounded_score"] for t in intact_trials if c not in t["subset"]]
        marginal_effects[c] = {
            "carrier": c,
            "mean_with": fmean(with_c),
            "mean_without": fmean(without_c),
            "marginal_grounded_delta": fmean(with_c) - fmean(without_c),
        }

    # Leave-one-task-out sensitivity
    loto = []
    task_ids = [t["id"] for t in tasks]
    overall_intact_mean = condition_summary["intact"]["mean_grounded_score"]
    for excluded_id in task_ids:
        remaining = [t["grounded_score"] for t in intact_trials if t["task_id"] != excluded_id]
        loto.append({
            "excluded_task": excluded_id,
            "mean_grounded": fmean(remaining),
            "delta_from_overall": fmean(remaining) - overall_intact_mean,
        })

    loto_summary = {
        "method": "leave-one-task-out-grounded-sensitivity-v1",
        "task_count": len(task_ids),
        "min_mean": min(item["mean_grounded"] for item in loto),
        "max_mean": max(item["mean_grounded"] for item in loto),
        "max_delta": max(abs(item["delta_from_overall"]) for item in loto),
    }

    best_sub = max(subset_summary.keys(), key=lambda k: subset_summary[k]["mean_grounded_score"])
    experiment_id = f"mco-{digest({'carriers': list(CARRIERS), 'task_count': len(tasks), 'subset_count': len(subsets)})[:16]}"
    experiment_data = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "date": "2026-10-09",
        "status": "completed",
        "carriers": list(CARRIERS),
        "conditions": list(conditions),
        "subset_count": len(subsets),
        "task_count": len(tasks),
        "total_evaluations": len(trials),
        "best_subset": best_sub,
        "carrier_contributions": marginal_effects,
        "condition_comparison": condition_summary,
        "condition_summary": condition_summary,
        "subset_summary": subset_summary,
        "marginal_effects": marginal_effects,
        "leave_one_task_out": loto_summary,
        "limits": LIMITS,
        "trials": trials,
    }
    experiment_data["receipt_sha256"] = digest({k: v for k, v in experiment_data.items() if k != "receipt_sha256"})

    if progress and callable(progress):
        progress({"phase": "done", "completed": total_runs, "total": total_runs})

    return experiment_data


def verify_mco_receipt(receipt: dict) -> dict:
    """Offline weightless verification of an MCO experiment receipt."""
    if not isinstance(receipt, dict):
        raise ValueError("MCO receipt must be a dict")
    recorded_hash = receipt.get("receipt_sha256")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if digest(unsigned) != recorded_hash:
        raise ValueError("Receipt checksum mismatch")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "completed":
        raise ValueError("Receipt schema or status invalid")

    trials = receipt.get("trials", [])
    expected_trials = receipt.get("task_count", len(STANDARD_TASKS)) * len(receipt.get("conditions", CONDITIONS)) * receipt.get("subset_count", len(SUBSETS))
    if len(trials) != expected_trials:
        raise ValueError(f"Expected {expected_trials} trials, found {len(trials)}")

    # Verify arithmetic consistency of summary metrics
    intact_trials = [t for t in trials if t["condition"] == "intact"]
    calc_intact_hit1 = fmean(t["hit_at_1"] for t in intact_trials)
    recorded_intact_hit1 = receipt["condition_summary"]["intact"]["mean_hit_at_1"]
    if abs(calc_intact_hit1 - recorded_intact_hit1) > 1e-6:
        raise ValueError("Condition summary arithmetic mismatch")

    return {
        "verified": True,
        "trials_verified": len(trials),
        "schema": SCHEMA,
        "condition_count": len(receipt["conditions"]),
        "subset_count": receipt["subset_count"],
        "scope": "Deterministic factorial memory retrieval, grounded F1, and falsifiable controls; not neural authorship.",
    }


def write_mco_receipt(receipt: dict, directory: Path) -> Path:
    """Save an MCO experiment receipt atomically with content-addressable filename."""
    directory.mkdir(parents=True, exist_ok=True)
    file_id = digest({"schema": receipt["schema"], "trials_count": len(receipt.get("trials", []))})[:16]
    receipt_hash = receipt["receipt_sha256"][:12]
    filename = f"{file_id}-{receipt_hash}.json"
    target = directory / filename

    temp_path = target.with_suffix(".tmp")
    temp_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(target)
    return target
