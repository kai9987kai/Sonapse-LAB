"""Read-only wiring and evidence probes; no candidate fitting or promotion."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.cognitive_agent import CognitiveAgent
from poseidon.cognitive_experiment import verify_cognitive_receipt
from poseidon.mnemosyne import MNEMOSYNE_MODES
from poseidon.research.candidate_evaluation import verify_evaluation_receipt
from poseidon.research.contracts import canonical_bytes
from poseidon.research.oracle_training import verify_fit_receipt
from poseidon.world import TidePool


torch.set_num_threads(2)
root = project_root / "outputs" / "architecture-audit-2026-10-11"
(root / "temporary").mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(root / "temporary")
result = {"audit_date": "2026-10-11", "git_head": subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True).strip()}
result["memory_probe"] = {}
if active_core_path().is_file():
    core = CoreRuntime(active_core_path())
    result["core_checkpoint"] = str(core.path.resolve())
    result["core_sha256"] = hashlib.sha256(core.path.read_bytes()).hexdigest()
    agent = CognitiveAgent(core=core)
    result["runtime_oracle"] = {
        "class": str(type(agent.oracle)), "is_calibrated": bool(agent.oracle.is_calibrated),
        "head_count": len(agent.oracle.heads),
    }
    for mode in MNEMOSYNE_MODES:
        candidate = CognitiveAgent(core=core, mode=mode)
        world = TidePool(seed=447000000, max_steps=8)
        episode = candidate.rollout(world, max_steps=8)
        result["memory_probe"][mode] = {
            "actions": [row["action"] for row in episode["trajectory"]],
            "reward": episode["total_reward"],
            "reported_final_health": episode["final_health"],
            "terminal_next_health": episode["trajectory"][-1]["next_observation"][0],
            "oracle_calibrated": bool(candidate.oracle.is_calibrated),
        }
else:
    result["runtime_oracle"] = {"not_run": "The active trained Tidal checkpoint is absent."}
result["probe_notes"] = [
    "One short seed is a wiring diagnostic, not a memory performance assay.",
    "Equal actions alone do not establish inert memory; dependency tracing is separately recorded in the audit.",
]
forged = {"schema": "not-a-cognitive-experiment", "episodes_data": {},
          "fabricated_performance": 1e9}
try:
    result["empty_forged_cognitive_receipt_accepted"] = verify_cognitive_receipt(forged)
except Exception as error:
    result["empty_forged_cognitive_receipt_accepted"] = False
    result["empty_forged_rejection_reason"] = str(error)
fit_path = Path("outputs/oracle-fit-442000000/fit_receipt.json")
try:
    result["fit_replay"] = verify_fit_receipt(fit_path, replay_data=True)
except Exception as error:
    result["fit_replay"] = {"error": str(error)}
if fit_path.is_file():
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    forged_fit = copy.deepcopy(fit)
    forged_fit["calibration"]["coverage"] = 0.99
    forged_fit.pop("receipt_sha256")
    forged_fit["receipt_sha256"] = hashlib.sha256(canonical_bytes(forged_fit)).hexdigest()
    with tempfile.TemporaryDirectory(prefix="poseidon-audit-") as directory:
        path = Path(directory) / "forged-fit.json"
        path.write_text(json.dumps(forged_fit), encoding="utf-8")
        try:
            result["resealed_fit_coverage_accepted"] = verify_fit_receipt(path, replay_data=True)
        except Exception as error:
            result["resealed_fit_coverage_accepted"] = {"error": str(error)}
else:
    result["resealed_fit_coverage_accepted"] = {"not_run": "Historical fit receipt is absent."}
evaluation_path = Path("outputs/oracle-eval-442000000/evaluation_receipt.json")
try:
    result["evaluation_replay"] = verify_evaluation_receipt(evaluation_path, replay=True)
except Exception as error:
    result["evaluation_replay"] = {"error": str(error)}
if evaluation_path.is_file():
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    result["pilot_metrics"] = {
        name: {"episodes": condition["episodes"], "metrics": condition["metrics"]}
        for name, condition in evaluation["results"].items()
    }
result["git_head_after"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
path = root / "diagnostics.json"
path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
