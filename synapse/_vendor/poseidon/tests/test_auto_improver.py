"""Regression coverage for candidate-local parameter search, without a real search."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import pytest

from tools import auto_improver as search


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    (root / "poseidon").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "runs").mkdir()
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text("1" * 40 + "\n")
    (root / "poseidon" / "active_inference.py").write_text(
        "base_epistemic_weight: float = 0.30\n# uncommitted source change\n"
    )
    (root / "dirty.txt").write_bytes(b"unrelated uncommitted work\r\n")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (root / "tests" / "test_gate.py").write_text(
        "from pathlib import Path\n"
        "def test_workspace_isolation():\n"
        "    assert not Path('.git').exists()\n"
        "    assert not Path('dirty.txt').exists()\n"
        "    assert Path('runs/search_core.pt').read_bytes() == b'frozen core'\n"
        "    assert Path('runs/search_oracle.pt').read_bytes() == b'frozen oracle'\n"
    )
    (root / "runs" / "core.pt").write_bytes(b"frozen core")
    (root / "runs" / "oracle.pt").write_bytes(b"frozen oracle")
    (root / "runs" / "active_core.json").write_text(json.dumps({"checkpoint": "runs/core.pt"}))
    monkeypatch.setattr(search, "SEARCH_SPACE", [search.SEARCH_SPACE[0]])
    monkeypatch.setattr(search, "TEST_FILES", ["tests/test_gate.py"], raising=False)
    return root


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def benchmark_result(seeds, reward, planning_enabled=True):
    rows = [{"seed": seed, "total_reward": reward, "survived": True,
             "steps": 8, "overrides": 0,
             "planning_enabled_steps": 8 if planning_enabled else 0,
             "fallback_reasons": {} if planning_enabled else {"insufficient_training_action_exposure": 8}}
            for seed in seeds]
    return {"per_seed": rows, "mean_reward": reward, "survival_rate": 1.0,
            "overrides": 0, "total_steps": 8 * len(seeds),
            "planning_enabled_steps": 8 * len(seeds) if planning_enabled else 0,
            "fallback_reasons": {} if planning_enabled else {"insufficient_training_action_exposure": 8 * len(seeds)},
            "rewards": [reward] * len(seeds)}


def run_one(root, out, **kwargs):
    kwargs.setdefault("oracle_path", root / "runs/oracle.pt")
    return search.run_search(source_root=root, output_root=out,
                             end_time=datetime.now() + timedelta(seconds=30),
                             seed=14, max_iterations=1, **kwargs)


def test_rejection_preserves_dirty_checkout_head_and_bound_core(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        assert workspace != source_tree
        reward = 1.0 if workspace.name == "baseline" else -1.0
        return benchmark_result(seeds, reward)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["status"] == "completed"
    assert result["selected_candidate"] is None
    record = result["candidates"][0]
    assert record["status"] == "REJECTED_NO_IMPROVEMENT"
    assert Path(record["workspace"]).joinpath("poseidon/active_inference.py").exists()
    assert record["baseline_binding"]["core_sha256"] == hashlib.sha256(b"frozen core").hexdigest()
    assert record["baseline_binding"]["oracle_sha256"] == hashlib.sha256(b"frozen oracle").hexdigest()
    assert record["candidate_binding"]["oracle_sha256"] == record["baseline_binding"]["oracle_sha256"]
    assert Path(record["workspace"]).joinpath("runs/search_oracle.pt").read_bytes() == b"frozen oracle"
    assert record["candidate_binding"]["source_sha256"] != record["baseline_binding"]["source_sha256"]
    assert [row["seed"] for row in record["selection"]["candidate"]["per_seed"]] == search.SELECTION_SEEDS
    assert json.loads(Path(result["report_path"]).read_text())["status"] == "completed"


def test_selection_remains_inactive_and_audit_is_separate(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)
    seen = []

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        seen.append((workspace.name, tuple(seeds)))
        reward = 1.0 if workspace.name == "baseline" else 2.0
        if seeds == search.AUDIT_SEEDS and workspace.name != "baseline":
            reward = -2.0
        return benchmark_result(seeds, reward)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["selected_candidate"]["active"] is False
    assert result["audit"]["status"] == "completed"
    assert result["audit"]["improvement_supported"] is False
    assert result["audit"]["delta_reward"] == -3.0
    assert not set(search.SELECTION_SEEDS) & set(search.AUDIT_SEEDS)
    assert seen == [("baseline", tuple(search.SELECTION_SEEDS)),
                    ("candidate_0001", tuple(search.SELECTION_SEEDS)),
                    ("baseline", tuple(search.AUDIT_SEEDS)),
                    ("candidate_0001", tuple(search.AUDIT_SEEDS))]


def test_proposal_seed_is_independent_of_global_random_state(source_tree):
    left = random.Random(123)
    right = random.Random(123)
    expected = [search.propose_mutation(source_tree, left) for _ in range(8)]
    random.seed(999)
    actual = [search.propose_mutation(source_tree, right) for _ in range(8)]
    assert actual == expected
    assert len({row["new"] for row in actual}) > 1
    assert all(row["old"] == 0.30 for row in actual)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_candidate_preserves_incumbent(source_tree, tmp_path, monkeypatch, interruption):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        if workspace.name == "baseline":
            return benchmark_result(seeds, 1.0)
        if interruption == "cancel":
            raise KeyboardInterrupt()
        search.run_command([sys.executable, "-c", "import time; time.sleep(5)"],
                           workspace=workspace, deadline=time.monotonic() + 0.1)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["status"] == {"timeout": "timed_out", "cancel": "cancelled"}[interruption]
    assert result["selected_candidate"] is None
    baseline = Path(result["baseline_workspace"])
    assert baseline.joinpath("poseidon/active_inference.py").read_bytes() == before[str(Path("poseidon/active_inference.py"))]
    assert result["candidates"][0]["status"] in {"TIMED_OUT", "CANCELLED"}


def test_expired_deadline_never_starts_benchmark(source_tree, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("an expired run must not start a benchmark")

    monkeypatch.setattr(search, "run_benchmark", fail)
    result = search.run_search(source_root=source_tree, output_root=tmp_path / "artifacts",
                               end_time=datetime.now() - timedelta(seconds=1), seed=1)
    assert result["status"] == "timed_out"
    assert result["selected_candidate"] is None


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected(hours):
    with pytest.raises(ValueError):
        search.parse_target_end_time(None, hours)


def test_baseline_core_mutation_invalidates_evidence(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        (workspace / search.CORE_RELATIVE_PATH).write_bytes(b"changed core")
        return benchmark_result(seeds, 1.0)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["status"] == "failed"
    assert result["candidates"] == []
    assert result["selected_candidate"] is None
    assert "baseline" in result["error"]


def test_cancel_after_selection_keeps_previous_inactive_candidate(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        if workspace.name == "candidate_0002":
            raise KeyboardInterrupt()
        return benchmark_result(seeds, 1.0 if workspace.name == "baseline" else 2.0)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = search.run_search(source_root=source_tree, output_root=tmp_path / "artifacts",
                               end_time=datetime.now() + timedelta(seconds=40), seed=14, max_iterations=2,
                               oracle_path=source_tree / "runs/oracle.pt")
    assert tree_bytes(source_tree) == before
    assert result["status"] == "cancelled"
    assert result["selected_candidate"]["iteration"] == 1
    assert result["selected_candidate"]["active"] is False
    selected = Path(result["selected_candidate"]["workspace"])
    assert search.source_binding(selected) == result["selected_candidate"]["binding"]
    assert result["candidates"][1]["status"] == "CANCELLED"
    assert result["audit"]["status"] == "not_run"


def install_benchmark_standins(source_tree, *, planning_enabled=True):
    # Tiny stand-ins keep the subprocess/protocol real without loading a trained model.
    (source_tree / "torch.py").write_text("def set_num_threads(n): pass\ndef manual_seed(n): pass\n")
    (source_tree / "poseidon/__init__.py").write_text("")
    (source_tree / "poseidon/core.py").write_text(
        "class CoreRuntime:\n"
        "    def __init__(self, path):\n"
        "        assert path == 'runs/search_core.pt'\n"
    )
    (source_tree / "poseidon/worldlab.py").write_text(
        "class POMDPConfig:\n"
        "    def __init__(self, **kwargs): self.seed = kwargs['seed']\n"
        "class POMDPTidePool:\n"
        "    def __init__(self, config, max_steps): self.seed = config.seed\n"
    )
    (source_tree / "poseidon/cognitive_agent.py").write_text(
        "from pathlib import Path\n"
        "class CognitiveAgent:\n"
        "    def __init__(self, core, oracle_checkpoint=None):\n"
        "        assert oracle_checkpoint is not None\n"
        "        assert Path(oracle_checkpoint).read_bytes() == b'frozen oracle'\n"
        "        assert Path(oracle_checkpoint).parent.resolve() == Path('runs').resolve()\n"
        "        self.used = False\n"
        "    def rollout(self, env, max_steps):\n"
        "        assert not self.used\n"
        "        self.used = True\n"
        f"        decision = {{'planning_enabled': {planning_enabled!r}, 'fallback_reason': "
        f"{None if planning_enabled else 'insufficient_training_action_exposure'!r}}}\n"
        "        return {'total_reward': env.seed / 10, 'survived': True, 'steps': 2, 'overrides': 1, 'decisions': [decision, decision]}\n"
    )


def test_benchmark_subprocess_records_complete_requested_seed_outcomes(source_tree):
    install_benchmark_standins(source_tree)
    result = search.run_benchmark(source_tree, seeds=[10, 20], deadline=time.monotonic() + 10,
                                  oracle_path=source_tree / "runs/oracle.pt")
    assert result["mean_reward"] == 1.5
    assert result["survival_rate"] == 1.0
    assert result["total_steps"] == 4
    assert result["planning_enabled_steps"] == 4
    assert result["fallback_reasons"] == {}
    assert [row["seed"] for row in result["per_seed"]] == [10, 20]
    assert [row["trajectory"]["total_reward"] for row in result["per_seed"]] == [1.0, 2.0]


def test_missing_oracle_stops_before_copy_or_benchmark(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)

    def fail(*args, **kwargs):
        pytest.fail("a search without an explicit trained oracle must not launch work")

    monkeypatch.setattr(search, "run_benchmark", fail)
    result = run_one(source_tree, tmp_path / "artifacts", oracle_path=None)
    assert result["status"] == "inactive"
    assert "--oracle" in result["reason"]
    assert result["candidates"] == []
    assert not Path(result["baseline_workspace"]).exists()
    assert tree_bytes(source_tree) == before


def test_all_fallback_baseline_stops_without_proposals(source_tree, tmp_path):
    install_benchmark_standins(source_tree, planning_enabled=False)
    before = tree_bytes(source_tree)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert result["status"] == "inactive"
    assert "planning" in result["reason"].lower()
    assert result["candidates"] == []
    assert result["baseline_selection"]["planning_enabled_steps"] == 0
    assert result["baseline_selection"]["fallback_reasons"] == {"insufficient_training_action_exposure": 8}
    assert tree_bytes(source_tree) == before


@pytest.mark.parametrize("stage", ["baseline", "candidate", "audit"])
def test_oracle_mutation_invalidates_stage_evidence(source_tree, tmp_path, monkeypatch, stage):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, oracle_path, label="selection"):
        assert oracle_path == workspace / "runs/search_oracle.pt"
        if ((stage == "baseline" and workspace.name == "baseline")
                or (stage == "candidate" and workspace.name == "candidate_0001")
                or (stage == "audit" and label == "audit")):
            oracle_path.write_bytes(b"changed oracle")
        return benchmark_result(seeds, 1.0 if workspace.name == "baseline" else 2.0)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    if stage == "audit":
        assert result["audit"]["status"] == "failed"
        assert "improvement_supported" not in result["audit"]
    else:
        assert result["status"] == "failed"
        assert result["selected_candidate"] is None


@pytest.mark.parametrize("stage", ["candidate", "audit"])
def test_fallback_only_candidate_or_audit_cannot_claim_improvement(source_tree, tmp_path, monkeypatch, stage):
    def benchmark(workspace, *, seeds, deadline, label="selection", **kwargs):
        disabled = ((stage == "candidate" and workspace.name != "baseline")
                    or (stage == "audit" and label == "audit"))
        return benchmark_result(seeds, 1.0 if workspace.name == "baseline" else 2.0,
                                planning_enabled=not disabled)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    if stage == "candidate":
        assert result["status"] == "inactive"
        assert result["selected_candidate"] is None
        assert result["candidates"][0]["status"] == "REJECTED_PLANNING_DISABLED"
        assert "improvement_supported" not in result["candidates"][0]["selection"]
    else:
        assert result["audit"]["status"] == "inactive"
        assert "improvement_supported" not in result["audit"]
        assert result["selected_candidate"]["active"] is False
