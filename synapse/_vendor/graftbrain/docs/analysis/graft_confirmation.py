"""Validate and summarize the frozen 2026-11-04 confirmation; never edit raw data."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/graft-credit-confirmation-20261104.json"
RUN = ROOT / "runs/graft-credit-confirmation-20261104"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


protocol = read(PROTOCOL)
protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
assert protocol_hash == PROTOCOL.with_suffix(".json.sha256").read_text().strip()
assert protocol["status"] == "frozen_before_confirmation"
manifest, summary, curves = (read(RUN / name) for name in ("manifest.json", "summary.json", "curves.json"))
assert manifest["status"] == "complete" and not manifest["source_changed_during_run"]
assert manifest["parameters"]["seeds"] == protocol["confirmation_seeds"]
assert manifest["parameters"]["train_steps"] == protocol["train_steps"]
assert manifest["parameters"]["recovery_steps"] == protocol["recovery_steps"]
for name, digest in protocol["source_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
for name, digest in manifest["source_sha256"].items():
    assert digest == protocol["source_sha256"][name] == manifest["source_sha256_end"][name]
results = {row["seed"]: row for row in summary["results"]}
assert len(results) == len(summary["results"]) == len(protocol["confirmation_seeds"])
assert set(results) == set(protocol["confirmation_seeds"])
assert all(row["source_unchanged"] for row in results.values())
T = protocol["recovery_steps"]
BRANCHES = ("sham", "disrupted_rewire", "disrupted_no_rewire")
points, metrics, events = {}, {}, []
for seed in protocol["confirmation_seeds"]:
    metrics[seed] = {}
    for branch in BRANCHES:
        rows = sorted((r for r in curves["rows"] if r["seed"] == seed and r["phase"] == "recovery"
                       and r["branch"] == branch), key=lambda r: r["step"])
        assert [r["step"] for r in rows] == list(range(T + 1))
        assert all(r["bucket_active"] == [6, 6] and r["active"] == 12 for r in rows)
        y = np.array([r["mse"] for r in rows], dtype=float)
        assert np.isfinite(y).all() and (y >= 0).all()
        auc = float(sum((a + b) * .5 for a, b in zip(y, y[1:])) / T)
        tail = float(y[-min(10, T):].mean())
        stored = results[seed]["branches"][branch]
        for key, value in (("recovery_auc_mean", auc), ("tail_mean_error", tail), ("final_error", float(y[-1]))):
            assert math.isclose(value, stored[key], rel_tol=1e-12, abs_tol=1e-12), (seed, branch, key)
        metrics[seed][branch] = {"auc": auc, "tail": tail, "final": float(y[-1]),
                                 "old_target_final": rows[-1]["old_target_mse"]}
        points[(seed, branch)] = y
        for event in stored["support_events"]:
            assert math.isclose(event["post_rewire_mse"] - event["pre_rewire_mse"], event["rewire_mse_delta"],
                                rel_tol=1e-12, abs_tol=1e-12)
            events.append({"seed": seed, "branch": branch, **event})

paired = [{"seed": seed, **{metric: metrics[seed]["disrupted_no_rewire"][metric] -
                             metrics[seed]["disrupted_rewire"][metric] for metric in ("auc", "tail", "final")}}
          for seed in protocol["confirmation_seeds"]]
auc_differences = np.array([row["auc"] for row in paired])
rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"])
indices = rng.integers(0, len(paired), size=(protocol["analysis"]["bootstrap_samples"], len(paired)))
interval = np.quantile(auc_differences[indices].mean(axis=1), [.025, .975]).tolist()
analysis = {
    "protocol_sha256": protocol_hash, "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seeds": protocol["confirmation_seeds"], "all_declared_seeds_retained": True,
    "per_seed_paired_differences": paired,
    "primary": {"definition": protocol["primary_metric"]["definition"],
                "mean": float(auc_differences.mean()), "median": float(np.median(auc_differences)),
                "positive": int((auc_differences > 0).sum()), "zero": int((auc_differences == 0).sum()),
                "negative": int((auc_differences < 0).sum()), "bootstrap_mean_95_percent_interval": interval,
                "uncertainty_scope": protocol["analysis"]["uncertainty"]},
    "secondary": {metric: {"mean": float(np.mean([row[metric] for row in paired])),
                           "positive": sum(row[metric] > 0 for row in paired),
                           "negative": sum(row[metric] < 0 for row in paired)} for metric in ("tail", "final")},
    "old_target_final_mean": {branch: float(np.mean([metrics[s][branch]["old_target_final"]
                                                    for s in protocol["confirmation_seeds"]])) for branch in BRANCHES},
    "recovery_edit_observations": {branch: {
        "events": sum(e["branch"] == branch for e in events),
        "error_increasing": sum(e["branch"] == branch and e["rewire_mse_delta"] > 0 for e in events),
        "error_decreasing": sum(e["branch"] == branch and e["rewire_mse_delta"] < 0 for e in events),
        "max_error_increase": max((e["rewire_mse_delta"] for e in events if e["branch"] == branch), default=0),
    } for branch in BRANCHES},
    "limits": protocol["limitations"],
}
(RUN / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"primary": analysis["primary"], "secondary": analysis["secondary"]}, indent=2))

# Optional export of the recorded observations; all seeds remain visible.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(paired))
axes[0].bar(x, auc_differences, color=["#15803d" if d > 0 else "#b91c1c" for d in auc_differences])
axes[0].axhline(0, color="#475569", linewidth=.8)
axes[0].set_xticks(x, [str(row["seed"]) for row in paired], rotation=60, fontsize=8)
axes[0].set_ylabel("Fixed minus adaptive normalized MSE area")
axes[0].set_title("Every declared paired seed")
for branch, label, color, style in (("sham", "Sham + adaptation", "#64748b", "--"),
                                   ("disrupted_rewire", "Replacement + adaptation", "#15803d", "-"),
                                   ("disrupted_no_rewire", "Replacement + fixed support", "#c2410c", "-")):
    all_y = np.stack([points[(seed, branch)] for seed in protocol["confirmation_seeds"]])
    axes[1].plot(range(T + 1), all_y.mean(axis=0), label=label, color=color, linestyle=style)
axes[1].set_xlabel("Recovery update")
axes[1].set_ylabel("Mean current MSE across 17 seeds")
axes[1].set_ylim(bottom=0)
axes[1].set_title("Mean full recovery trajectories")
axes[1].legend(fontsize=8, frameon=False)
for ax in axes:
    ax.grid(axis="y", alpha=.2)
fig.suptitle("Frozen confirmation on one supervised teacher fixture", fontsize=13)
fig.text(.5, .005, "Positive paired area favors adaptation. This is not an RL, retention, or biological result.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .94))
fig.savefig(RUN / "confirmation.png", dpi=160)
fig.savefig(RUN / "confirmation.svg")
