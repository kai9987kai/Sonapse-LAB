# Sonapse-LAB DSAG baseline audit

## Frozen starting point

- Repository: Sonapse-LAB, GitHub remote `origin`, Hugging Face remote `hf`.
- Baseline branch: `main`.
- Baseline commit: `02c64ea5ff6b6d73c5c3433af5b732232d8aff64`.
- Baseline Git tree: `391ef49d860878aee14b5be676e63b9e25cdf2e2`.
- Working tree was clean before this feature branch.
- Feature branch: `codex/dsag-lab`.
- Python: `3.12.10`; Torch: `2.11.0+cpu`; configured Torch threads: 1; CUDA unavailable.
- Historical Tidal checkpoint fixture `artifacts/tidal_core.pt`: SHA-256 `A053104AE63DC7F86BAA15B4B34E46730E1CCB92117F221801FB21F90D2D6C27`.
- Test inventory: 120 `tests/test_*.py` files.

The DSAG brief also names Sonapse-Evo. That repository exists locally at `C:\Users\kai99\Desktop\New folder (13)`, but this feature branch changes Sonapse-LAB only; the Evo checkout and its running services were left untouched.

## Baseline runtime and CLI

`python -m synapse --nodes 16 status` exited successfully with zero steps, a 16-neuron proxy graph, one CPU Torch thread, and no CUDA device. Its JSON is the legacy default-off path; DSAG had not been added. `--nodes 2` was rejected by the existing `nodes_per_source` lower bound of 16.

The `pytest -q` executable failed collection because that launcher did not put this checkout on Python's import path (`ModuleNotFoundError: synapse` for 112 modules). The repository-compatible invocation is `python -m pytest -q`.

## Pre-change test result

Command: `python -m pytest -q`

Result: **651 passed, 12 failed, 2 warnings in 900.52 seconds**. These failures were observed on baseline commit `02c64ea5ff6b6d73c5c3433af5b732232d8aff64`, before any tracked source changes:

- `tests/test_antigravity_contract_regressions.py::test_all_inherited_advanced_features_replay_learned_checkpoint_exactly`
- `tests/test_evidence_sleep_regressions.py::test_sleep_trains_observation_conditioned_native_path_and_preserves_live_context`
- `tests/test_experience_integration.py::test_pending_transition_and_learning_resume_exactly`
- `tests/test_neuron_count.py::test_exact_total_builds_1000_neurons_and_replays`
- `tests/test_neuron_count.py::test_cli_default_is_exactly_4000`
- `tests/test_neuron_count.py::test_exact_total_builds_2000_neurons_and_replays`
- `tests/test_neuron_count.py::test_exact_total_builds_4000_neurons_and_replays`
- `tests/test_planning_workbench.py::test_policy_horizon_persists_and_manual_learning_works_through_api`
- `tests/test_runtime.py::test_all_world_backends`
- `tests/test_runtime_contracts.py::test_changed_endpoint_checkpoint_replays_in_fresh_runtime`
- `tests/test_stdp_allostasis_reconsolidation.py::test_server_stdp_allostasis_reconsolidation_endpoints`
- `tests/test_tandem_max_and_innovations.py::test_max_tandem_config_hybrid_planner_and_checkpoint_replay`

These are baseline failures, not DSAG regressions. The full suite includes long resource and large-neuron tests; preserve this result when assessing post-change runs.

## Inspected integration surfaces

Reviewed the existing LAB `synapse/graft_bus.py`, `synapse/agent.py`, `synapse/config.py`, `synapse/runtime.py`, `synapse/neurowave_x2/packets.py`, `synapse/neurowave_x2/fabric.py`, `synapse/cli.py`, `synapse/server.py`, `synapse/web/index.html`, `synapse/web/app.js`, and the 120-file test inventory. `NeuralGraftCommunicationBus` owns the final clamp at `max_aux_current`; `NeuralPacket` is frozen and computes its wire checksum in `packets.py`; `synapse/neural_graft.py` is an executed-transition policy graft and remains separate.

The pinned baseline does not prove that every historical test or every hidden/live service is healthy. The failures above are the starting evidence boundary for the DSAG work.

## Post-change regression run

Command: `python -m pytest -q`

Result: **666 passed, 12 failed, 2 warnings in 1217.73 seconds**. The 12 failing test names were exactly the same 12 listed in the pre-change run; no new suite failure appeared. The pass count increased by 15, matching the new DSAG focused tests. The DSAG implementation checks, CLI validation/smoke/ablation, a two-step LAB runtime, live `/api/dsag/status`, and browser panel were also exercised separately. The browser pilot advanced one simulation step and showed the channel ramp at alpha `0.003`.

Post-change validation establishes integration and deterministic synthetic fixture behavior only. It does not establish a task-level improvement, biological fidelity, or promotion readiness. The 12 inherited failures remain unresolved and are not represented as passing.
