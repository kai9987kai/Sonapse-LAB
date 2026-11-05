# Adaptive graft verification — 2026-11-03

Implemented on `codex/adaptive-graft-plasticity`. At this verification checkpoint,
changes were uncommitted and no publication, push or release had been performed.
See [publication notes](publication.md) for later repository delivery.

## Delivered behavior

- Observation-dependent, differentiable site gains in `(0, 2)`, initially one.
- Fixed-budget pruning/regrowth within existing candidate edge records, preserving
  every `(site, source connectome, target connectome)` bucket budget.
- Dormant-candidate coactivity, deterministic selection, and reset of changed
  records' eligibility, pending Hebbian updates, and optimizer moments.
- Five modes: legacy, static, routing, structural, and adaptive.
- Checkpointed routing/structural state with candidate-identity validation.
- CLI controls, effective-flow/site telemetry, and paired benchmark artifacts.
- Complete rollout/terminal reward accounting and final headless rollout flush.

The old checkpoint loader also handles saved interfaces with fewer unique ports
than the previous requested port count. Matching partial CLI overrides preserve
all other saved plasticity settings; actual conflicts require a fresh run.

## Verification performed

`python -m pytest -q`: **59 passed in 21.58 seconds**.

Tests cover zero-noise legacy forward and gradient parity, integrated dormant
current/weight isolation, router learning, directional budgets, tiny/empty-site
behavior, trace/moment resets, deterministic structural restoration, graph
identity rejection, legacy checkpoint shapes, partial override semantics,
terminal/rollout accounting, frozen evaluation, benchmark pairing/artifacts,
new journal directories, strict finite snapshot JSON, and real aiohttp
HTTP/WebSocket responses.

`python -m compileall -q graftbrain tests`, viewer JavaScript syntax checking
with Node, and `git diff --check` passed. Git emitted only line-ending notices.
The viewer also received isolated DOM smoke checks for telemetry, empty sites,
and escaping of dataset-derived strings.

The original oracle script again printed 10/10 for all nine rows. That script
does not assert results, and its webcam row counts positive return despite a
timeout outcome. This is not a 90/90 terminal-success test or evidence of learned
policy performance.

The integrated viewer server served its local endpoint. The in-app browser
could not attach its webview, so visual browser verification remains unavailable.
The live check exposed a preexisting missing-directory journal failure; a
regression test now covers the repair. Temporary viewer state was removed.

## Recorded CPU pilot

Artifacts:

- [Manifest](../runs/adaptive-grafts-20261103/manifest.json)
- [Per-episode observations](../runs/adaptive-grafts-20261103/episodes.json)
- [Summary](../runs/adaptive-grafts-20261103/summary.json)

Command:

```powershell
python -m graftbrain benchmark-plasticity --out runs/adaptive-grafts-20261103 --train-steps 512 --tasks forage go_nogo --seeds 0 --eval-seeds 1000 1001 --max-neurons 400 --microsteps 2
```

Ten arms completed: five modes × two tasks, one training/graph seed, 512 training
steps per arm (5,120 total), and two held-out evaluation episodes per arm.
The proxy graph had 800 nodes and 1,590 candidate graft records. All four sparse
arms started with the same 952 active records. Structural and adaptive arms each
swapped 90 records per task while preserving that budget; routing/adaptive arms
developed nonzero site-gain spread. Frozen evaluation used fresh trained-state
clones and policy mean actions.

| Mode | Active/candidate records | Swaps per task | Forage mean eval return | Go/no-go mean eval return |
|---|---:|---:|---:|---:|
| legacy | 1590/1590 | 0 | -4.75 | 2.60 |
| static | 952/1590 | 0 | -4.71 | 2.60 |
| routing | 952/1590 | 0 | -4.74 | 2.60 |
| structural | 952/1590 | 90 | -4.74 | 2.60 |
| adaptive | 952/1590 | 90 | -4.76 | 2.60 |

Every forage evaluation timed out. Both held-out go/no-go seeds selected the
**no-go** condition; their successes do not establish the go condition or a
balanced inhibitory-control result. This small pilot verifies mechanism activity
and budget accounting. It does **not** establish a task advantage. Wall times
were measured on a contended machine and support no efficiency comparison.

The manifest records base revision `e6595d04590cb75f6214fe1562732cf53a064c70`
and a dirty working tree. It is not an immutable released-code replay receipt.
Later review repairs concern resume overrides, empty diagnostics, journal
directories and CLI final flush; the pilot uses the direct experiment runner.

## Preserved state and limits

The existing checkpoint loaded read-only as legacy mode at step 16,712 with
12,000 nodes and 95,920 graft records. These SHA-256 values were identical before
and after work:

```text
BDB74CE2A931D92CF04DA0B0A773F61350AC4CA1CEDB85190580FB4CB91FCFF6  state/graftbrain.pt
A2F89473F0BF854488C0FF2132A78E07146A2D1C4CE7237EE2A80A291D5C456C  state/graftbrain_journal.jsonl
```

The learner is a one-pass GAE actor-critic, not multi-epoch PPO. Full-world exact
continuation, pending-rollout recovery, memory-module repairs, and arbitrary new
anatomical endpoints are outside this increment. Candidate-sized tensors remain
allocated/computed. Default proxies and the bounded pilot do not establish
biological fidelity or general learning capability.

## Launch an isolated adaptive run

```powershell
python -m graftbrain run --fresh --graft-mode adaptive --state state/adaptive-grafts.pt
```

Resume without `--fresh` and without mode changes to retain saved settings.
