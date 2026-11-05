# Local workbench reliability and evidence upgrade

This upgrade strengthens the existing CPU-local research workbench. It does not
train new weights or promote an experimental controller. Distribution metadata
follows the runtime source identity; integrated validation uses the shared
checkout's 3.8.0 version.

## Changes

- Completed jobs can be reopened from saved history after reloading the page.
  Queued and running jobs can be monitored without submitting duplicate work.
  Polling and cancellation failures remain visible, with retry where appropriate.
- Workbench experiment requests match each backend's accepted settings.
  METAMORPH uses the durable job queue. The unsupported NEXUS paired comparison
  is explicitly unavailable instead of silently submitting an Atlas experiment.
- MCO results have a dedicated view for retrieval-condition scores, carrier
  effects, and the best observed carrier subset. Historical result requests have
  generation ownership, so a late response cannot replace a newer monitored job.
- A missing optional status card cannot make a connected server appear offline.
  Optional component failures and invalid local reports are isolated and exposed
  in `status_errors` while other status information remains available.
- Status reads bounded JSON reports and a bounded tail of training metrics.
  Valid recent records survive a partial or damaged JSONL record. Non-finite
  values cannot break the status JSON response.
- Source snapshots cover nested research packages with relative paths. Editing,
  adding, or removing a nested Python source requires restarting a loaded process
  before it can produce new experiment evidence.
- TRIDENT, TITAN, CHIMERA, HYPERION and MCO status summaries check receipt seals
  before exposing their metrics. `receipt_integrity` means a checksum check;
  `receipt_verified` and `replay_verified` remain false during readiness polling.
  MCO also checks its existing summary arithmetic. These checks do not establish
  provenance, statistical significance, model quality, or controller superiority.
- Reasoning verification follows the answer-specific solver verdict. Fixed
  synthetic safety examples and physical demonstrations do not certify a user's
  query. Parsed-model checks do not certify natural-language interpretation.
  Generated prose is marked unverified; a session fingerprint is not a proof.
- Distribution metadata obtains its version from `poseidon.__version__` without
  importing the runtime during packaging. SciPy is a declared dependency. The
  release builder and verifier accept a consistent set of supported versions.

## Use and verify

Start a fresh process after updating source:

```powershell
python -m poseidon status
python -m poseidon serve --port 8792
```

In the workbench, use **Refresh jobs** and **View result** or **Monitor job** to
recover an earlier experiment. Fixed memory protocols expose their actual
settings instead of accepting controls that the backend ignores or rejects.

Run focused checks or the full regression suite:

```powershell
python -m pytest tests/test_provenance.py tests/test_status_resilience.py tests/test_omni_evidence.py tests/test_web_jobs.py tests/test_packaging.py tests/test_release_verification.py -q
python -m pytest -q
```

UI handler regressions execute the shipped JavaScript against the current HTML
IDs with controlled HTTP responses. They require Node.js; the wrapper skips them
when Node.js is absent. Local HTTP integration tests require loopback sockets.
Packaging tests use temporary source copies and do not reinstall the live
environment. A pre-existing editable installation can therefore still report its
old distribution version until the user reinstalls it.

Historical experiment files and model checkpoints are retained. An old receipt
may no longer satisfy a current verifier after source/schema changes. Use the
explicit `verify-*` command and its documented scope to audit a particular
receipt; readiness polling never automatically reruns a controller experiment.

## Local validation on 21 October 2026

- Full Python suite: **1,048 passed** in 947.04 seconds with loopback access.
- Browser-handler suite: **38 passed**, exercising the current shipped script and
  HTML, including deferred-response races and MCO rendering.
- Live browser: submitted one MCO job, reloaded the page, recovered it from
  history, and reopened its completed retrieval results. The benchmark evaluated
  1,024 fixed-protocol trials. This is a workflow check, not a model-performance
  improvement claim.
- Live maths: `3*x + 7 = 22` returned `x = 5` and checked both sides equal 22.
- Fresh CLI status: version 3.8.0, local core and language artifacts available,
  current source snapshot, and no status errors.
- Offline wheel: `outputs/upgrade-build-current/supermix_poseidon-3.8.0-py3-none-any.whl`.
  All 179 package source/asset files matched the checkout byte for byte. Metadata
  reports 3.8.0 and declares SciPy. This source application wheel excludes model
  weights and is not a published model release.

Local validation artifacts are in `outputs/upgrade-full-tests.log`,
`outputs/upgrade-browser-handlers.log`, `outputs/upgrade-live-status.json`, and
`outputs/upgrade-browser/mco-history-result.jpg`. These generated outputs remain
outside version control. Source commits made concurrently in the shared checkout
were retained; this upgrade did not perform a remote publication.

Version configuration follows the setuptools
[dynamic metadata guidance](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#dynamic-metadata).
