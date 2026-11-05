# Repository delivery — 2026-11-04

This package was prepared in response to the user's explicit request to commit
and push to [kai9987kai/Graft-brain](https://github.com/kai9987kai/Graft-brain).
The delivery branch is `codex/adaptive-graft-plasticity`; commit history records
the exact revision. The newer `CODE_OF_CONDUCT.md` and `SECURITY.md` commits from
remote `main` were preserved before preparing the feature commit.

## Included

- Adaptive site routing, fixed-budget structural plasticity, objective-gradient
  recruitment, maturation and checkpoint compatibility.
- CLI, viewer telemetry, controlled benchmarks and synthetic recovery probes.
- Regression tests, approved design/implementation records and verification
  reports, with positive, null and unfavorable outcomes retained.
- The five durable evidence directories referenced by the reports, including
  the frozen confirmation protocol/checksum, raw observations and figures.
- A synthetic viewer snapshot without camera data, retained as an API example.

Fresh pre-commit checks: `python -m pytest -q` passed **143 tests**. Additional
checks cover Python compilation, viewer text rendering, JavaScript syntax,
Markdown links, staged-file scope and recorded source/protocol fingerprints.

## Excluded and preserved

Local changes to the originally tracked checkpoint/journal, newly generated
training states, live logs, temporary viewer files and a provisional duplicate
probe were excluded from this package. Historical state already present in the
base repository is left unchanged; this commit does not publish updated weights.
The running simulation was not paused, restarted or reconfigured.

Plans and verification reports describe earlier authorization and evidence
checkpoints. Their historical uncommitted/no-publication notes are not current
delivery-status claims. No model/controller promotion, hosted service deployment
or GitHub release is included in this request.

## Evidence portability

Recorded Python source uses LF bytes. `.gitattributes` preserves those bytes on
checkout so computational source fingerprints remain verifiable across systems.
The locally frozen protocol JSON is an explicit `-text` exception: its recorded
CRLF bytes must remain exact for the committed SHA-256 sidecar to verify. Its
contents and checksum were not rewritten for publication.

The evidence supports finite synthetic mechanism results. The longer RL study
still shows unfavorable forage changes and unchanged go/no-go outcomes; it does
not establish task benefit, biological fidelity or general robustness. See the
[continuation report](verification-graft-continuation.md) for the boundaries.
