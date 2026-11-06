# 1000-neuron live workbench

6 November 2026. CLI and launcher defaults now select exactly 1000 total neurons,
allocated 334/333/333 across FlyWire, MANC and Optic Lobe. An optional
`total_neurons` configuration retains exact totals. Explicit per-source settings
and legacy checkpoints keep their existing interpretation. The live snapshot
includes all 1000 neurons; the session form edits the total directly.

Live verification at port 8876: 1000 executing and displayed neurons, 32 learned
transitions, successful checkpoint serialization. The former 48-neuron session
was saved before rebuilding the differently shaped graph, as
`outputs/workbench-check/before-1000-neurons-20261106.pt`. The new session was saved
as `outputs/workbench-check/1000-neurons-live-20261106.pt`.

Behavioral checks cover exact allocation, full display count, learning and replay
at 1000 neurons, legacy configuration matching, invalid totals, CLI defaults,
standalone CLI sizing and an API reset from 48 to 1000. JavaScript syntax and diff
checks pass. The full suite passes 332 tests after the sizing changes.
Increasing graph size does not itself establish improved task ability.
