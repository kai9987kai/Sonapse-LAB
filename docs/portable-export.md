# Portable Simulation Export: Capabilities and Limits

## What Export does

Choose **Export Simulation** in the live workbench. The request waits for a
simulation step boundary, captures the runtime state under the same lock used by
the simulation worker, and then releases the runtime so it can continue while
the archive and HTML are written. There is no minimum elapsed time or experiment
completion requirement. The button reports completion only after the HTML has
been written and validated and then starts a browser download.

The generated file is named
`Sonapse_Portable_Simulation_YYYYMMDD_HHMMSS_<id>.html`. It opens directly from
disk and makes no network requests. It contains its CSS, JavaScript, a snapshot
view, bounded replay data, and the exact bytes of the native checkpoint.

## Preserved computational state

The embedded checkpoint uses the existing `synapse-state-v1` archive and the
runtime's schema-1 checkpoint payload. It includes the configuration, agent
export state, world and world RNG, current observation, step count, recent
outcomes, latest telemetry/action, reusable skills, and runtime monitors. The
agent export includes graph topology, learned module and optimizer state, RNG
and transient variables, plus state from enabled memory, plasticity, spiking,
communication, and other configured subsystems. The checkpoint is serialized
using the existing safe tensor/primitive codec; arbitrary Python objects are not
accepted.

The HTML metadata records the checkpoint byte count and SHA-256. The exporter
checks the archive before packaging, checks it again while embedding, writes the
HTML atomically, and reports the HTML byte count and SHA-256. The underlying
checkpoint remains compatible with the existing native loader when restored
into a runtime with a matching configuration.

## Replay and visualization

The workbench records up to 2,048 transitions after the recorder attaches. Each
frame stores step, action, reward, terminal flag, and activity for 128 evenly
spaced neurons. The standalone page supports play/pause, timeline scrubbing,
single-frame stepping, node selection, role filtering, and graph zoom/pan. The
graph and connection weights use the workbench's sampled preview (up to 8,000
nodes, 3,200 base edges, and 2,400 active graft edges), not the full graph
archive. The replay is an inspection trace, not a complete event-by-event
reconstruction or a history of past connection-weight changes. If the recorder
starts after step zero, replay begins at its recorded start step. When the
2,048-frame cap is reached, the
oldest replay frames are dropped; the complete checkpoint remains unchanged.

## Continuation compatibility boundary

The standalone HTML does **not** run or continue the neural simulation. The
current project execution path uses Python world implementations, PyTorch
modules/autograd, optimizers, and many model-specific Python state handlers.
There is no bundled WebAssembly/PyTorch runtime, faithful JavaScript port, or
browser equivalent for these enabled systems. Replacing them with a rate-only
animation would not preserve simulation behavior, so this export explicitly
disables continuation in the viewer and does not claim mathematical equivalence.

To continue, use **Download embedded native checkpoint**, then place the `.pt`
file in the active Sonapse artifact/state directory under a safe checkpoint
name and load it with the matching runtime configuration. The native runtime
can then continue learning and produce a later portable export. Browser-side
reset, parameter edits, simulation save, and re-export are unavailable because
the HTML is an inspector, not an executable engine.

## Performance and browser behavior

The inspector uses embedded Canvas 2D rendering and works without WebGPU, a
server, or internet access. Rendering performance is separate from simulation
performance; the page reports its own canvas frame rate and does not present it
as neural steps per second. The full checkpoint is stored as base64 inside the
HTML, so the file is about one third larger than the native archive. Very large
checkpoints require corresponding disk space and browser memory when the user
chooses to extract the native checkpoint.

The live simulation is held at a safe step boundary while the detached full
checkpoint and current preview are captured. Disk serialization and HTML
packaging happen after that capture. Peak memory still includes the detached
checkpoint, so an export can fail with a clear error if the host cannot allocate
the required snapshot; the source runtime is not reset or rewritten by export.
