# Systems and media research for Supermix Poseidon

Reviewed 2026-10-08. These are source-level observations, not reproduced upstream benchmarks. Repositories were inspected read-only through GitHub, raw pinned files and branch references. No upstream project was installed or executed. No source code, game assets, binaries or repository text from these projects was inserted into Poseidon's training corpus.

## Source inventory and concrete lessons

| Repository | Inspected revision | Observed root license | Files inspected |
|---|---|---|---|
| [NexusFlow](https://github.com/kai9987kai/nexusflow) | `3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec` | MIT, Kai Piper | README, LICENSE, `nexusflow/cli.py`, first 190 lines of `NEXUSFLOW_LANGUAGE.md` |
| [universal-modder](https://github.com/rehan-remade/universal-modder) | `8370faa8e114baf33acdb23079aff552a7728c4b` | MIT, Rehan and contributors | LICENSE, first 190 lines of README, repository tree |
| [REA](https://github.com/morluto/rea) | `84a17d55199b1b46493aea6bf4e41029a3b200fc` (refreshed) | MIT | ADR-0001, ADR-0002 removal notice; earlier LICENSE and repository-tree review at `4fc6565b0ffb36d7105b3524e08d0c5d603ae07d` |
| [Odysseus](https://github.com/odysseus-dev/odysseus) | `a8c147b238db01dbd00b57774ed8b453f1e43971` (`dev`) | AGPL-3.0 | LICENSE header, `docs/AGENT_TURN_CONTRACT.md`, `core/atomic_io.py`, repository tree |
| [Periodic NTAG emulator](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator) | `98fd7478253f9041195adc8eedfa95eabe084aff` | No license file found; API license field empty | Full repository tree, `application.fam`, first 150 lines of `ntag213_rotator.c` |
| [quibble-builds](https://github.com/7coil/quibble-builds) | `f89481a1a3b81abfcbc1d43a2d539204101201aa` | LGPL-3.0 | LICENCE header, first 120 lines of README, `.github/workflows/build-amd64.yml` |
| [SVideo](https://github.com/7coil/svideo) | `d4b34633e92be30fd6395da1ffde21d6cac96917` | MIT, Leondro Lio | LICENCE header, first 130 lines of README, `src/Video.ts`, `src/ProjectBuilder.ts` |

License labels describe the inspected repository root. They do not relicense dependencies, model weights, generated assets or datasets. Poseidon uses the following design lessons through independent implementation, without vendoring these repositories.

### NexusFlow: explicit representations and execution

The [CLI](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/nexusflow/cli.py) separates parsing, JSON intermediate-representation export and execution. The [language reference](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/NEXUSFLOW_LANGUAGE.md) distinguishes a training metadata stub from a PyTorch training operation. The README describes SVG/PPM procedural output and OBJ/glTF/GLB metadata and scene exports; it explicitly bounds native 3D rendering claims.

Poseidon lesson: preserve a validated scene plan between prediction and rendering, and record which component actually produced each result. A successful export is not evidence of a learned visual generator. The bounded scene representation is the most direct connection among this group of projects.

### universal-modder: media pipelines need end-to-end verification

The [README](https://github.com/rehan-remade/universal-modder/blob/8370faa8e114baf33acdb23079aff552a7728c4b/README.md) describes agent skills, a CLI and knowledge notes around game modification. Media generation uses external fal services or a local ComfyUI server, with Blender for 3D-to-sprite rendering and FFmpeg for video. Its documented workflow tests modifications in the running game and records exact versions and verified behavior.

Poseidon lesson: export real files, open or decode them, and record compatible formats. This repository does not establish a single model with all media capabilities, and its commercial-game examples are not training data for Poseidon.

### REA: producer identity and reproducible evidence

[ADR-0001](https://github.com/morluto/rea/blob/84a17d55199b1b46493aea6bf4e41029a3b200fc/docs/adr/0001-provider-selection-and-analysis-profiles.md) explains deterministic provider selection, analysis-profile commitments and distinctions between availability and target support. It rejects treating a composite provider identity as evidence that several engines jointly produced a result. Its cache contract includes the producer version, normalized settings and target identity. The refreshed [ADR-0002](https://github.com/morluto/rea/blob/84a17d55199b1b46493aea6bf4e41029a3b200fc/docs/adr/0002-controlled-replay-authority-and-sandbox.md) explicitly states that the controlled JavaScript replay implementation was removed. Its bounded, content-addressed replay design remains historical context, not a supported current REA capability.

Poseidon lesson: attach checkpoint, configuration and output identity to an experiment; distinguish learned policy, language inference, solver and renderer. Reverse engineering functionality itself is not part of the model architecture, and this review did not independently verify REA's implementation status.

### Odysseus: runtime contracts and durable state

The [turn contract](https://github.com/odysseus-dev/odysseus/blob/a8c147b238db01dbd00b57774ed8b453f1e43971/docs/AGENT_TURN_CONTRACT.md) separates capability selection, available schemas and execution authority, and warns that routing success is not model accuracy. [Atomic IO](https://github.com/odysseus-dev/odysseus/blob/a8c147b238db01dbd00b57774ed8b453f1e43971/core/atomic_io.py) writes temporary sibling files, flushes, synchronizes and replaces them; it also defines transaction locks.

Poseidon lesson: make the selected backend visible and make interrupted state writes recoverable. This is a workspace/runtime precedent, not a pretrained model source. AGPL code was inspected but not copied into Poseidon.

### Periodic NTAG emulator: a narrow lifecycle analogy

The [manifest](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator/blob/98fd7478253f9041195adc8eedfa95eabe084aff/application.fam) declares an external Flipper app requiring GUI and NFC. The inspected [C source](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator/blob/98fd7478253f9041195adc8eedfa95eabe084aff/ntag213_rotator.c) maintains a timer, event queue, listener lifecycle and visible attempt/success/error counters. It uses an ISO14443-3A listener; its name alone is insufficient evidence of full NTAG memory/protocol behavior.

Poseidon lesson: model periodic work as state transitions and retain outcome counters. Relevance is limited to runtime lifecycle design. No NFC behavior or unlicensed code is incorporated.

### quibble-builds: bounded platform claims and build identity

The [README](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/README.md) describes an experimental extensible Windows bootloader. Its [amd64 workflow](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/.github/workflows/build-amd64.yml) checks out the triggering commit, builds debug and release configurations and names uploaded artifacts with that revision.

Poseidon lesson: bind exports and receipts to known versions and state tested platform boundaries. Bootloader implementation is unrelated to multimodal learning. Nothing from this project was installed, booted or integrated.

### SVideo: frames, timing and packaging

The [README](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/README.md) describes packing video frames into sprite sheets and eliminating consecutive near-duplicates. [Video.ts](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/src/Video.ts) reads stream metadata and represents frame rate as both a rational pair and numeric value. [ProjectBuilder.ts](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/src/ProjectBuilder.ts) assembles Scratch costume/audio records and writes the project description.

Poseidon lesson: time and encoded frame counts must be verified, especially when encoders collapse identical frames. SVideo converts existing media; it is not evidence for video understanding or neural video generation.

## Primary research and implementation references

- [DeepCAD, Wu, Xiao and Zheng, ICCV 2021](https://arxiv.org/abs/2105.09492) models shapes as sequences of CAD operations. This supports investigating compact structured output instead of treating meshes as unconstrained text. Poseidon's primitive labels are much simpler than DeepCAD's operation sequences; its datasets, weights and reported results were not adopted or reproduced.
- [Vitruvion, Seff et al.](https://arxiv.org/abs/2109.14124) autoregressively models sketch primitives and constraints. It is adjacent work demonstrating that geometry-aware representations precede Poseidon. Poseidon does not claim novel program-based graphics generation.
- [W3C SVG 1.1](https://www.w3.org/TR/SVG11/) specifies declarative vector shapes, transforms and animation. It provides a representation reference; Poseidon's present raster exporter uses Pillow and geometric projection rather than SVG execution.
- [Khronos glTF 2.0 specification](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc) defines scene nodes, meshes, materials, buffers, accessors and animation storage. Poseidon exports a self-contained glTF 2.0 static mesh with positions, normals and triangle indices. Motion currently belongs to GIF/MP4 output, not a glTF animation track.
- [PyTorch GRU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) supplies a practical recurrent building block. A small shared state with separately scored scene and action heads is feasible to prototype on CPU; useful transfer between tasks must be demonstrated by ablation, not assumed from shared parameters.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html) explain seeding and determinism limitations. [Thread controls](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html) support explicit CPU resource limits. Neither source promises cross-version or cross-platform bitwise equivalence or a specific training time.
- [ImageIO](https://pypi.org/project/ImageIO/) and [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) provide optional local video encoding. ImageIO needs NumPy and Pillow; FFmpeg is a separate executable delivered by supported wheels. The wrapper license does not replace the bundled encoder's own license.

## Implemented scene contract and its limits

`poseidon.media.render_scene(scene, output_dir, kind="image", seed=42)` requires all five fields:

| Field | Allowed values |
|---|---|
| shape | cube, sphere, pyramid, cylinder |
| color | red, blue, green, yellow, purple, cyan |
| motion | still, orbit, bounce, spin |
| count | integer 1, 2, or 3 |
| scale | small, medium, large |

The renderer never extracts these fields from a prompt. It creates a 512 x 384 PNG preview, a 320 x 240 GIF/optional MP4 with a 2.4-second timeline, or OBJ/MTL plus self-contained glTF. Static GIFs may contain one timed frame after encoder deduplication. Every export includes the validated scene, seed, method, limits and hashes. Meshes use a single solid color, a uniform scale and triangular surfaces; generated shapes are basic procedural assets, not arbitrary reconstructed objects. Low-resolution rendering uses orthographic projection and face shading, with no learned pixel decoder, photorealism, sound or physical material simulation.

`generate_scene_examples(n, seed=42, split="train")` creates original controlled-language supervision. Every prompt explicitly supplies every field; synonyms are part of this bounded grammar. There are 864 possible semantic scenes and 384 unique wording variants per scene. Complete semantic tuples are assigned with a fixed SHA-256 bucket: 0 development, 1 test, 2–9 training. Different seeds change ordering, not split membership. Tests check group and exact-prompt separation. The finite synthetic corpus tests composition inside this grammar; large row counts are not evidence of broad visual understanding or the diversity of natural images and videos.
