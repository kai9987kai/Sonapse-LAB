# INTERMITTENT: Passive Energy Harvesting & NTAG Ephemeral Resuscitation

## 1. Motivation & Principles

Inspired by `Flipper-Zero-Periodic-NTAG-Emulator`, the **Intermittent Computing** module introduces battery-less, power-harvesting state persistence into the agent's cognitive architecture.

In intermittent computing, embedded devices running without batteries periodically harvest energy from ambient radio-frequency (RF) fields and execute non-volatile state checkpointing. When power fails, the system undergoes a blackout, but as soon as ambient power recharges an internal capacitor, the device instantly resumes from the checkpoint.

In Poseidon TidePool:
- **Passive Energy Harvesting:** When stationary in shelter, foraging, or resting near resource patches, the agent inductively harvests ambient potential into a virtual capacitor $C_{\text{harvest}}$.
- **Compact Binary NTAG Checkpoint Frame:** At regular intervals or when vitals enter danger zones, a 38-byte NFC Type-2 / NTAG 215 compatible frame is packed with a cryptographic SHA-256 seal.
- **Metabolic Resuscitation:** If stamina falls below critical threshold ($< 0.08$), the virtual capacitor discharges an emergency metabolic boost, preventing blackout death and recovering coherent quantum phases.

---

## 2. NTAG 215 Binary Frame Specification (38 Bytes)

| Offset | Length | Field | Data Type | Encoding / Scaling |
|---|---|---|---|---|
| `0x00` | 4 bytes | Magic Header | ASCII | `b"NTAG"` (`0x4E544147`) |
| `0x04` | 4 bytes | Step Counter | Big-Endian uint32 | Simulation tick |
| `0x08` | 2 bytes | Health | Big-Endian uint16 | Scaled $[0, 65535]$ |
| `0x0A` | 2 bytes | Stamina | Big-Endian uint16 | Scaled $[0, 65535]$ |
| `0x0C` | 6 bytes | 3D Lattice Pos | 3 x int16 | Geodesic coordinates $(x, y, z) \times 100$ |
| `0x12` | 12 bytes | Quantum Phases | 6 x uint16 | Phase angles $\theta_0 \dots \theta_5$ normalized to $[0, 2\pi)$ |
| `0x1E` | 8 bytes | Checksum | Truncated SHA-256 | Digital authenticity seal |

---

## 3. Resuscitation Protocol

When:
$$\text{Stamina}_t < \theta_{\text{resuscitation}} \quad \text{or} \quad \text{Health}_t < \theta_{\text{resuscitation}}$$
and $C_{\text{harvest}} \ge 0.20$, the resuscitation routine triggers:
1. Virtual capacitor discharges an emergency energy boost $\Delta E = \min(C_{\text{harvest}}, \text{boost})$.
2. The agent recovers stamina to continue foraging.
3. Coherent phase angles are stabilized, avoiding erratic random thrashing during crises.
