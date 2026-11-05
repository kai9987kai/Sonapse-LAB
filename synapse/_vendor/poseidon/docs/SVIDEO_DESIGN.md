# S-Video & Analog Scanline Video Telemetry Engine Design

## 1. Overview and Theoretical Formulation

The **S-Video & Analog Scanline Video Telemetry Engine** (`poseidon/svideo.py`) synthesizes authentic NTSC / S-Video (Y/C separate) analog television broadcast waveforms from agent cognitive states, vital telemetry, and quantum registers, inspired by `svideo` and `quibble-builds`.

### NTSC Waveform Specifications & Timing

Each scanline adheres strictly to the NTSC RS-170A standard with a total horizontal line time of $63.555\ \mu\text{s}$:
- **Sampling Frequency**: $4\times F_{\text{sc}} = 14.31818\text{ MHz}$ ($910\text{ discrete samples per scanline}$).
- **Sync Tip**: $-40\text{ IRE}$ ($4.7\ \mu\text{s}$, samples 0–68).
- **Breezeway**: $0\text{ IRE}$ ($0.6\ \mu\text{s}$, samples 68–76).
- **Colorburst**: $3.579545\text{ MHz}$ sine wave, $40\text{ IRE}$ peak-to-peak ($2.5\ \mu\text{s}$, samples 76–112).
- **Back Porch**: $0\text{ IRE}$ ($1.6\ \mu\text{s}$, samples 112–135).
- **Active Video**: $7.5\text{ IRE}$ (black setup pedestal) to $100\text{ IRE}$ (peak white) ($52.6\ \mu\text{s}$, samples 135–888).
- **Front Porch**: $0\text{ IRE}$ ($1.5\ \mu\text{s}$, samples 888–910).

### Telemetry Mapping to Y/C Channels

1. **Luminance Channel ($Y$)**: Encodes scalar survival vitals (health, energy, hydration, stamina) into active video luminance levels between 7.5 IRE and 100.0 IRE.
2. **Chrominance Channel ($C$)**: Modulates 3D diamond lattice coordinates and Causeway quantum phases onto the 3.58 MHz color subcarrier via Quadrature Amplitude Modulation (QAM):
   $$C(t) = I(t) \cos(2\pi F_{\text{sc}} t) + Q(t) \sin(2\pi F_{\text{sc}} t)$$
3. **Signal Export**: Supports both unsigned 8-bit PCM byte quantization (for SDR / FPGA transmission) and inline SVG oscillograms with exact IRE gridlines.

---

## 2. API Reference

```python
from poseidon.svideo import SVideoTelemetryEngine

engine = SVideoTelemetryEngine()
engine.reset()

scanline = engine.encode_telemetry_frame(
    vitals={"health": 0.85, "energy": 0.70, "hydration": 0.90, "stamina": 0.65},
    lattice_pos=[0.25, 0.50, -0.15],
    quantum_phases=[0.0, 1.57, 3.14, 0.78, 2.35, 4.71],
)

# Export to 8-bit PCM bytes (910 samples)
pcm_data = scanline.to_pcm_bytes()
assert len(pcm_data) == 910

# Export to vector SVG oscillogram
svg_markup = scanline.to_svg_oscillogram(width=910, height=240)
```

---

## 3. Empirical Verification

Unit test coverage in `tests/test_svideo.py` validates:
- Signal timing matches the 910-sample 4x Fsc grid precisely.
- Sync tip adheres strictly to -40.0 IRE.
- Active video stays bounded within standard IRE range [7.5, 100.0].
- SVG oscillograms render valid SVG markup with IRE reference lines.
