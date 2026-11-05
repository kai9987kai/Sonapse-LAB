"""S-Video & Analog Scanline Video Telemetry Engine.

Synthesizes authentic NTSC / PAL composite and S-Video (Y/C) analog television
waveforms from agent vitals, 3D Diamond Lattice coordinates, and Causeway quantum phases:
1. Strict NTSC signal timing and voltage levels in standard IRE units (-40 IRE to 100 IRE).
2. Separate Y (Luminance, 0 to 100 IRE) and C (Chrominance, 3.579545 MHz subcarrier QAM) channels.
3. Colorburst reference sync (9 cycles at 3.58 MHz, 40 IRE peak-to-peak).
4. Discrete 4x Fsc sampling (14.31818 MHz = 910 samples per 63.555 us scanline).
5. SVG oscillogram and raw byte PCM export for physical signal auditing.
"""
from __future__ import annotations

import base64
import math
from typing import Dict, List, Tuple


class ScanlineSignal:
    """Discrete representation of a single analog NTSC scanline."""

    TOTAL_SAMPLES: int = 910  # 4 * 3.579545 MHz * 63.555 us = 910 samples/line
    # Standard NTSC sample segment boundaries at 14.31818 MHz
    SYNC_TIP_END: int = 68       # 4.7 us sync tip (-40 IRE)
    BREEZEWAY_END: int = 76      # 0.6 us breezeway (0 IRE)
    BURST_END: int = 112         # 2.5 us colorburst (3.58 MHz sine)
    BACK_PORCH_END: int = 135    # 1.6 us back porch (0 IRE)
    ACTIVE_VIDEO_END: int = 888  # 52.6 us active video (7.5 to 100 IRE)
    # Remaining 22 samples (888 to 910) = 1.5 us front porch (0 IRE)

    def __init__(self, line_number: int, y_samples: List[float], c_samples: List[float]) -> None:
        self.line_number = line_number
        self.y_samples = y_samples  # Luminance in IRE [-40, 100]
        self.c_samples = c_samples  # Chrominance in IRE [-20, 20]

    def to_pcm_bytes(self) -> bytes:
        """Quantize composite signal (Y + C) to unsigned 8-bit PCM (0 = -40 IRE, 255 = +100 IRE)."""
        buffer = bytearray(self.TOTAL_SAMPLES)
        for i in range(self.TOTAL_SAMPLES):
            composite = self.y_samples[i] + self.c_samples[i]
            # Map [-40, 100] IRE to [0, 255]
            norm = (composite + 40.0) / 140.0
            quantized = int(max(0.0, min(1.0, norm)) * 255.0)
            buffer[i] = quantized
        return bytes(buffer)

    def to_svg_oscillogram(self, width: int = 910, height: int = 240) -> str:
        """Render an SVG oscillogram waveform of this scanline with IRE markings."""
        # IRE scale: 100 IRE -> y=30, 0 IRE -> y=150, -40 IRE -> y=210
        def ire_to_y(ire: float) -> float:
            return 150.0 - (ire / 100.0) * 120.0

        points = []
        for x, (y_val, c_val) in enumerate(zip(self.y_samples, self.c_samples)):
            comp = y_val + c_val
            py = round(ire_to_y(comp), 1)
            points.append(f"{x},{py}")

        path_data = "M " + " L ".join(points)
        y0 = ire_to_y(0.0)
        y100 = ire_to_y(100.0)
        ym40 = ire_to_y(-40.0)
        y75 = ire_to_y(7.5)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background:#0a0e14;font-family:monospace;font-size:10px;">
  <!-- Grid Lines -->
  <line x1="0" y1="{y100}" x2="{width}" y2="{y100}" stroke="#2e3846" stroke-dasharray="4,4"/>
  <text x="5" y="{y100 - 4}" fill="#7ee787">100 IRE (Peak White)</text>
  <line x1="0" y1="{y75}" x2="{width}" y2="{y75}" stroke="#2e3846" stroke-dasharray="2,2"/>
  <text x="5" y="{y75 - 3}" fill="#8b949e">7.5 IRE (Black Setup)</text>
  <line x1="0" y1="{y0}" x2="{width}" y2="{y0}" stroke="#484f58"/>
  <text x="5" y="{y0 - 4}" fill="#58a6ff">0 IRE (Blanking Level)</text>
  <line x1="0" y1="{ym40}" x2="{width}" y2="{ym40}" stroke="#2e3846" stroke-dasharray="4,4"/>
  <text x="5" y="{ym40 + 12}" fill="#f85149">-40 IRE (Sync Tip)</text>

  <!-- Section Markers -->
  <line x1="{self.SYNC_TIP_END}" y1="0" x2="{self.SYNC_TIP_END}" y2="{height}" stroke="#30363d" stroke-dasharray="2,2"/>
  <text x="{self.SYNC_TIP_END + 2}" y="20" fill="#8b949e">Sync</text>
  <line x1="{self.BURST_END}" y1="0" x2="{self.BURST_END}" y2="{height}" stroke="#30363d" stroke-dasharray="2,2"/>
  <text x="{self.BURST_END - 30}" y="35" fill="#e3b341">Burst (3.58 MHz)</text>
  <line x1="{self.BACK_PORCH_END}" y1="0" x2="{self.BACK_PORCH_END}" y2="{height}" stroke="#30363d"/>
  <text x="{self.BACK_PORCH_END + 10}" y="20" fill="#388bfd">Active Video (Line {self.line_number})</text>

  <!-- Waveform Trace -->
  <path d="{path_data}" fill="none" stroke="#39d353" stroke-width="1.5"/>
</svg>"""
        return svg


class SVideoTelemetryEngine:
    """Analog composite and S-Video scanline telemetry encoder."""

    SUBCARRIER_FREQ_HZ: float = 3579545.0  # 3.579545 MHz
    SAMPLING_RATE_HZ: float = 14318180.0  # 4 * Fsc = 14.31818 MHz

    def __init__(self) -> None:
        self.frame_count: int = 0
        self.last_scanline: ScanlineSignal | None = None

    def reset(self) -> None:
        self.frame_count = 0
        self.last_scanline = None

    def synthesize_scanline(
        self,
        line_number: int,
        luminance_profile: List[float],
        chrominance_u: float,
        chrominance_v: float,
    ) -> ScanlineSignal:
        """Synthesize a complete 910-sample NTSC scanline."""
        y_samples = [0.0] * ScanlineSignal.TOTAL_SAMPLES
        c_samples = [0.0] * ScanlineSignal.TOTAL_SAMPLES

        # 1. Sync tip (-40 IRE)
        for i in range(ScanlineSignal.SYNC_TIP_END):
            y_samples[i] = -40.0

        # 2. Breezeway (0 IRE)
        for i in range(ScanlineSignal.SYNC_TIP_END, ScanlineSignal.BREEZEWAY_END):
            y_samples[i] = 0.0

        # 3. Colorburst (3.58 MHz, 9 cycles, 40 IRE peak-to-peak -> amplitude = 20.0)
        # At 4x Fsc, 1 subcarrier cycle = 4 samples.
        for i in range(ScanlineSignal.BREEZEWAY_END, ScanlineSignal.BURST_END):
            cycle_phase = (2.0 * math.pi / 4.0) * float(i - ScanlineSignal.BREEZEWAY_END)
            c_samples[i] = 20.0 * math.sin(cycle_phase)
            y_samples[i] = 0.0

        # 4. Back porch (0 IRE)
        for i in range(ScanlineSignal.BURST_END, ScanlineSignal.BACK_PORCH_END):
            y_samples[i] = 0.0

        # 5. Active video (7.5 IRE to 100 IRE)
        active_len = ScanlineSignal.ACTIVE_VIDEO_END - ScanlineSignal.BACK_PORCH_END
        prof_len = max(1, len(luminance_profile))

        # Modulate QAM Chrominance: C(t) = U * sin(omega * t) + V * cos(omega * t)
        for idx in range(active_len):
            i = ScanlineSignal.BACK_PORCH_END + idx
            # Sample luminance profile
            prof_idx = min(prof_len - 1, int((idx / active_len) * prof_len))
            norm_lum = max(0.0, min(1.0, luminance_profile[prof_idx]))
            # NTSC active luminance: 7.5 IRE (setup) to 100 IRE
            y_samples[i] = 7.5 + norm_lum * 92.5

            # 4 samples per cycle
            phase = (math.pi / 2.0) * float(idx)
            c_val = chrominance_u * 15.0 * math.sin(phase) + chrominance_v * 15.0 * math.cos(phase)
            c_samples[i] = max(-20.0, min(20.0, c_val))

        # 6. Front porch (0 IRE)
        for i in range(ScanlineSignal.ACTIVE_VIDEO_END, ScanlineSignal.TOTAL_SAMPLES):
            y_samples[i] = 0.0

        return ScanlineSignal(line_number, y_samples, c_samples)

    def encode_telemetry_frame(
        self,
        vitals: Dict[str, float],
        lattice_pos: List[float],
        quantum_phases: List[float],
    ) -> Dict[str, object]:
        """Encode agent state vector into a verifiable S-Video scanline frame."""
        self.frame_count += 1

        # Profile assembled from vitals: health, energy, hydration, stamina, exposure
        profile = [
            vitals.get("health", 1.0),
            vitals.get("energy", 1.0),
            vitals.get("hydration", 1.0),
            vitals.get("stamina", 1.0),
            1.0 - vitals.get("exposure", 0.0),
        ]

        # Chrominance U/V derived from quantum phases and 3D diamond lattice
        # U encodes mean quantum phase alignment; V encodes 3D lattice distance
        u_chroma = math.sin(quantum_phases[0]) if quantum_phases else 0.0
        v_chroma = math.cos(quantum_phases[1]) if len(quantum_phases) > 1 else 0.0
        if lattice_pos:
            lattice_radius = math.sqrt(sum(p * p for p in lattice_pos[:3]))
            v_chroma = 0.5 * v_chroma + 0.5 * math.sin(lattice_radius)

        scanline = self.synthesize_scanline(
            line_number=(self.frame_count % 525) + 1,
            luminance_profile=profile,
            chrominance_u=u_chroma,
            chrominance_v=v_chroma,
        )
        self.last_scanline = scanline

        pcm_bytes = scanline.to_pcm_bytes()
        svg_osc = scanline.to_svg_oscillogram()

        return {
            "frame_index": self.frame_count,
            "line_number": scanline.line_number,
            "samples_per_line": ScanlineSignal.TOTAL_SAMPLES,
            "subcarrier_mhz": round(self.SUBCARRIER_FREQ_HZ / 1e6, 4),
            "chrominance_u": round(u_chroma, 4),
            "chrominance_v": round(v_chroma, 4),
            "pcm_base64": base64.b64encode(pcm_bytes[:64]).decode("ascii"),
            "svg_oscillogram": svg_osc,
        }
