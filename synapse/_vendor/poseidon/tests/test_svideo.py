"""Tests for S-Video & Analog Scanline Video Telemetry Engine."""
from poseidon.svideo import SVideoTelemetryEngine, ScanlineSignal


def test_svideo_scanline_synthesis():
    engine = SVideoTelemetryEngine()
    engine.reset()

    scanline = engine.synthesize_scanline(
        line_number=100,
        luminance_profile=[0.8, 0.5, 0.9],
        chrominance_u=0.5,
        chrominance_v=-0.5,
    )

    assert len(scanline.y_samples) == ScanlineSignal.TOTAL_SAMPLES
    assert len(scanline.c_samples) == ScanlineSignal.TOTAL_SAMPLES
    # Sync tip check
    assert scanline.y_samples[0] == -40.0
    # Breezeway check
    assert scanline.y_samples[70] == 0.0
    # Active video check (should be between 7.5 and 100 IRE)
    assert 7.5 <= scanline.y_samples[200] <= 100.0


def test_svideo_pcm_and_svg_export():
    engine = SVideoTelemetryEngine()
    scanline = engine.synthesize_scanline(
        line_number=50,
        luminance_profile=[0.6, 0.4],
        chrominance_u=0.2,
        chrominance_v=0.1,
    )

    pcm_bytes = scanline.to_pcm_bytes()
    assert len(pcm_bytes) == ScanlineSignal.TOTAL_SAMPLES
    assert isinstance(pcm_bytes, bytes)

    svg = scanline.to_svg_oscillogram()
    assert "<svg" in svg
    assert "100 IRE (Peak White)" in svg
    assert "-40 IRE (Sync Tip)" in svg


def test_svideo_encode_telemetry_frame():
    engine = SVideoTelemetryEngine()
    engine.reset()

    vitals = {"health": 0.9, "energy": 0.8, "hydration": 0.7, "stamina": 0.6, "exposure": 0.1}
    frame = engine.encode_telemetry_frame(vitals, lattice_pos=[1.0, 2.0, 0.0], quantum_phases=[0.5, -0.5])

    assert frame["frame_index"] == 1
    assert 1 <= frame["line_number"] <= 525
    assert frame["samples_per_line"] == 910
    assert "pcm_base64" in frame
    assert "<svg" in frame["svg_oscillogram"]
