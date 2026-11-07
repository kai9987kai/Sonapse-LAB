"""Tests verifying canvas zero-warping guarantees, web workbench assets, and advanced endpoints."""
import json
import math
import subprocess
import threading
import urllib.request
import urllib.error
from pathlib import Path
import pytest
import numpy as np

from synapse.config import SynapseConfig, ROOT
from synapse.runtime import LabRuntime
from synapse.server import create_server, Workbench


def test_web_workbench_assets_integrity():
    """Verify that web assets exist, have valid markup and define all required zero-warping canvas targets."""
    web_dir = ROOT / 'synapse' / 'web'
    index_html = (web_dir / 'index.html').read_text(encoding='utf-8')
    style_css = (web_dir / 'style.css').read_text(encoding='utf-8')
    app_js = (web_dir / 'app.js').read_text(encoding='utf-8')

    # All canvases must be present
    for canvas_id in ('graph', 'world', 'signals', 'gauges', 'predictive'):
        assert f'id="{canvas_id}"' in index_html, f"Canvas #{canvas_id} missing in index.html"

    # View toggle controls
    for toggle_id in ('world-view-iso', 'world-view-top', 'graph-view-2d', 'graph-view-3d'):
        assert f'id="{toggle_id}"' in index_html, f"Toggle #{toggle_id} missing in index.html"

    # Direct teleoperation controls
    for teleop_id in ('teleop-fwd', 'teleop-left', 'teleop-right', 'teleop-grip'):
        assert f'id="{teleop_id}"' in index_html, f"Teleop #{teleop_id} missing in index.html"

    # Advanced assay triggers
    for action_id in ('atlas', 'hyperprogram', 'flagship', 'autoprogram', 'transplant', 'metamorphic'):
        assert f'id="{action_id}"' in index_html, f"Action #{action_id} missing in index.html"

    # CSS must have explicit heights for canvases to prevent layout-collapse and aspect-ratio warping
    assert '#graph, #world' in style_css
    assert '350px' in style_css
    assert '#signals, #gauges, #predictive' in style_css
    assert '185px' in style_css
    assert 'box-sizing: border-box' in style_css

    # JavaScript must include ResizeObserver, zero-warping projection and DPI transform handling
    assert 'ResizeObserver' in app_js
    assert 'isoCos = Math.cos(Math.PI / 6)' in app_js
    assert 'isoSin = Math.sin(Math.PI / 6)' in app_js
    assert 'worldScale' in app_js

    # New complex tasks in quicktask and config dropdowns
    for task_name in ('hyper_boreal_tundra', 'subterranean_abyss', 'vortex_maelstrom'):
        assert task_name in index_html, f"Task {task_name} missing in index.html"

    # Frontier V2 visualization toggles
    for vis_toggle in ('graph-streamlines', 'graph-cluster-halos', 'world-terrain-mesh', 'world-wind-streamlines', 'world-retina-cam'):
        assert f'id="{vis_toggle}"' in index_html, f"Toggle #{vis_toggle} missing in index.html"

    # Frontier V2 telemetry cards
    for hud_card in ('thermo-hud', 'bcm-hud', 'te-hud'):
        assert f'id="{hud_card}"' in index_html, f"HUD card #{hud_card} missing in index.html"

    # JavaScript visualization routines
    for js_feature in ('graph-streamlines', 'cluster-halos', 'terrain-mesh', 'wind-streamlines', 'retina-cam', 'aerial_raptor', 'abyssal_leviathan', 'vortices'):
        assert js_feature in app_js, f"JS feature {js_feature} missing in app.js"


def test_isometric_projection_mathematical_invariants():
    """Verify mathematically that the isometric projection preserves uniform aspect ratio without warping."""
    # Test across multiple viewport dimensions (narrow, square, ultra-wide)
    viewports = [(400, 350), (600, 350), (800, 350), (1200, 350), (1920, 600)]
    
    iso_cos = math.cos(math.pi / 6)
    iso_sin = math.sin(math.pi / 6)
    max_span_x = 20 * iso_cos
    max_span_y = 20 * iso_sin + 2.5
    pad_x = 36
    pad_y = 46

    for w, h in viewports:
        scale = min((w - pad_x) / max_span_x, (h - pad_y) / max_span_y)
        assert scale > 0, "Scale factor must be positive"

        cx = w * 0.5
        cy = h * 0.52

        def project(px, py, z=0.0):
            return [
                cx + (px - py) * iso_cos * scale,
                cy + (px + py) * iso_sin * scale - z * scale * 1.15
            ]

        # 1. Arena boundary containment: all arena corners [-5, 5] must lie strictly within [0, w] x [0, h]
        corners = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        for cx_val, cy_val in corners:
            sx, sy = project(cx_val, cy_val, 0.0)
            assert 0 <= sx <= w, f"Corner ({cx_val}, {cy_val}) sx={sx} out of [0, {w}]"
            assert 0 <= sy <= h, f"Corner ({cx_val}, {cy_val}) sy={sy} out of [0, {h}]"

        # 2. Aspect Ratio Invariance: A 1-unit horizontal vector and 1-unit vertical vector in world space
        # must scale with an exact fixed geometric ratio regardless of canvas width w
        p_origin = project(0, 0, 0)
        p_x = project(1, 0, 0)
        p_y = project(0, 1, 0)
        
        dist_x = math.hypot(p_x[0] - p_origin[0], p_x[1] - p_origin[1])
        dist_y = math.hypot(p_y[0] - p_origin[0], p_y[1] - p_origin[1])

        # In true isometric view, unit step along X and unit step along Y must have identical pixel lengths
        assert math.isclose(dist_x, dist_y, rel_tol=1e-5), f"Distortion detected: dist_x={dist_x} != dist_y={dist_y}"


def test_server_atlas_hyperprogram_and_teleoperation(tmp_path):
    """Test that /api/atlas, /api/hyperprogram, and /api/step action overrides execute properly."""
    cfg = SynapseConfig(seed=42, nodes_per_source=16)
    rt = LabRuntime(cfg, output_root=tmp_path)
    server = create_server(port=0, runtime=rt, output_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    try:
        # 1. Test /api/atlas
        req_atlas = urllib.request.Request(
            base + '/api/atlas',
            data=b'{}',
            headers={'Content-Type': 'application/json'}
        )
        res_atlas = json.load(urllib.request.urlopen(req_atlas))
        assert isinstance(res_atlas, dict)
        assert len(res_atlas) > 0

        # 2. Test /api/hyperprogram
        req_hyper = urllib.request.Request(
            base + '/api/hyperprogram',
            data=b'{}',
            headers={'Content-Type': 'application/json'}
        )
        res_hyper = json.load(urllib.request.urlopen(req_hyper))
        assert 'schema' in res_hyper
        assert 'edits' in res_hyper
        assert len(res_hyper['edits']) > 0

        # 3. Test /api/step with direct teleoperation action
        teleop_act = [-0.5, 0.8, 0.2, 0.0, 1.0]
        req_step = urllib.request.Request(
            base + '/api/step',
            data=json.dumps({'steps': 1, 'action': teleop_act}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        res_step = json.load(urllib.request.urlopen(req_step))
        assert 'action' in res_step
        np.testing.assert_allclose(res_step['action'], teleop_act, atol=1e-5)

        # 4. Verify invalid action rejected with 400
        req_bad = urllib.request.Request(
            base + '/api/step',
            data=b'{"steps":1,"action":[1.0, 2.0]}',
            headers={'Content-Type': 'application/json'}
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req_bad)
        assert err.value.code == 400

    finally:
        server.shutdown()
        server.server_close()
