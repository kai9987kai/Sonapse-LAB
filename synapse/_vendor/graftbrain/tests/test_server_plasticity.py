import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from graftbrain.plasticity import PlasticityConfig
from graftbrain.runtime import Simulation
from graftbrain.server import make_app


def test_http_and_websocket_expose_current_candidate_and_active_support(tmp_path):
    sim = Simulation(["proxy:flywire", "proxy:manc"], str(tmp_path / "viewer.pt"),
                     max_neurons=40, microsteps=1, fresh=True,
                     plasticity_config=PlasticityConfig(mode="adaptive"))
    sim.step_once()
    sim._make_snapshot(0.0)

    async def check():
        async with TestClient(TestServer(make_app(sim))) as client:
            response = await client.get("/")
            assert response.status == 200
            assert 'id="plasticity"' in await response.text()
            response = await client.get("/api/topology")
            topology = await response.json()
            assert topology["edge_semantics"] == "candidate graft edge records"
            assert topology["plasticity"]["active"] < topology["graft_edges"]
            assert sum(s["active"] for s in topology["active_sites"]) == topology["plasticity"]["active"]
            async with client.ws_connect("/ws") as socket:
                snapshot = await socket.receive_json(timeout=5)
                assert snapshot["plasticity"]["mode"] == "adaptive"
                json.dumps(snapshot, allow_nan=False)

    try:
        asyncio.run(check())
    finally:
        sim.cam.close()
