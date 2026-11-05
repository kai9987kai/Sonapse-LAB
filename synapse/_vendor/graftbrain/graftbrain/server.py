"""Local web server (127.0.0.1 only): live 3D view of the world + brain telemetry over WebSocket."""
from __future__ import annotations

import asyncio
import json
import os

from aiohttp import WSMsgType, web

WEB = os.path.join(os.path.dirname(__file__), "web")


def make_app(sim):
    app = web.Application()

    async def index(_):
        return web.FileResponse(os.path.join(WEB, "index.html"))

    async def topology(_):
        snapshot = sim.get_snapshot()
        description = {**sim.graph.describe(), "edge_semantics": "candidate graft edge records",
                       "plasticity": snapshot.get("plasticity", {}),
                       "active_sites": snapshot.get("sites", [])}
        return web.json_response(description, dumps=lambda o: json.dumps(o, default=str))

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=20)
        await sock.prepare(request)

        async def pump():
            while not sock.closed:
                snap = sim.get_snapshot()
                if snap:
                    await sock.send_str(json.dumps(snap, default=float))
                await asyncio.sleep(1 / 15)

        task = asyncio.create_task(pump())
        try:
            async for msg in sock:
                if msg.type == WSMsgType.TEXT:
                    sim.cmds.put(json.loads(msg.data))
        finally:
            task.cancel()
        return sock

    app.router.add_get("/", index)
    app.router.add_get("/api/topology", topology)
    app.router.add_get("/ws", ws)
    return app


def serve(sim, port=8770):
    print(f"[viewer] open http://127.0.0.1:{port}")
    web.run_app(make_app(sim), host="127.0.0.1", port=port, print=None)
