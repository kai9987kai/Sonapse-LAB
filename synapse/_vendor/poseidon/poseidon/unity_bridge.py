"""Unity WorldLab Integration Bridge.

Provides a bidirectional local IPC / TCP socket interface connecting Supermix Beyond
with Unity 3D environments (e.g., Unity ML-Agents or custom procedural C# game loops).

Features:
1. Low-latency TCP socket server binding to 127.0.0.1.
2. Protocol: Newline-delimited JSON messages.
3. Message types:
   - 'handshake': Exchanges observation/action space contracts.
   - 'step': Receives Unity observation vector, queries the UncertaintyAwarePlanner
             and persistent memory, and returns risk-calibrated action decisions.
   - 'get_scene': Streams procedural ObjectCentricSceneGraph for Unity instantiation.
   - 'close': Terminates session cleanly.
4. Built-in mock client simulator for testing and validation without requiring an active Unity instance.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from .core import CoreRuntime
from .ensemble import RiskConfig, UncertaintyAwarePlanner
from .persistent_memory import CrossStepRecurrentMemory, EpisodicMemoryStore, EpisodicRecord
from .scene_graph import ObjectCentricSceneGraph, create_procedural_beyond_world
from .world import ACTIONS, N_ACTIONS, OBS_SIZE

logger = logging.getLogger("poseidon.unity_bridge")


class UnityBridgeServer:
    """Listens for connections from Unity C# client and serves Beyond model inferences."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8085,
        core_path: str = "runs/tidal_dagger/core.pt",
        scene_graph: ObjectCentricSceneGraph | None = None,
    ):
        self.host = host
        self.port = port
        self.runtime = CoreRuntime(core_path)
        self.planner = UncertaintyAwarePlanner(self.runtime, config=RiskConfig(horizon=2))
        self.memory = CrossStepRecurrentMemory()
        self.hidden_state = self.memory.init_state(1)
        self.episodic_store = EpisodicMemoryStore(capacity=1000)
        self.scene_graph = scene_graph or create_procedural_beyond_world()

        self._running = False
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.last_action = 0
        self.total_steps_served = 0

    def start(self) -> None:
        """Starts the bridge server in a background thread."""
        if self._running:
            return
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(0.5)

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("UnityBridgeServer listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stops the bridge server."""
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("UnityBridgeServer stopped.")

    def _listen_loop(self) -> None:
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
                self._handle_client(client_sock)
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    break

    def _handle_client(self, sock: socket.socket) -> None:
        sock.settimeout(2.0)
        buffer = ""
        try:
            while self._running:
                data = sock.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    req = json.loads(line)
                    res = self.process_request(req)
                    res_bytes = (json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8")
                    sock.sendall(res_bytes)
        except Exception as e:
            logger.debug("Client connection error: %s", e)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def process_request(self, req: dict[str, Any]) -> dict[str, Any]:
        """Dispatches an incoming JSON request."""
        cmd = req.get("command", "step")

        if cmd == "handshake":
            return {
                "status": "connected",
                "system": "Supermix Beyond — Adaptive Cognitive World Model",
                "obs_dim": OBS_SIZE,
                "n_actions": N_ACTIONS,
                "actions": list(ACTIONS),
                "planner": "UncertaintyAwarePlanner (3-head ensemble)",
            }

        elif cmd == "get_scene":
            return {
                "status": "ok",
                "scene": self.scene_graph.to_dict(),
            }

        elif cmd == "step":
            obs = req.get("observation", [0.0] * OBS_SIZE)
            if len(obs) < OBS_SIZE:
                obs = obs + [0.0] * (OBS_SIZE - len(obs))
            elif len(obs) > OBS_SIZE:
                obs = obs[:OBS_SIZE]

            # 1. Update persistent cross-step memory
            obs_t = torch.tensor([obs], dtype=torch.float32)
            act_t = torch.tensor([self.last_action], dtype=torch.long)
            self.hidden_state = self.memory.step(obs_t, act_t, self.hidden_state)

            # 2. Risk-sensitive MPC planning
            plan_out = self.planner.plan(obs)
            chosen_action = plan_out["action"]
            self.last_action = chosen_action
            self.total_steps_served += 1

            # 3. Store episodic experience
            pos = req.get("position", [0.0, 0.0])
            self.episodic_store.append(EpisodicRecord(
                tick=self.total_steps_served,
                x=int(pos[0]) if len(pos) > 0 else 0,
                y=int(pos[1]) if len(pos) > 1 else 0,
                observation=obs,
                action=chosen_action,
                reward=float(req.get("reward", 0.0)),
                uncertainty=plan_out["epistemic_uncertainty"],
                resource_found=req.get("resource_found", "none"),
            ))

            # 4. Apply physical action to scene graph
            self.scene_graph.apply_agent_impulse("agent", plan_out["action_name"])
            self.scene_graph.step_simulation(dt=0.1)

            return {
                "status": "ok",
                "action": chosen_action,
                "action_name": plan_out["action_name"],
                "epistemic_uncertainty": plan_out["epistemic_uncertainty"],
                "failure_probability": plan_out["failure_probability"],
                "is_high_surprise": plan_out["is_high_surprise"],
                "policy_prior": plan_out["policy_probs"],
                "step_count": self.total_steps_served,
            }

        elif cmd == "reset":
            self.hidden_state = self.memory.init_state(1)
            self.last_action = 0
            return {"status": "reset_complete"}

        elif cmd == "ping":
            return {"status": "pong", "time": time.time()}

        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}


def simulate_unity_session(port: int = 8089, steps: int = 10) -> dict[str, Any]:
    """Runs a simulated Unity client session to verify IPC loop."""
    server = UnityBridgeServer(port=port)
    server.start()
    time.sleep(0.1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(("127.0.0.1", port))
        
        # 1. Handshake
        client.sendall(b'{"command": "handshake"}\n')
        handshake_res = json.loads(client.recv(4096).decode("utf-8").strip())

        # 2. Get Scene
        client.sendall(b'{"command": "get_scene"}\n')
        scene_res = json.loads(client.recv(8192).decode("utf-8").strip())

        # 3. Step loop
        step_logs = []
        for i in range(steps):
            obs = [0.8, 0.7, 0.6, 0.5, 0.1, 0.0, 0.4, 0.5, 0.8, 0.1, 0.4, 0.5, 0.2, 0.3, 0.0, i / steps]
            req = {
                "command": "step",
                "observation": obs,
                "position": [float(i), float(i % 3)],
                "reward": 0.1,
            }
            client.sendall((json.dumps(req) + "\n").encode("utf-8"))
            step_res = json.loads(client.recv(4096).decode("utf-8").strip())
            step_logs.append(step_res)

        return {
            "handshake": handshake_res,
            "scene_objects_count": len(scene_res.get("scene", {}).get("objects", {})),
            "steps_completed": len(step_logs),
            "final_action": step_logs[-1]["action_name"],
            "mean_uncertainty": round(sum(s["epistemic_uncertainty"] for s in step_logs) / len(step_logs), 5),
        }
    finally:
        client.close()
        server.stop()
