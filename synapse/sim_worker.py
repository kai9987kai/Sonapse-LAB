"""Server-Owned Background Simulation Worker for Uncapped / Maximum Throughput.

Grounding & Safety Invariants:
- Uses an explicit re-entrant threading lock (``runtime_lock``) to prevent concurrent step() mutations.
- Runs non-blocking atomic batches in a background daemon thread while UI polls at a bounded rate.
- Tracks genuine measured completed steps/second, inference latency, and CPU time without falsified metrics.
- Provides predictable start, pause, stop, step-quota limits, and graceful checkpoint-on-exit.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional
import numpy as np


class SimWorker:
    """Backend-owned background execution worker for high-throughput and uncapped simulation."""

    def __init__(self, runtime, lock: Optional[Any] = None, default_batch_size: int = 16, max_steps_per_tick: Optional[int] = None):
        self.runtime = runtime
        self.default_batch_size = max(1, int(max_steps_per_tick or default_batch_size))
        self.lock = lock or threading.RLock()

        self.running: bool = False
        self.paused: bool = False
        self._stop_requested: bool = False
        self.thread: Optional[threading.Thread] = None
        self.target_sps: Optional[float] = None
        self.learn: bool = False
        self.action_vec: Optional[np.ndarray] = None
        self.quota_remaining: Optional[int] = None

        # Telemetry
        self.start_wallclock: float = 0.0
        self.start_cputime: float = 0.0
        self.steps_completed: int = 0
        self.batches_completed: int = 0
        self.recent_latencies_ms: list[float] = []
        self.last_sps_measured: float = 0.0

    def set_target_sps(self, sps: Optional[float]) -> None:
        self.target_sps = float(sps) if (sps is not None and sps > 0) else None

    def start(
        self,
        batch_size: Optional[int] = None,
        learn: bool = False,
        quota: Optional[int] = None,
        action: Optional[list] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if self.running and not self.paused:
                return self.status()

            if batch_size is not None:
                self.default_batch_size = max(1, min(256, int(batch_size)))
            self.learn = bool(learn)
            self.quota_remaining = int(quota) if quota is not None and quota > 0 else None
            if action is not None and len(action) == 5:
                self.action_vec = np.asarray(action, dtype=np.float32)
            else:
                self.action_vec = None

            self.paused = False
            self._stop_requested = False
            self.start_wallclock = time.perf_counter()
            self.start_cputime = time.process_time()
            self.steps_completed = 0
            self.batches_completed = 0
            self.recent_latencies_ms = []

            if self.thread is None or not self.thread.is_alive():
                self.running = True
                self.thread = threading.Thread(target=self._run_loop, daemon=True, name="SynapseSimWorker")
                self.thread.start()

        return self.status()

    def pause(self) -> Dict[str, Any]:
        with self.lock:
            self.paused = True
        return self.status()

    def resume(self) -> Dict[str, Any]:
        with self.lock:
            self.paused = False
        return self.status()

    def stop(self, save_checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            self._stop_requested = True
            self.paused = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self.lock:
            self.running = False
            self.thread = None
            if save_checkpoint_name:
                try:
                    self.runtime.save(save_checkpoint_name)
                except Exception:
                    pass

        return self.status()

    def _run_loop(self) -> None:
        while not self._stop_requested:
            if self.paused:
                time.sleep(0.02)
                continue

            with self.lock:
                if self._stop_requested:
                    break

                batch = self.default_batch_size
                if self.quota_remaining is not None:
                    if self.quota_remaining <= 0:
                        break
                    batch = min(batch, self.quota_remaining)

                t0 = time.perf_counter()
                try:
                    self.runtime.step(steps=batch, learn=self.learn, action=self.action_vec)
                except Exception:
                    # In case of runtime error, exit loop gracefully
                    break

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.steps_completed += batch
                self.batches_completed += 1
                if self.quota_remaining is not None:
                    self.quota_remaining -= batch

                self.recent_latencies_ms.append(elapsed_ms)
                if len(self.recent_latencies_ms) > 32:
                    self.recent_latencies_ms = self.recent_latencies_ms[-32:]

            # Yield briefly to avoid complete CPU starvation of the web server
            time.sleep(0.001)

        with self.lock:
            self.running = False
            self.paused = False

    def status(self) -> Dict[str, Any]:
        with self.lock:
            wallclock = max(0.001, time.perf_counter() - self.start_wallclock) if self.running else 0.0
            cputime = max(0.0, time.process_time() - self.start_cputime) if self.running else 0.0
            sps = (float(self.steps_completed) / wallclock) if (self.running and wallclock > 0.01) else 0.0
            mean_lat = float(np.mean(self.recent_latencies_ms)) if self.recent_latencies_ms else 0.0

            return {
                "running": bool(self.running),
                "paused": bool(self.paused),
                "steps_completed": int(self.steps_completed),
                "batches_completed": int(self.batches_completed),
                "batch_size": int(self.default_batch_size),
                "learn": bool(self.learn),
                "quota_remaining": self.quota_remaining,
                "measured_steps_per_sec": round(sps, 2),
                "mean_batch_latency_ms": round(mean_lat, 2),
                "elapsed_wallclock_sec": round(wallclock, 2),
                "cpu_seconds": round(cputime, 2),
                "total_runtime_steps": int(self.runtime.total_steps),
            }


# Backwards-compatible alias
SimulationWorker = SimWorker
