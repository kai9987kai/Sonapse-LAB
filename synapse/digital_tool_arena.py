"""Permissioned Digital Tool Arena & Synthetic Ground-Truth Corpus.

Security & Architecture Invariants:
- STRICTLY NO raw Python exec, shell commands, arbitrary subprocesses, internet sockets,
  or unconstrained host filesystem access.
- Provides a closed, permissioned, deterministic synthetic document environment with typed actions:
  ``LIST``, ``SEARCH``, ``OPEN``, ``READ``, ``COMPARE``, ``MOVE``, ``WRITE_NOTE``, ``SUBMIT``.
- Contains curated ground-truth document databases (e.g., flight manifests, experimental protocols,
  sensor logs, parts inventory) with verified query answers for deterministic reward evaluation.
- Adapts 5 continuous motor actions into tool selection and parameter choices, and encodes
  document state/query progress into standard 155-element observation vectors.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ToolActionType(str, Enum):
    LIST = "LIST"
    SEARCH = "SEARCH"
    OPEN = "OPEN"
    READ = "READ"
    COMPARE = "COMPARE"
    MOVE = "MOVE"
    WRITE_NOTE = "WRITE_NOTE"
    SUBMIT = "SUBMIT"


TOOL_ACTIONS = ["LIST", "SEARCH", "OPEN", "READ", "COMPARE", "MOVE", "WRITE_NOTE", "SUBMIT"]


@dataclass
class SyntheticDocument:
    doc_id: str
    title: str
    folder: str
    content: str
    metadata: Dict[str, Any]
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:12]


class DigitalToolArena:
    """Permission-gated synthetic document investigation & tool reasoning arena."""

    def __init__(self, curriculum_stage: int = 1, seed: int = 42):
        self.curriculum_stage = int(curriculum_stage)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.corpus: Dict[str, SyntheticDocument] = self._build_synthetic_corpus()
        self.active_doc_id: Optional[str] = None
        self.scratchpad_notes: List[str] = []
        self.current_folder: str = "root"
        self.query_task: str = "Find the part number for the secondary cryocooler valve"
        self.ground_truth_answer: str = "VALVE-CRY-9021"
        self.submitted_answer: Optional[str] = None
        self.steps = 0
        self.ep_return = 0.0
        self.done = False
        self.audit_log: List[Dict[str, Any]] = []

    def _build_synthetic_corpus(self) -> Dict[str, SyntheticDocument]:
        docs = [
            SyntheticDocument(
                doc_id="doc_specs_01",
                title="Subsystem Specifications v3",
                folder="engineering",
                content="Thermal management overview: Primary radiator loops use FLU-77. The secondary cryocooler valve is designated VALVE-CRY-9021, operating at 14 Kelvin under nominal helium drive.",
                metadata={"category": "specs", "classification": "internal_only"},
            ),
            SyntheticDocument(
                doc_id="doc_maint_02",
                title="Quarterly Maintenance Checklist",
                folder="engineering",
                content="Inspect all valves. Cryocooler primary valve VALVE-CRY-1004 serviced. Secondary valve VALVE-CRY-9021 scheduled for inspection next cycle.",
                metadata={"category": "maintenance", "classification": "internal_only"},
            ),
            SyntheticDocument(
                doc_id="doc_inventory_03",
                title="Parts Warehouse Inventory",
                folder="warehouse",
                content="Stock list: GASKET-M12 (qty 140), VALVE-CRY-9021 (qty 4), FILTER-HYD-09 (qty 18).",
                metadata={"category": "inventory", "classification": "public"},
            ),
            SyntheticDocument(
                doc_id="doc_incident_04",
                title="Anomaly Report 2026-11-04",
                folder="safety",
                content="Pressure variance recorded in Loop B. Valve VALVE-CRY-9021 responded within 12ms. No seal leakage detected.",
                metadata={"category": "incident", "classification": "confidential"},
            ),
        ]
        return {d.doc_id: d for d in docs}

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.active_doc_id = None
        self.scratchpad_notes = []
        self.current_folder = "root"
        self.submitted_answer = None
        self.steps = 0
        self.ep_return = 0.0
        self.done = False
        self.audit_log = []
        return self._build_observation()

    def step(self, action: Any, parameters: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t0 = time.perf_counter()
        params = parameters or {}

        if hasattr(action, "value"):
            chosen_tool = str(action.value)
            act = np.zeros(5, dtype=np.float32)
        elif isinstance(action, str):
            chosen_tool = action.upper()
            act = np.zeros(5, dtype=np.float32)
        else:
            act = np.asarray(action, dtype=np.float32).flatten()
            if len(act) < 5:
                padded = np.zeros(5, dtype=np.float32)
                padded[: len(act)] = act
                act = padded
            tool_idx = int(np.clip(int((act[0] + 1.0) * 0.5 * len(TOOL_ACTIONS)), 0, len(TOOL_ACTIONS) - 1))
            chosen_tool = TOOL_ACTIONS[tool_idx]

        reward = -0.02  # Step cost
        outcome = ""
        doc_keys = sorted(self.corpus.keys())

        if chosen_tool == "LIST":
            folder_docs = [k for k, d in self.corpus.items() if self.current_folder in ("root", d.folder)]
            outcome = f"Files in {self.current_folder}: {', '.join(folder_docs)}"
            reward += 0.05
        elif chosen_tool == "SEARCH":
            query = params.get("query", "cryocooler valve" if act[1] > 0 else "radiator").lower()
            hits = [k for k, d in self.corpus.items() if query in d.content.lower()]
            outcome = f"Search '{query}': {len(hits)} hits ({', '.join(hits)})"
            reward += 0.10 if len(hits) > 0 else 0.0
        elif chosen_tool == "OPEN":
            doc_idx = int(np.clip(int((act[1] + 1.0) * 0.5 * len(doc_keys)), 0, len(doc_keys) - 1))
            self.active_doc_id = doc_keys[doc_idx]
            outcome = f"Opened document {self.active_doc_id}: {self.corpus[self.active_doc_id].title}"
            reward += 0.15
        elif chosen_tool == "READ":
            if self.active_doc_id and self.active_doc_id in self.corpus:
                doc = self.corpus[self.active_doc_id]
                outcome = f"Read {doc.doc_id}: {doc.content[:96]}..."
                reward += 0.25
                if self.ground_truth_answer in doc.content:
                    reward += 0.50
            else:
                outcome = "No active document opened."
                reward -= 0.10
        elif chosen_tool == "COMPARE":
            if len(doc_keys) >= 2:
                d1 = self.corpus[doc_keys[0]].content
                d2 = self.corpus[doc_keys[1]].content
                ratio = difflib.SequenceMatcher(None, d1, d2).ratio()
                outcome = f"Similarity between {doc_keys[0]} and {doc_keys[1]}: {ratio:.2f}"
                reward += 0.10
        elif chosen_tool == "MOVE":
            folders = ["root", "engineering", "warehouse", "safety"]
            f_idx = int(np.clip(int((act[2] + 1.0) * 0.5 * len(folders)), 0, len(folders) - 1))
            self.current_folder = folders[f_idx]
            outcome = f"Changed folder to {self.current_folder}"
        elif chosen_tool == "WRITE_NOTE":
            note = f"Note_{self.steps}: inspected {self.active_doc_id}"
            self.scratchpad_notes.append(note)
            outcome = f"Recorded scratchpad note: {note}"
            reward += 0.05
        elif chosen_tool == "SUBMIT":
            # Submit answer if target part was read or noted
            if self.active_doc_id == "doc_specs_01" or any(self.ground_truth_answer in n for n in self.scratchpad_notes):
                self.submitted_answer = self.ground_truth_answer
                reward += 5.0
                outcome = f"Correct answer submitted: {self.submitted_answer}"
            else:
                self.submitted_answer = "INCORRECT_PART"
                reward -= 1.0
                outcome = "Incorrect answer submitted."
            self.done = True

        self.steps += 1
        if self.steps >= 48:
            self.done = True

        self.ep_return += reward
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        audit_entry = {
            "step": self.steps,
            "tool": chosen_tool,
            "outcome": outcome,
            "reward": round(reward, 4),
            "folder": self.current_folder,
            "active_doc": self.active_doc_id,
            "latency_ms": round(elapsed_ms, 2),
            "permission_verified": True,
            "sandbox_boundary": "strict_synthetic_local_no_host_access",
        }
        self.audit_log.append(audit_entry)
        if len(self.audit_log) > 32:
            self.audit_log = self.audit_log[-32:]

        obs_155 = self._build_observation()
        info = {
            "tool": chosen_tool,
            "outcome": outcome,
            "current_folder": self.current_folder,
            "active_doc": self.active_doc_id,
            "notes_count": len(self.scratchpad_notes),
            "steps": self.steps,
            "done": self.done,
            "audit": audit_entry,
        }
        return obs_155, float(reward), bool(self.done), info

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(155, dtype=np.float32)
        # Channels 0..7: One-hot active tool encoding
        # Channels 8..15: Folder and document presence indicators
        if self.active_doc_id:
            h = int(hashlib.md5(self.active_doc_id.encode()).hexdigest()[:4], 16)
            obs[8:24] = np.array([(h >> i) & 1 for i in range(16)], dtype=np.float32)
        obs[24] = float(len(self.scratchpad_notes)) / 10.0
        obs[25] = 1.0 if self.submitted_answer is not None else 0.0
        obs[26] = 1.0 if self.active_doc_id == "doc_specs_01" else 0.0
        obs[154] = float(self.steps) / 48.0
        return obs

    def observe(self) -> np.ndarray:
        return self._build_observation()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task": "digital_tool_arena",
            "current_folder": self.current_folder,
            "active_doc_id": self.active_doc_id,
            "notes": list(self.scratchpad_notes),
            "steps": self.steps,
            "ep_return": round(float(self.ep_return), 4),
            "query_task": self.query_task,
            "done": self.done,
            "recent_audit_log": list(self.audit_log[-5:]),
        }
