"""Tessera: Independent-Lineage Ratified Macro-Action Commons.

Inspired by Tessera Lab (independent-lineage ratification, finite-domain semantic checking,
and expiring instruction libraries).
Macro-action candidate sequences discovered in an episode are isolated in quarantine.
A candidate is promoted to the public ratified instruction set ONLY when independently
discovered and validated by at least two distinct lineages (seeds) with positive net return deltas.
Unreinforced opcodes expire after an epoch horizon.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Tuple, Optional, List, Dict


class TesseraMacroCommons:
    """Library of independent-lineage ratified macro-action sequences."""

    EXPIRATION_EPOCHS = 3

    def __init__(self, current_epoch: int = 1) -> None:
        self.current_epoch = current_epoch
        # Candidates in quarantine: sequence_key -> list of {lineage_id, delta_r, epoch}
        self.quarantine: Dict[str, List[Dict[str, Any]]] = {}
        # Ratified public opcodes: opcode_id -> {sequence, ratifiers, delta_r_mean, usage_count, last_epoch}
        self.ratified: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear episode-learned candidates so paired runs start from the same state."""
        self.current_epoch = 1
        self.quarantine.clear()
        self.ratified.clear()

    @staticmethod
    def seq_key(actions: List[int]) -> str:
        return "-".join(str(a) for a in actions)

    def propose_candidate(self, lineage_id: str, actions: List[int], delta_r: float) -> Optional[str]:
        """Propose a macro-action candidate from an independent episode lineage.
        
        Returns opcode_id if newly ratified, else None.
        """
        if not actions or len(actions) < 2 or len(actions) > 6 or delta_r <= 0.0:
            return None
            
        key = self.seq_key(actions)
        
        # Check if already ratified
        for opcode_id, data in self.ratified.items():
            if data["sequence_key"] == key:
                if lineage_id not in data["ratifiers"]:
                    data["ratifiers"].append(lineage_id)
                data["usage_count"] += 1
                data["last_epoch"] = self.current_epoch
                return opcode_id

        # Register proposal in quarantine
        if key not in self.quarantine:
            self.quarantine[key] = []
            
        # Avoid duplicate claims from same lineage
        existing_lineages = {entry["lineage_id"] for entry in self.quarantine[key]}
        if lineage_id not in existing_lineages:
            self.quarantine[key].append({
                "lineage_id": lineage_id,
                "delta_r": float(delta_r),
                "epoch": self.current_epoch,
            })

        # Ratification rule: >= 2 distinct lineages with delta_r > 0
        distinct_lineages = {entry["lineage_id"] for entry in self.quarantine[key] if entry["delta_r"] > 0}
        if len(distinct_lineages) >= 2:
            opcode_id = f"OP_{hashlib.sha256(key.encode()).hexdigest()[:8].upper()}"
            deltas = [entry["delta_r"] for entry in self.quarantine[key]]
            mean_delta = sum(deltas) / len(deltas)
            self.ratified[opcode_id] = {
                "opcode_id": opcode_id,
                "actions": list(actions),
                "sequence_key": key,
                "ratifiers": sorted(list(distinct_lineages)),
                "delta_r_mean": round(float(mean_delta), 4),
                "usage_count": 1,
                "created_epoch": self.current_epoch,
                "last_epoch": self.current_epoch,
            }
            # Remove from quarantine once ratified
            del self.quarantine[key]
            return opcode_id
            
        return None

    def advance_epoch(self) -> List[str]:
        """Age out unreinforced opcodes that exceed the expiration epoch horizon."""
        self.current_epoch += 1
        expired = []
        for opcode_id, data in list(self.ratified.items()):
            if self.current_epoch - data["last_epoch"] > self.EXPIRATION_EPOCHS:
                expired.append(opcode_id)
                del self.ratified[opcode_id]
        return expired

    def retire_expired(self) -> List[str]:
        """Retire expired opcodes upon cyclic beacon phase advance."""
        return self.advance_epoch()

    def validate_safety(self, actions: List[int], current_stamina: float, current_depth: float) -> bool:
        """Finite-domain semantic check: verify stamina budget across the sequence."""
        # Action stamina costs: 0 (rest) recovers +0.10, movements 1..4 cost 0.05, forage 5 costs 0.08
        stamina = current_stamina
        depth = current_depth
        for a in actions:
            if a == 0:
                stamina = min(1.0, stamina + 0.10)
            elif a in (1, 2, 3, 4):
                stamina -= 0.05
                if a == 2:  # Down increases depth
                    depth += 0.04
            elif a == 5:
                stamina -= 0.08
                
            if stamina <= 0.05 or depth > 0.90:
                return False  # Danger: unsafe macro-action in current physical context
        return True

    def export_commons(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "ratified_count": len(self.ratified),
            "quarantine_count": len(self.quarantine),
            "ratified_opcodes": self.ratified,
            "quarantined_candidates": {
                k: [{"lineage": e["lineage_id"], "delta_r": round(e["delta_r"], 4)} for e in v]
                for k, v in self.quarantine.items()
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export_commons(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TesseraMacroCommons":
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = cls(current_epoch=payload.get("current_epoch", 1))
        instance.ratified = payload.get("ratified_opcodes", {})
        return instance
