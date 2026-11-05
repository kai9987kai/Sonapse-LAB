"""Bounded, explicit four-carrier retrieval memory with inspectable provenance.

This is deterministic lexical retrieval, not a biological memory model or a
neural weight update. The runtime must call remember only on explicit opt-in.
Stored text is untrusted data; retrieval never executes it as an instruction.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections import Counter
from typing import Iterable

CARRIERS = ("episodic", "body", "habitat", "social")
MEMORY_SCHEMA = "poseidon-memory-v1"
_STOP = frozenset("a an and are as at be by can do for from how i in is it me my of on or please tell that the this to was what where which who with you your".split())


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
            if token not in _STOP]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value: object) -> object:
    try:
        serialized = _canonical(value)
        if len(serialized.encode("utf-8")) > 16384:
            raise ValueError("Memory metadata exceeds 16 KiB")
        return json.loads(serialized)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Memory metadata must be finite acyclic JSON data") from exc


def _carriers(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    result = set(values)
    if not result <= set(CARRIERS):
        raise ValueError(f"Unknown carriers: {sorted(result - set(CARRIERS))}")
    return result


class MemoryBank:
    """A local bank of opted-in facts/notes split by explicit software carrier.

    BM25 scores rank lexical matches and are not probabilities or truth scores.
    The oldest record is evicted when capacity is reached. Every API returns a
    copy so a retrieved note cannot mutate the saved bank accidentally.
    """

    def __init__(self, max_entries: int = 512):
        if type(max_entries) is not int or not 1 <= max_entries <= 4096:
            raise ValueError("max_entries must be an integer from 1 to 4096")
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self._next_id = 1
        self.disabled_carriers: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[dict]:
        return copy.deepcopy(self._entries)

    def remember(self, text: str, carrier: str = "episodic", metadata: dict | None = None) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError("Memory text must contain 1 to 12000 characters")
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        validated_metadata = _json_copy(metadata or {})
        sequence = self._next_id
        record = {"id": f"memory-{sequence:08d}", "sequence": sequence,
                  "text": text.strip(), "carrier": carrier, "metadata": validated_metadata}
        self._entries.append(record)
        self._entries = self._entries[-self.max_entries:]
        self._next_id += 1
        return copy.deepcopy(record)

    def retrieve(self, query: str, disabled_carriers: Iterable[str] | None = None,
                 top_k: int = 5, mode: str = "bm25") -> list[dict]:
        if not isinstance(query, str) or len(query) > 20000:
            raise ValueError("query must be text up to 20000 characters")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        if mode not in ("bm25", "hybrid", "dense"):
            raise ValueError("mode must be 'bm25', 'hybrid', or 'dense'")
        disabled = self.disabled_carriers | _carriers(disabled_carriers)
        terms = set(_tokens(query))
        if not terms and mode == "bm25":
            return []
        records = [entry for entry in self._entries if entry["carrier"] not in disabled]
        if not records:
            return []

        # 1. Lexical BM25 computation
        documents = [_tokens(entry["text"]) for entry in records]
        counts = [Counter(tokens) for tokens in documents]
        mean_length = max(1.0, sum(map(len, documents)) / len(documents))
        frequency = {term: sum(term in counts_ for counts_ in counts) for term in terms}

        bm25_hits = {}
        for entry, tokens, term_counts in zip(records, documents, counts):
            score = 0.0
            matched = []
            for term in sorted(terms):
                occurrence = term_counts.get(term, 0)
                if occurrence:
                    matched.append(term)
                    idf = math.log(1 + (len(records) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                    denominator = occurrence + 1.2 * (0.25 + 0.75 * len(tokens) / mean_length)
                    score += idf * occurrence * 2.2 / denominator
            if score > 0:
                bm25_hits[entry["id"]] = {"score": score, "matched": matched}

        if mode == "bm25":
            hits = []
            for entry in records:
                if entry["id"] in bm25_hits:
                    result = copy.deepcopy(entry)
                    result.update({"score": bm25_hits[entry["id"]]["score"],
                                   "matched_terms": bm25_hits[entry["id"]]["matched"],
                                   "provenance": {"memory_id": entry["id"], "carrier": entry["carrier"],
                                                  "sequence": entry["sequence"], "metadata": copy.deepcopy(entry["metadata"]),
                                                  "retrieval": "bm25-lexical-v1", "trusted_instruction": False}})
                    hits.append(result)
            return sorted(hits, key=lambda hit: (-hit["score"], -hit["sequence"], hit["id"]))[:top_k]

        # 2. Dense semantic hashing features (blake2b lemma bigrams)
        from .core import hash_features
        import torch

        q_feat = hash_features([query])[0]
        doc_feats = hash_features([entry["text"] for entry in records])
        # Since hash_features is L2-normalized, cosine similarity is the inner product
        cosine_sims = (doc_feats @ q_feat).tolist()

        dense_hits = {}
        for entry, sim in zip(records, cosine_sims):
            if sim > 0.08:  # positive semantic correlation threshold
                dense_hits[entry["id"]] = float(sim)

        if mode == "dense":
            hits = []
            for entry in records:
                if entry["id"] in dense_hits:
                    result = copy.deepcopy(entry)
                    result.update({"score": dense_hits[entry["id"]],
                                   "matched_terms": bm25_hits.get(entry["id"], {}).get("matched", []),
                                   "provenance": {"memory_id": entry["id"], "carrier": entry["carrier"],
                                                  "sequence": entry["sequence"], "metadata": copy.deepcopy(entry["metadata"]),
                                                  "retrieval": "dense-hashing-v1", "trusted_instruction": False,
                                                  "dense_cosine": round(dense_hits[entry["id"]], 4)}})
                    hits.append(result)
            return sorted(hits, key=lambda hit: (-hit["score"], -hit["sequence"], hit["id"]))[:top_k]

        # 3. Hybrid Reciprocal Rank Fusion (RRF, Cormack et al. 2009, k=60)
        bm25_ranked = sorted(bm25_hits.keys(), key=lambda i: -bm25_hits[i]["score"])
        dense_ranked = sorted(dense_hits.keys(), key=lambda i: -dense_hits[i])
        bm25_rank_map = {mid: rank + 1 for rank, mid in enumerate(bm25_ranked)}
        dense_rank_map = {mid: rank + 1 for rank, mid in enumerate(dense_ranked)}

        rrf_k = 60.0
        candidate_ids = set(bm25_hits.keys()) | set(dense_hits.keys())
        hits = []
        for entry in records:
            eid = entry["id"]
            if eid in candidate_ids:
                rrf_score = 0.0
                if eid in bm25_rank_map:
                    rrf_score += 1.0 / (rrf_k + bm25_rank_map[eid])
                if eid in dense_rank_map:
                    rrf_score += 1.0 / (rrf_k + dense_rank_map[eid])

                result = copy.deepcopy(entry)
                result.update({
                    "score": round(rrf_score, 6),
                    "matched_terms": bm25_hits.get(eid, {}).get("matched", []),
                    "provenance": {
                        "memory_id": entry["id"], "carrier": entry["carrier"],
                        "sequence": entry["sequence"], "metadata": copy.deepcopy(entry["metadata"]),
                        "retrieval": "hybrid-rrf-v1", "trusted_instruction": False,
                        "bm25_score": round(bm25_hits[eid]["score"], 4) if eid in bm25_hits else None,
                        "dense_cosine": round(dense_hits[eid], 4) if eid in dense_hits else None,
                        "bm25_rank": bm25_rank_map.get(eid),
                        "dense_rank": dense_rank_map.get(eid),
                    }
                })
                hits.append(result)

        return sorted(hits, key=lambda hit: (-hit["score"], -hit["sequence"], hit["id"]))[:top_k]

    def with_ablation(self, disabled_carriers: Iterable[str]) -> "MemoryBank":
        """Return an independent retrieval view; original records stay intact."""
        clone = self.from_snapshot(self.snapshot())
        clone.disabled_carriers |= _carriers(disabled_carriers)
        return clone

    def ablate(self, disabled_carriers: Iterable[str]) -> "MemoryBank":
        return self.with_ablation(disabled_carriers)

    def transfer(self, preserve_carriers: Iterable[str] | None = None,
                 context: str | None = None) -> "MemoryBank":
        """Copy selected carriers into a new bank with optional transfer metadata."""
        retained = set(CARRIERS) if preserve_carriers is None else _carriers(preserve_carriers)
        if context is not None and (not isinstance(context, str) or len(context) > 1000):
            raise ValueError("context must be a string up to 1000 characters")
        result = self.from_snapshot(self.snapshot())
        result._entries = [entry for entry in result._entries if entry["carrier"] in retained]
        result.disabled_carriers &= retained
        if context is not None:
            for entry in result._entries:
                metadata = {**entry["metadata"], "transfer_context": context,
                            "origin_memory_id": entry["metadata"].get("origin_memory_id", entry["id"])}
                entry["metadata"] = _json_copy(metadata)
        return result

    def forget(self, memory_id: str) -> bool:
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry["id"] != memory_id]
        return len(self._entries) != before

    def clear(self) -> None:
        self._entries.clear()

    def snapshot(self) -> dict:
        payload = {"schema": MEMORY_SCHEMA, "max_entries": self.max_entries,
                   "next_id": self._next_id, "entries": self.entries,
                   "disabled_carriers": sorted(self.disabled_carriers)}
        payload["sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "MemoryBank":
        if not isinstance(snapshot, dict) or snapshot.get("schema") != MEMORY_SCHEMA:
            raise ValueError("Unsupported memory snapshot schema")
        if set(snapshot) != {"schema", "max_entries", "next_id", "entries", "disabled_carriers", "sha256"}:
            raise ValueError("Unexpected memory snapshot fields")
        payload = copy.deepcopy(snapshot)
        expected = payload.pop("sha256")
        if expected != hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest():
            raise ValueError("Memory snapshot checksum mismatch")
        result = cls(payload.get("max_entries"))
        records = payload.get("entries")
        next_id = payload.get("next_id")
        if type(next_id) is not int or not 1 <= next_id <= 2**53:
            raise ValueError("Invalid memory sequence counter")
        if not isinstance(records, list) or len(records) > result.max_entries:
            raise ValueError("Invalid memory entries")
        sequences = []
        for entry in records:
            if not isinstance(entry, dict) or set(entry) != {"id", "sequence", "text", "carrier", "metadata"}:
                raise ValueError("Invalid memory record")
            sequence = entry["sequence"]
            if type(sequence) is not int or not 1 <= sequence < next_id:
                raise ValueError("Invalid memory sequence")
            if entry["id"] != f"memory-{sequence:08d}":
                raise ValueError("Memory ID does not match sequence")
            if not isinstance(entry["text"], str) or not entry["text"].strip() or len(entry["text"]) > 12000:
                raise ValueError("Invalid memory text")
            if entry["carrier"] not in CARRIERS or not isinstance(entry["metadata"], dict):
                raise ValueError("Invalid memory carrier or metadata")
            _json_copy(entry["metadata"])
            sequences.append(sequence)
        if sequences != sorted(set(sequences)):
            raise ValueError("Memory records must have unique increasing sequence numbers")
        if not isinstance(payload["disabled_carriers"], list):
            raise ValueError("disabled_carriers must be a list")
        result.disabled_carriers = _carriers(payload["disabled_carriers"])
        result._entries = copy.deepcopy(records)
        result._next_id = next_id
        return result

    def save(self, path: str | Path) -> dict:
        """Atomic local JSON replacement; returns an inspectable save receipt."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(snapshot, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {"path": str(destination.resolve()), "entries": len(self), "sha256": snapshot["sha256"]}

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        source = Path(path)
        if not source.exists():
            return cls()
        if source.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Memory file exceeds 64 MiB limit")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Memory file is not valid UTF-8 JSON") from exc
        return cls.from_snapshot(data)
