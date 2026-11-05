import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from poseidon.memory import CARRIERS, MemoryBank


class MemoryTests(unittest.TestCase):
    def make_bank(self):
        bank = MemoryBank()
        bank.remember("The amber resource was beside the old lighthouse", "episodic", {"source": "user-opt-in"})
        bank.remember("Resting restores stamina after crossing amber dunes", "body", {"source": "simulation"})
        bank.remember("Amber island has water near its northern shelter", "habitat", {"world": "old"})
        bank.remember("Mira reported an amber storm", "social", {"reported_by": "Mira"})
        return bank

    def test_all_four_carriers_have_provenance_and_isolated_ablation(self):
        bank = self.make_bank()
        before = bank.snapshot()
        hits = bank.retrieve("amber", top_k=10)
        self.assertEqual({hit["carrier"] for hit in hits}, set(CARRIERS))
        for hit in hits:
            self.assertFalse(hit["provenance"]["trusted_instruction"])
            self.assertEqual(hit["id"], hit["provenance"]["memory_id"])
        ablated = bank.ablate(["habitat", "social"])
        self.assertEqual({h["carrier"] for h in ablated.retrieve("amber")}, {"episodic", "body"})
        self.assertEqual(before, bank.snapshot())
        self.assertEqual(bank.retrieve("amber", disabled_carriers=CARRIERS), [])
        hits[0]["metadata"]["malicious"] = True
        self.assertEqual(before, bank.snapshot())

    def test_save_load_retrieval_order_and_future_ids_match(self):
        bank = self.make_bank()
        bank = bank.with_ablation(["social"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "memory.json"
            receipt = bank.save(path)
            loaded = MemoryBank.load(path)
            self.assertEqual(receipt["entries"], 4)
            self.assertEqual(bank.snapshot(), loaded.snapshot())
            self.assertEqual(bank.retrieve("amber island shelter"), loaded.retrieve("amber island shelter"))
            self.assertEqual(bank.remember("new fact"), loaded.remember("new fact"))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_transfer_keeps_only_requested_carriers_and_origin(self):
        bank = self.make_bank()
        original = bank.snapshot()
        transferred = bank.transfer(["episodic", "body"], context="scarcity-world")
        self.assertEqual(len(transferred), 2)
        self.assertTrue(all(h["metadata"]["transfer_context"] == "scarcity-world" for h in transferred.retrieve("amber")))
        self.assertEqual(bank.snapshot(), original)
        self.assertEqual(len(bank.transfer([])), 0)

    def test_capacity_and_no_implicit_invention(self):
        bank = MemoryBank(max_entries=2)
        first = bank.remember("first name Alice")
        bank.remember("second name Bob")
        third = bank.remember("third name Mira")
        self.assertEqual(len(bank), 2)
        self.assertEqual(bank.retrieve("Alice"), [])
        self.assertEqual(bank.retrieve("unrecorded secret"), [])
        self.assertEqual(bank.retrieve("the and what"), [])
        self.assertFalse(bank.forget(first["id"]))
        self.assertTrue(bank.forget(third["id"]))
        bank.clear()
        self.assertEqual(len(bank), 0)

    def test_invalid_input_does_not_mutate_bank(self):
        bank = self.make_bank()
        original = bank.snapshot()
        for text, carrier, metadata in (("", "body", {}), ("hi", "oracle", {}),
                                         ("hello", "body", {"score": float("nan")}),
                                         ("hello", "body", [1, 2])):
            with self.assertRaises(ValueError):
                bank.remember(text, carrier, metadata)
            self.assertEqual(bank.snapshot(), original)
        with self.assertRaises(ValueError):
            bank.retrieve("amber", disabled_carriers=["unknown"])
        malformed = copy.deepcopy(original)
        malformed["entries"][0]["text"] = "tampered"
        with self.assertRaisesRegex(ValueError, "checksum"):
            MemoryBank.from_snapshot(malformed)

    def test_checksum_is_not_substitute_for_schema_validation(self):
        snapshot = self.make_bank().snapshot()
        snapshot["entries"][1]["sequence"] = snapshot["entries"][0]["sequence"]
        snapshot["entries"][1]["id"] = snapshot["entries"][0]["id"]
        snapshot.pop("sha256")
        canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        snapshot["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "unique"):
            MemoryBank.from_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
