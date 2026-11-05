"""Download pinned, bounded public artifacts and construct auditable chat splits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DATA_ID = "HuggingFaceTB/smol-smoltalk"
DATA_REV = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md"]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {path.name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while block := response.read(1048576):
            out.write(block)
    os.replace(tmp, path)

def prepare(root=".", rows=25000):
    root = Path(root)
    model_dir = root / "models/language"
    for name in MODEL_FILES:
        download(f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REV}/{name}", model_dir / name)
    model_manifest = {"id": MODEL_ID, "revision": MODEL_REV, "license": "Apache-2.0", "files": {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (model_dir / "manifest.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    raw = root / "data/language/source.parquet"
    download(f"https://huggingface.co/datasets/{DATA_ID}/resolve/{DATA_REV}/data/train-00000-of-00004.parquet", raw)
    download(f"https://huggingface.co/datasets/{DATA_ID}/resolve/{DATA_REV}/README.md", raw.parent / "SOURCE_CARD.md")
    import pyarrow.parquet as pq
    counts = {"train": 0, "dev": 0, "test": 0}
    seen = set()
    handles = {k: (raw.parent / f"{k}.jsonl").open("w", encoding="utf-8") for k in counts}
    try:
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=512):
            for item in batch.to_pylist():
                messages = item.get("messages", [])
                if not messages or any(m.get("role") not in ("user", "assistant", "system") or not isinstance(m.get("content"), str) for m in messages):
                    continue
                users = [m["content"] for m in messages if m["role"] == "user"]
                if not users or not any(m["role"] == "assistant" for m in messages):
                    continue
                # Keep all turns together and group by initial user intent to avoid exact-prompt leakage.
                identity = hashlib.sha256(" ".join(users[0].lower().split()).encode()).hexdigest()
                length = sum(len(m["content"]) for m in messages)
                if identity in seen or length > 8000 or length < 40:
                    continue
                seen.add(identity)
                bucket = int(identity[:8], 16) % 100
                split = "test" if bucket < 5 else "dev" if bucket < 10 else "train"
                record = {"id": identity, "messages": messages, "source": item.get("source", DATA_ID), "source_revision": DATA_REV}
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                if sum(counts.values()) >= rows:
                    break
            if sum(counts.values()) >= rows:
                break
    finally:
        for f in handles.values():
            f.close()
    manifest = {"dataset": DATA_ID, "revision": DATA_REV, "license": "Apache-2.0 per source card; retain source attribution", "counts": counts, "source_sha256": sha256(raw), "split_method": "sha256(normalized initial user prompt) modulo 100; 90/5/5", "files": {k: sha256(raw.parent / f"{k}.jsonl") for k in counts}, "limits": "Public synthetic conversation subset; exact prompt dedup is not semantic dedup; upstream pretraining may contain these tasks."}
    (raw.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--rows", type=int, default=25000)
    args = p.parse_args()
    if args.rows < 1:
        p.error("rows must be positive")
    prepare(args.root, args.rows)
