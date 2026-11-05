"""CPU-bounded LoRA experiment; saves candidates, never silently promotes them."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

def encode_record(tokenizer, item, max_length=160):
    messages = item["messages"]
    user = next((m["content"] for m in messages if m["role"] == "user"), None)
    answer = None
    found_user = False
    for m in messages:
        if m["role"] == "user":
            found_user = True
        elif found_user and m["role"] == "assistant":
            answer = m["content"]
            break
    if user is None or answer is None:
        return None
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True, return_dict=False)
    # Reject long prompts, rather than train on an answer disconnected from its question.
    if len(prefix) > max_length - 32:
        return None
    suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (prefix + suffix)[:max_length]
    return {"input_ids": ids, "labels": [-100]*len(prefix) + ids[len(prefix):], "id": item["id"]}

def _read(path, tokenizer, limit, max_length):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            record = encode_record(tokenizer, json.loads(line), max_length)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No usable records in {path}")
    return records

def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
    data_manifest = json.loads(Path(args.data,"manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        actual = hashlib.sha256(Path(args.data,f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != data_manifest["files"][split]:
            raise ValueError(f"{split} data changed since the preparation manifest; rebuild it or use the original files.")
    records = _read(Path(args.data)/"train.jsonl", tokenizer, args.examples, args.max_length)
    dev = _read(Path(args.data)/"dev.jsonl", tokenizer, 24, args.max_length)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], layers_to_transform=list(range(model.config.num_hidden_layers-4, model.config.num_hidden_layers)), task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    def batch(items):
        n = max(len(x["input_ids"]) for x in items)
        ids = [x["input_ids"] + [tokenizer.eos_token_id]*(n-len(x["input_ids"])) for x in items]
        labels = [x["labels"] + [-100]*(n-len(x["labels"])) for x in items]
        mask = [[1]*len(x["input_ids"]) + [0]*(n-len(x["input_ids"])) for x in items]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}
    def evaluate():
        model.eval()
        weighted, tokens = 0.0, 0
        with torch.inference_mode():
            for start in range(0, len(dev), args.batch_size):
                packed = batch(dev[start:start+args.batch_size])
                count = int((packed["labels"][:,1:] != -100).sum())
                weighted += float(model(**packed).loss)*count
                tokens += count
        return weighted/max(tokens,1)
    fingerprint = hashlib.sha256(Path(args.data,"manifest.json").read_bytes()).hexdigest()
    run_config = {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "max_length": args.max_length, "ids": [r["id"] for r in records], "data_hash": fingerprint, "model_manifest_hash": hashlib.sha256(Path(args.model,"manifest.json").read_bytes()).hexdigest()}
    start_step, tokens_seen, examples_seen = 0, 0, 0
    baseline = None
    order_gen = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(records), generator=order_gen).tolist()
    state_path = out/"training_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["run_config"] != run_config:
            raise ValueError("Resume configuration or corpus changed; use a new run directory.")
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        start_step, tokens_seen, examples_seen, baseline = state["step"], state["tokens_seen"], state["examples_seen"], state["baseline_dev_loss"]
    elif state_path.exists():
        raise ValueError("Existing run requires --resume or a different --output.")
    if baseline is None:
        baseline = evaluate()
    print(json.dumps({"event": "start", "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "usable_records": len(records), "baseline_dev_loss": baseline}), flush=True)
    start_time = time.perf_counter()
    log = (out/"metrics.jsonl").open("a", encoding="utf-8")
    def save(step, status, final_loss=None):
        from peft import get_peft_model_state_dict
        adapter = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        tmp = out/"training_state.tmp"
        torch.save({"adapter": adapter, "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "step": step, "tokens_seen": tokens_seen, "examples_seen": examples_seen, "run_config": run_config, "baseline_dev_loss": baseline}, tmp)
        os.replace(tmp, state_path)
        model.save_pretrained(out/"adapter", safe_serialization=True)
        report = {"status": status, "steps": step, "examples_seen": examples_seen, "unique_usable_records": len(records), "tokens_seen": tokens_seen, "baseline_dev_loss": baseline, "candidate_dev_loss": final_loss, "active_by_default": False, "reason": "Candidate only: dev loss is not proof of better conversation or reasoning.", "seconds_this_session": time.perf_counter()-start_time, "data_hash": fingerprint}
        temp_report = out/"report.tmp"
        temp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(temp_report, out/"report.json")
    step = start_step
    if not state_path.exists():
        save(start_step, "initialized")
    try:
        for step in range(start_step+1, args.steps+1):
            model.train()
            indexes = [order[((step-1)*args.batch_size+j) % len(order)] for j in range(args.batch_size)]
            packed = batch([records[i] for i in indexes])
            optimizer.zero_grad(set_to_none=True)
            loss = model(**packed).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            examples_seen += len(indexes)
            tokens_seen += int((packed["labels"][:,1:] != -100).sum())
            if step % 10 == 0 or step == 1:
                event = {"step": step, "loss": float(loss.detach()), "examples_seen": examples_seen, "tokens_seen": tokens_seen, "elapsed_seconds": time.perf_counter()-start_time}
                log.write(json.dumps(event)+"\n"); log.flush()
                print(json.dumps(event), flush=True)
            if step % 50 == 0:
                save(step, "running")
        candidate = evaluate()
        save(step, "completed", candidate)
        print((out/"report.json").read_text(), flush=True)
    except KeyboardInterrupt:
        # A signal can interrupt an optimizer update; preserve the last durable state.
        print("Interrupted. The previous atomic checkpoint is retained; resume from it.", flush=True)
        raise
    finally:
        log.close()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="models/language")
    p.add_argument("--data", default="data/language")
    p.add_argument("--output", default="runs/language")
    p.add_argument("--examples", type=int, default=2048)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--resume", action="store_true")
    args=p.parse_args()
    if min(args.examples,args.steps,args.batch_size,args.threads) < 1 or args.max_length < 48:
        p.error("Counts must be positive; max length must be at least 48.")
    train(args)
