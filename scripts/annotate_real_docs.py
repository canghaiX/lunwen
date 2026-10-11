#!/usr/bin/env python3
"""Create Qwen3-Next initial annotations for natural VitaminC records."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from revimpact.evaluation.metrics import load_jsonl
from revimpact.llm.annotation import initial_messages, openai_chat, parse_initial


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in load_jsonl(path) if row.get("id")}


def _transformers_runner(model_path: str, max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def run_batch(messages_batch):
        texts = [tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                  tokenize=False, enable_thinking=False) for messages in messages_batch]
        inputs = tokenizer(texts, padding=True, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(messages):
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", enable_thinking=False,
            )
        except (TypeError, ValueError):
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )
        device = next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device
        inputs = inputs.to(device) if hasattr(inputs, "to") else {key: value.to(device) for key, value in inputs.items()}
        if hasattr(inputs, "items"):
            model_inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}}
        else:
            model_inputs = {"input_ids": inputs}
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = model_inputs["input_ids"].ne(tokenizer.pad_token_id or 0).long()
        with torch.inference_mode():
            generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_ids = model_inputs["input_ids"]
        return tokenizer.decode(generated[0][input_ids.shape[-1]:], skip_special_tokens=True)

    run.batch = run_batch
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--output", default="data/revimpact_800/annotations_qwen80b.jsonl")
    parser.add_argument("--model", default="/models/Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--api-url", default="http://127.0.0.1:30000")
    parser.add_argument("--api-model", default="Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    rows = [row for split in ("train", "dev", "test") for row in load_jsonl(Path(args.data) / f"{split}.jsonl")
            if row.get("metadata", {}).get("source_type") == "natural_vitaminc"]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.output)
    existing = {} if args.overwrite else _load_existing(output)
    if args.backend == "transformers":
        run = _transformers_runner(args.model, args.max_new_tokens)
    else:
        run = lambda messages: openai_chat(args.api_url, messages, args.api_model, max_tokens=args.max_new_tokens)

    output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(existing)
    rows = [row for row in rows if args.overwrite or row["id"] not in results
            or (args.retry_failed and results[row["id"]].get("parse_failed"))]
    prefetched = {}
    for index, row in enumerate(rows):
        if args.batch_size > 1 and hasattr(run, "batch") and index % args.batch_size == 0:
            group = rows[index:index + args.batch_size]
            texts = run.batch([initial_messages(item) for item in group])
            prefetched = {item["id"]: text for item, text in zip(group, texts)}
        raw_output = ""
        parse_failed = False
        error = None
        for attempt in range(args.retry + 1):
            try:
                raw_output = prefetched.pop(row["id"]) if row["id"] in prefetched else run(initial_messages(row))
                prediction, parse_failed = parse_initial(raw_output)
                if not parse_failed or attempt == args.retry:
                    break
            except Exception as exc:  # preserve failures for resumable auditing
                error = f"{type(exc).__name__}: {exc!r}"
                if attempt == args.retry:
                    prediction = {
                        "old_label": "unknown", "new_label": "unknown", "should_change": False,
                        "old_evidence_ids": [], "new_evidence_ids": [], "impact_condition": {},
                        "action": "human_review", "confidence": 0.0, "uncertainty": ["inference_failed"],
                    }
                    parse_failed = True
        results[row["id"]] = {
            "id": row["id"], "annotator": "qwen3-next-80b", "annotation_version": "initial",
            "prediction": prediction, "raw_output": raw_output, "parse_failed": parse_failed,
            "error": error, "timestamp": _now(),
        }
        with output.open("w", encoding="utf-8") as handle:
            for key in sorted(results):
                handle.write(json.dumps(results[key], ensure_ascii=False) + "\n")
        print(json.dumps({"id": row["id"], "parse_failed": parse_failed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
