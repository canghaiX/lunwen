#!/usr/bin/env python3
"""Audit initial annotations with a local Transformers model or API."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from revimpact.evaluation.metrics import load_jsonl
from revimpact.llm.annotation import openai_chat, parse_review, review_messages


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transformers_runner(model_path: str, max_new_tokens: int):
    """Load a locally compatible causal LM and return a chat inference callable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def run(messages):
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", enable_thinking=False,
            )
        except (TypeError, ValueError):
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt",
            )
        device = next(
            parameter for parameter in model.parameters()
            if parameter.device.type != "meta"
        ).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        if hasattr(inputs, "items"):
            model_inputs = {
                key: value for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
        else:
            model_inputs = {"input_ids": inputs}
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = model_inputs["input_ids"].ne(
                tokenizer.pad_token_id or 0
            ).long()
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_ids = model_inputs["input_ids"]
        return tokenizer.decode(
            generated[0][input_ids.shape[-1]:], skip_special_tokens=True,
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--initial", default="data/revimpact_800/annotations_qwen80b.jsonl")
    parser.add_argument("--output", default="data/revimpact_800/annotations_qwen35b_review.jsonl")
    parser.add_argument("--backend", choices=("transformers", "openai"), default="openai")
    parser.add_argument("--model", default="/models/Qwen/Qwen3-4B")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--reviewer", default="qwen3.6-35b")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = {row["id"]: row for split in ("train", "dev", "test") for row in load_jsonl(Path(args.data) / f"{split}.jsonl")
            if row.get("metadata", {}).get("source_type") == "natural_vitaminc"}
    initial = {row["id"]: row for row in load_jsonl(args.initial)}
    ids = sorted(set(rows) & set(initial))
    if args.limit:
        ids = ids[:args.limit]
    output = Path(args.output)
    existing = {} if args.overwrite or not output.exists() else {row["id"]: row for row in load_jsonl(output)}
    results = dict(existing)
    if args.backend == "transformers":
        run = _transformers_runner(args.model, max_new_tokens=768)
    else:
        run = lambda messages: openai_chat(
            args.api_url, messages, args.api_model, max_tokens=768,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    for row_id in ids:
        if row_id in results and not args.overwrite:
            continue
        row, proposal = rows[row_id], initial[row_id].get("prediction", {})
        raw_output, parse_failed, error = "", False, None
        for attempt in range(args.retry + 1):
            try:
                raw_output = run(review_messages(row, proposal))
                review, parse_failed = parse_review(raw_output)
                if not parse_failed or attempt == args.retry:
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt == args.retry:
                    review = {"overall_decision": "human_review", "confidence": 0.0, "reason_codes": ["inference_failed"]}
                    parse_failed = True
        results[row_id] = {
            "id": row_id, "reviewer": args.reviewer, "annotation_version": "review",
            "initial_annotation_id": row_id, "review": review, "raw_output": raw_output,
            "parse_failed": parse_failed, "error": error, "timestamp": _now(),
        }
        with output.open("w", encoding="utf-8") as handle:
            for key in sorted(results):
                handle.write(json.dumps(results[key], ensure_ascii=False) + "\n")
        print(json.dumps({"id": row_id, "parse_failed": parse_failed,
                          "overall_decision": review.get("overall_decision")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
