#!/usr/bin/env python3
"""Evaluate RevImpact pilot baselines and optional sampled local Qwen predictions."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Callable

from revimpact.evaluation.baselines import condition_rule_prediction, evidence_overlap_prediction, text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl


def _majority_prediction(row: dict[str, Any], majority: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "pred_old_label": majority["old_label"],
        "pred_new_label": majority["new_label"],
        "pred_should_change": majority["should_change"],
        "pred_evidence_ids": [],
    }


def _majority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def most_common(key: str, default: str) -> str:
        values = [str(row.get(key, default)) for row in rows]
        return collections.Counter(values).most_common(1)[0][0] if values else default

    changes = [bool(row.get("should_change")) for row in rows if row.get("metadata", {}).get("scorable", True)]
    return {
        "old_label": most_common("old_label", "unknown"),
        "new_label": most_common("new_label", "unknown"),
        "should_change": collections.Counter(changes).most_common(1)[0][0] if changes else False,
    }


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _diagnostics(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    family_overlap = {}
    documents = {}
    query_document_overlap = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            family_overlap[f"{left}:{right}"] = len(
                {row.get("family_id") for row in splits[left]} & {row.get("family_id") for row in splits[right]}
            )
    for split, rows in splits.items():
        pairs = [(row.get("old_document", ""), row.get("new_document", "")) for row in rows]
        pair_families: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        old_families: dict[str, set[str]] = collections.defaultdict(set)
        new_families: dict[str, set[str]] = collections.defaultdict(set)
        for row, pair in zip(rows, pairs):
            family = str(row.get("family_id", ""))
            pair_families[pair].add(family)
            old_families[pair[0]].add(family)
            new_families[pair[1]].add(family)
        documents[split] = {
            "n": len(pairs),
            "within_family_repeated_pairs": len(pairs) - len(pair_families),
            "cross_family_duplicate_pairs": sum(len(families) > 1 for families in pair_families.values()),
            "cross_family_duplicate_old_documents": sum(len(families) > 1 for families in old_families.values()),
            "cross_family_duplicate_new_documents": sum(len(families) > 1 for families in new_families.values()),
        }
        overlap = 0
        for row in rows:
            query = row.get("query", "").lower().split()
            document = f"{row.get('old_document', '')} {row.get('new_document', '')}".lower()
            if query and all(token in document for token in query if len(token) > 2):
                overlap += 1
        query_document_overlap[split] = overlap / max(len(rows), 1)
    return {"family_overlap": family_overlap, "documents": documents, "query_document_exact_overlap": query_document_overlap}


def _group_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(row.get("metadata", {}).get(key, row.get(key, "unknown"))) for row in rows})
    output = {}
    for value in values:
        indices = [index for index, row in enumerate(rows) if str(row.get("metadata", {}).get(key, row.get(key, "unknown"))) == value]
        output[value] = evaluate_records([rows[index] for index in indices], [predictions[index] for index in indices])
    return output


def _sample_stratified(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row.get("metadata", {}).get("category", "unknown"))].append(row)
    rng = random.Random(seed)
    sampled = []
    for group in groups.values():
        rng.shuffle(group)
        quota = max(1, round(limit * len(group) / len(rows)))
        sampled.extend(group[:quota])
    rng.shuffle(sampled)
    if len(sampled) < limit:
        selected = {row["id"] for row in sampled}
        remainder = [row for row in rows if row["id"] not in selected]
        rng.shuffle(remainder)
        sampled.extend(remainder[: limit - len(sampled)])
    return sampled[:limit]


def _run_qwen(rows: list[dict[str, Any]], model_path: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from scripts.predict_vitaminc_llm import _parse
        except ModuleNotFoundError:
            from predict_vitaminc_llm import _parse
    except Exception as exc:
        return [], {"status": "unavailable", "error": repr(exc)}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model_kwargs = {"local_files_only": True, "trust_remote_code": True, "torch_dtype": "auto"}
        try:
            import accelerate  # noqa: F401
            model_kwargs["device_map"] = "auto"
        except ImportError:
            pass
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if "device_map" not in model_kwargs and torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as exc:
        return [], {"status": "load_failed", "error": repr(exc)}
    predictions = []
    parse_failures = 0
    device = next(model.parameters()).device
    for row in rows:
        prompt = (
            "Compare original and revised evidence. Return only JSON with keys "
            "old_label, new_label, should_change, evidence_ids, confidence. "
            "Labels must be entailed, contradicted, or unknown.\n"
            f"CLAIM: {row['query']}\nORIGINAL: {row['old_document']}\nREVISED: {row['new_document']}\n"
            f"CONDITION: {json.dumps(row.get('condition', {}), ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a fact verification classifier. Return only valid JSON. /no_think"},
            {"role": "user", "content": prompt},
        ]
        try:
            use_template = hasattr(tokenizer, "apply_chat_template")
            if use_template:
                try:
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False)
                    inputs = inputs.to(device) if hasattr(inputs, "to") else {key: value.to(device) for key, value in inputs.items()}
                except (ImportError, RuntimeError, ValueError):
                    use_template = False
            if not use_template:
                inputs = {key: value.to(device) for key, value in tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).items()}
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = output[0][inputs["input_ids"].shape[-1]:]
            parsed = _parse(tokenizer.decode(generated, skip_special_tokens=True))
        except Exception as exc:
            parsed = {"pred_old_label": "unknown", "pred_new_label": "unknown", "pred_should_change": False, "confidence": 0.0, "raw_output": repr(exc)}
        if parsed.get("pred_old_label") == "unknown" and parsed.get("pred_new_label") == "unknown":
            parse_failures += 1
        predictions.append({"id": row["id"], **parsed})
    return predictions, {"status": "ok", "n": len(predictions), "parse_failures": parse_failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/revimpact_pilot_results.json")
    parser.add_argument("--model", default="/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--llm-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    data_path = Path(args.data)
    splits = {split: list(load_jsonl(data_path / f"{split}.jsonl")) for split in ("train", "dev", "test")}
    majority = _majority(splits["train"])
    baseline_functions: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "majority": lambda row: _majority_prediction(row, majority),
        "text_diff": text_diff_prediction,
        "evidence_overlap": evidence_overlap_prediction,
        "condition_rule": condition_rule_prediction,
        "rice_lite": condition_rule_prediction,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"data": str(data_path), "majority": majority, "splits": {}, "diagnostics": _diagnostics(splits)}
    for split, rows in splits.items():
        result["splits"][split] = {}
        for name, function in baseline_functions.items():
            predictions = [function(row) for row in rows]
            _write_predictions(output_path.parent / f"{split}.{name}.predictions.jsonl", predictions)
            result["splits"][split][name] = {
                "overall": evaluate_records(rows, predictions),
                "by_category": _group_metrics(rows, predictions, "category"),
                "by_domain": _group_metrics(rows, predictions, "domain_template"),
            }
    sample = _sample_stratified(splits["test"], args.llm_samples, seed=7)
    llm_predictions, llm_status = _run_qwen(sample, args.model, args.max_new_tokens)
    if llm_predictions:
        _write_predictions(output_path.parent / "test.qwen_sample.predictions.jsonl", llm_predictions)
        sampled_by_id = {row["id"]: row for row in sample}
        llm_rows = [sampled_by_id[pred["id"]] for pred in llm_predictions]
        result["llm"] = {"status": llm_status, "metrics": evaluate_records(llm_rows, llm_predictions), "by_category": _group_metrics(llm_rows, llm_predictions, "category")}
    else:
        result["llm"] = {"status": llm_status}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
