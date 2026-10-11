#!/usr/bin/env python3
"""Evaluate deterministic baselines and an optional trained RICA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from revimpact.evaluation.baselines import condition_rule_prediction, evidence_overlap_prediction, text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl
from revimpact.models.rica import RICAAdapter
from train_rica import LABELS, make_collator, move_batch, condition_features, prompt


def majority_prediction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = sum(bool(row.get("should_change")) for row in rows)
    value = changes >= len(rows) / 2
    return [{"id": row["id"], "pred_should_change": value,
             "pred_old_label": "unknown", "pred_new_label": "unknown", "pred_evidence_ids": []} for row in rows]


def _complete(row, prediction):
    prediction = dict(prediction)
    prediction.setdefault("id", row["id"])
    prediction.setdefault("pred_old_label", "unknown")
    prediction.setdefault("pred_new_label", "unknown")
    prediction.setdefault("pred_should_change", False)
    prediction.setdefault("pred_evidence_ids", [])
    return prediction


@torch.no_grad()
def rica_predictions(rows: list[dict[str, Any]], checkpoint: Path, model_name: str, device: str, batch_size: int, max_length: int, impact_threshold: float = 0.5):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = RICAAdapter(model_name, condition_feature_dim=20).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    # The training collator excludes unsupported rows by design. Predict them
    # as unknown/needs review while retaining them for the separate audit.
    eligible = [row for row in rows if row.get("metadata", {}).get("scorable", True)]
    predictions = {row["id"]: {"id": row["id"], "pred_old_label": "unknown", "pred_new_label": "unknown",
                               "pred_should_change": False, "pred_evidence_ids": [], "confidence": 0.0, "impact_confidence": 0.0} for row in rows}
    collate = make_collator(tokenizer, max_length)
    for start in range(0, len(eligible), batch_size):
        batch = collate(eligible[start:start + batch_size])
        batch = move_batch(batch, torch.device(device))
        output = model(batch["old"]["input_ids"], batch["old"]["attention_mask"],
                       batch["new"]["input_ids"], batch["new"]["attention_mask"], batch["condition_features"])
        old_prob = output["old_logits"].softmax(-1)
        new_prob = output["new_logits"].softmax(-1)
        impact_prob = output["impact_logits"].softmax(-1)
        old_ids = {index: list(row.get("old_evidence", [])) for index, row in enumerate(batch["rows"])}
        new_ids = {index: list(row.get("new_evidence", [])) for index, row in enumerate(batch["rows"])}
        for index, row in enumerate(batch["rows"]):
            evidence = []
            if output["old_evidence_logits"][index].max().item() > 0:
                evidence.extend(str(span.get("source_id") or f"old_evidence:{i}") for i, span in enumerate(old_ids[index]))
            if output["new_evidence_logits"][index].max().item() > 0:
                evidence.extend(str(span.get("source_id") or f"new_evidence:{i}") for i, span in enumerate(new_ids[index]))
            predictions[row["id"]] = {
                "id": row["id"], "pred_old_label": list(LABELS)[int(old_prob[index].argmax())],
                "pred_new_label": list(LABELS)[int(new_prob[index].argmax())],
                "pred_should_change": bool(impact_prob[index, 1] >= impact_threshold),
                "pred_evidence_ids": evidence,
                "confidence": float(max(old_prob[index].max(), new_prob[index].max(), impact_prob[index].max())),
                "impact_confidence": float(impact_prob[index, 1]),
            }
    return [predictions[row["id"]] for row in rows]


def grouped(rows, predictions):
    groups = {}
    for row, prediction in zip(rows, predictions):
        key = row.get("metadata", {}).get("category", "unknown")
        groups.setdefault(key, [[], []])
        groups[key][0].append(row); groups[key][1].append(prediction)
    return {key: evaluate_records(pair[0], pair[1]) for key, pair in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--checkpoint", default="outputs/rica.pt")
    parser.add_argument("--model", default=os.environ.get("REVIMPACT_MODEL_PATH", "/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--output", default="outputs/rica_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--impact-threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = list(load_jsonl(Path(args.data) / f"{args.split}.jsonl"))
    methods: dict[str, list[dict[str, Any]]] = {}
    methods["majority"] = majority_prediction(rows)
    methods["text_diff"] = [_complete(row, text_diff_prediction(row)) for row in rows]
    methods["evidence_overlap"] = [_complete(row, evidence_overlap_prediction(row)) for row in rows]
    # This uses structured revision fields and is intentionally reported as an
    # oracle/reference upper bound, never as a learned method.
    methods["condition_rule_oracle"] = [_complete(row, condition_rule_prediction(row)) for row in rows]
    if Path(args.checkpoint).exists():
        methods["rica"] = rica_predictions(rows, Path(args.checkpoint), args.model, args.device, args.batch_size, args.max_length, args.impact_threshold)
    result = {"n": len(rows), "methods": {name: {"overall": evaluate_records(rows, preds), "groups": grouped(rows, preds)} for name, preds in methods.items()}}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, predictions in methods.items():
        path = output.with_name(output.stem + f"_{name}_predictions.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
