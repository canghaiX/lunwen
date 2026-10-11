#!/usr/bin/env python3
"""Evaluate selective maintenance with a policy fixed before test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from revimpact.evaluation.baselines import text_diff_prediction, evidence_overlap_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl
from revimpact.evaluation.selective import apply_policy, policy_grid, select_policy
from scan_rica_thresholds import predict_scores


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(rows, predictions):
    result = {"overall": evaluate_records(rows, predictions)}
    for field in ("source_type", "category", "domain", "condition_status"):
        grouped = {}
        for row, prediction in zip(rows, predictions):
            value = str(row.get(field, row.get("metadata", {}).get(field, "unknown")))
            group = grouped.setdefault(value, ([], []))
            group[0].append(row)
            group[1].append(prediction)
        result[f"by_{field}"] = {key: evaluate_records(*pair) for key, pair in grouped.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--checkpoint", default="outputs/rica_selective_seed7.pt")
    parser.add_argument("--model", default=os.environ.get("REVIMPACT_MODEL_PATH", "/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output", default="outputs/rica_selective_dev.json")
    parser.add_argument("--test-output", default="outputs/rica_selective_test.json")
    parser.add_argument("--predictions-output", default="outputs/rica_selective_test_predictions.jsonl")
    parser.add_argument("--threshold-mode", choices=("head", "threshold", "calibrated_head"), default="threshold")
    parser.add_argument("--minimum-changed-recall", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 <= args.minimum_changed_recall <= 1:
        parser.error("minimum changed recall must be between zero and one")
    root = Path(args.data)
    splits = {split: list(load_jsonl(root / f"{split}.jsonl")) for split in ("train", "dev", "test")}
    families = {split: {row["family_id"] for row in rows} for split, rows in splits.items()}
    if any(families[a] & families[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("document families must not overlap across splits")
    provenance = {"data": str(root), "checkpoint": args.checkpoint, "model": args.model,
                  "max_length": args.max_length, "decision_mode": args.threshold_mode,
                  "label_status": "existing labels; no annotation status changed",
                  "dataset_sha256": {split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
                                     for split in splits}, "family_split_disjoint": True,
                  "evidence_status": "not evaluated by this score-only predictor"}
    dev_rows = splits["dev"]
    raw_dev = predict_scores(dev_rows, Path(args.checkpoint), args.model, args.device,
                             args.batch_size, args.max_length)
    selection = select_policy(dev_rows, raw_dev, policy_grid(args.threshold_mode), args.minimum_changed_recall)
    policy = selection["selected"]["policy"]
    output = Path(args.output)
    # Persist the dev decision before running test inference.
    write_json(output.with_name(output.stem + "_selection.json"), selection)
    write_jsonl(output.with_name(output.stem + "_scores.jsonl"), raw_dev)
    dev_report = report(dev_rows, apply_policy(raw_dev, policy))
    write_json(output, {**provenance, "selection": selection, "selected_result": dev_report["overall"],
                        "groups": dev_report})
    test_rows = splits["test"]
    raw_test = predict_scores(test_rows, Path(args.checkpoint), args.model, args.device,
                              args.batch_size, args.max_length)
    test_pred = apply_policy(raw_test, policy)
    test_report = report(test_rows, test_pred)
    baselines = {}
    for name, method in (("text_diff", text_diff_prediction), ("evidence_overlap", evidence_overlap_prediction)):
        # Strip labels, evidence annotations, and category before baseline prediction.
        baseline_pred = [method({key: row[key] for key in ("id", "old_document", "new_document", "query", "condition")
                                 if key in row}) for row in test_rows]
        baselines[name] = report(test_rows, baseline_pred)
    write_json(args.test_output, {**provenance, "selection_source": "dev", "policy": policy,
                                 "dev_constraint_satisfied": selection["constraint_satisfied"],
                                 "result": test_report["overall"], "groups": test_report, "baselines": baselines})
    write_jsonl(args.predictions_output, test_pred)
    test_output = Path(args.test_output)
    write_jsonl(test_output.with_name(test_output.stem + "_scores.jsonl"), raw_test)
    print(json.dumps({"selected": selection["selected"], "test": test_report["overall"],
                      "sources": {key: value["selective"] for key, value in test_report["by_source_type"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
