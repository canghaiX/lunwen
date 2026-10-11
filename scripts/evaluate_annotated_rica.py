#!/usr/bin/env python3
"""Audit annotation agreement and evaluate baselines on finalized data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from revimpact.evaluation.baselines import evidence_overlap_prediction, text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl


def _maps(path: Path | None) -> dict[str, dict[str, Any]]:
    return {} if path is None or not path.exists() else {row["id"]: row for row in load_jsonl(path)}


def _agreement(left: list[Any], right: list[Any]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    return {"n": len(pairs), "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None}


def annotation_agreement(initial, review, human):
    result = {}
    for field in ("old_label", "new_label", "should_change"):
        initial_values, review_values, human_values = [], [], []
        for row_id in sorted(set(initial) & set(review) & set(human)):
            initial_values.append(initial[row_id].get("prediction", {}).get(field))
            review_field = review[row_id].get("review", {}).get(field, {})
            review_values.append(review_field.get("value") if review_field.get("decision") != "uncertain" else None)
            human_values.append(human[row_id].get({"old_label": "human_old_label", "new_label": "human_new_label", "should_change": "human_should_change"}[field]))
        result[field] = {
            "qwen80b_vs_reviewer": _agreement(initial_values, review_values),
            "reviewer_vs_human": _agreement(review_values, human_values),
            "qwen80b_vs_human": _agreement(initial_values, human_values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800_annotated")
    parser.add_argument("--initial", default="data/revimpact_800/annotations_qwen80b.jsonl")
    parser.add_argument("--review", default="data/revimpact_800/annotations_qwen35b_review.jsonl")
    parser.add_argument("--human", default="data/revimpact_800/annotations_human_final.jsonl")
    parser.add_argument("--output", default="outputs/annotated_rica_results.json")
    args = parser.parse_args()

    initial, review, human = _maps(Path(args.initial)), _maps(Path(args.review)), _maps(Path(args.human))
    methods = {}
    for split in ("train", "dev", "test"):
        rows = list(load_jsonl(Path(args.data) / f"{split}.jsonl"))
        for name, fn in (("text_diff", text_diff_prediction), ("evidence_overlap", evidence_overlap_prediction)):
            predictions = []
            for row in rows:
                prediction = dict(fn(row))
                prediction.setdefault("pred_old_label", "unknown")
                prediction.setdefault("pred_new_label", "unknown")
                predictions.append(prediction)
            methods.setdefault(name, {})[split] = evaluate_records(rows, predictions)

    result = {
        "annotation_agreement": annotation_agreement(initial, review, human),
        "methods": methods,
        "n_initial": len(initial), "n_review": len(review), "n_human": len(human),
        "data": args.data,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
