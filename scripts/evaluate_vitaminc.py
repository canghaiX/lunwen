#!/usr/bin/env python3
"""Run VitaminC baselines and write prediction/evaluation artifacts."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from revimpact.evaluation.baselines import text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl


def _overlap_prediction(row):
    claim_tokens = {token.lower() for token in row["query"].split() if len(token) > 2}
    old_tokens = {token.lower() for token in row["old_document"].split()}
    new_tokens = {token.lower() for token in row["new_document"].split()}
    old_score = len(claim_tokens & old_tokens) / max(len(claim_tokens), 1)
    new_score = len(claim_tokens & new_tokens) / max(len(claim_tokens), 1)
    changed = abs(old_score - new_score) > 0.05
    return {"id": row["id"], "pred_should_change": changed, "confidence": abs(old_score - new_score)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/vitaminc_baselines.json")
    args = parser.parse_args()
    data_path = Path(args.data)
    files = sorted(data_path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no converted JSONL files under {data_path}")
    result = {}
    all_family_ids = {}
    all_documents = {}
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.name == "conversion_summary.json":
            continue
        rows = list(load_jsonl(file))
        all_family_ids[file.stem] = [row.get("family_id") for row in rows]
        all_documents[file.stem] = [(row.get("old_document", ""), row.get("new_document", "")) for row in rows]
        for name, fn in (("text_diff", text_diff_prediction), ("evidence_overlap", _overlap_prediction)):
            predictions = [fn(row) for row in rows]
            pred_path = output_dir / f"{file.stem}.{name}.predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as handle:
                for prediction in predictions:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            result[f"{file.stem}/{name}"] = evaluate_records(rows, predictions)
            paired_indices = [index for index, row in enumerate(rows) if row.get("metadata", {}).get("pairing")]
            if paired_indices and len(paired_indices) != len(rows):
                paired_rows = [rows[index] for index in paired_indices]
                paired_predictions = [predictions[index] for index in paired_indices]
                result[f"{file.stem}/{name}/paired_only"] = evaluate_records(paired_rows, paired_predictions)
    split_names = list(all_family_ids)
    family_overlap = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            family_overlap[f"{left}:{right}"] = len(set(all_family_ids[left]) & set(all_family_ids[right]))
    document_pairs = {}
    for name, pairs in all_documents.items():
        document_pairs[name] = {"n": len(pairs), "duplicate_pairs": len(pairs) - len(set(pairs))}
    result["diagnostics"] = {"family_overlap": family_overlap, "document_pairs": document_pairs}
    distribution = {}
    for file in files:
        rows = list(load_jsonl(file))
        distribution[file.stem] = {
            "n": len(rows),
            "should_change": dict(collections.Counter(str(row.get("should_change")) for row in rows)),
            "old_label": dict(collections.Counter(row.get("old_label") for row in rows)),
            "new_label": dict(collections.Counter(row.get("new_label") for row in rows)),
            "revision_type": dict(collections.Counter(row.get("revision_type") for row in rows)),
            "pairing": dict(collections.Counter(row.get("metadata", {}).get("pairing", "single") for row in rows)),
        }
    result["diagnostics"]["distribution"] = distribution
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
