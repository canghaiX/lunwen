#!/usr/bin/env python3
"""Create the human-review template or merge verified annotations into data."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from revimpact.data.schema import Action, Conclusion
from revimpact.evaluation.metrics import load_jsonl


LABELS = {item.value for item in Conclusion}
ACTIONS = {item.value for item in Action}
STATUSES = {"verified", "uncertain", "excluded"}


def _natural_rows(data: Path) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for split in ("train", "dev", "test")
        for row in load_jsonl(data / f"{split}.jsonl")
        if row.get("metadata", {}).get("source_type") == "natural_vitaminc"
    }


def _load_map(path: Path | None) -> dict[str, dict[str, Any]]:
    return {} if path is None or not path.exists() else {row["id"]: row for row in load_jsonl(path)}


def create_template(data: Path, initial_path: Path, review_path: Path, output: Path) -> dict[str, Any]:
    rows = _natural_rows(data)
    initial, reviews = _load_map(initial_path), _load_map(review_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row_id in sorted(rows):
            initial_value = initial.get(row_id, {}).get("prediction", {})
            review_value = reviews.get(row_id, {}).get("review", {})
            handle.write(json.dumps({
                "id": row_id, "family_id": rows[row_id]["family_id"],
                "old_document": rows[row_id]["old_document"],
                "new_document": rows[row_id]["new_document"],
                "query": rows[row_id]["query"],
                "condition": rows[row_id].get("condition", {}),
                "old_evidence": rows[row_id].get("old_evidence", []),
                "new_evidence": rows[row_id].get("new_evidence", []),
                "qwen80b_prediction": initial_value,
                # Keep the historical field only for an actual 35B review;
                # fallback reviews are exposed through qwen_review.
                "qwen35b_review": review_value if reviews.get(row_id, {}).get("reviewer") == "qwen3.6-35b" else {},
                "qwen_review": review_value,
                "qwen_reviewer": reviews.get(row_id, {}).get("reviewer"),
                "human_old_label": None, "human_new_label": None, "human_should_change": None,
                "human_old_evidence_ids": [], "human_new_evidence_ids": [],
                "human_action": None, "review_status": "pending", "reviewer_id": None,
                "review_notes": "",
            }, ensure_ascii=False) + "\n")
    return {"n": len(rows), "output": str(output), "initial_n": len(initial), "review_n": len(reviews)}


def _valid_human(row: dict[str, Any]) -> tuple[bool, str | None]:
    status = row.get("review_status")
    if status not in STATUSES:
        return False, "review_status must be verified, uncertain, or excluded"
    if status == "verified":
        if row.get("human_old_label") not in LABELS or row.get("human_new_label") not in LABELS:
            return False, "verified rows require valid old/new labels"
        if not isinstance(row.get("human_should_change"), bool):
            return False, "verified rows require boolean human_should_change"
        if row.get("human_action") not in ACTIONS:
            return False, "verified rows require a valid human_action"
    return True, None


def _apply_human(row: dict[str, Any], human: dict[str, Any], initial: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(row))
    metadata = dict(result.get("metadata", {}))
    status = human.get("review_status")
    reviewer = review.get("reviewer")
    metadata.update({
        "annotation_status": "human_verified" if status == "verified" else f"human_{status}",
        "human_review_status": status,
        "human_reviewer_id": human.get("reviewer_id"),
        "human_review_notes": human.get("review_notes", ""),
        "qwen80b_annotation_file": "annotations_qwen80b.jsonl",
        "qwen_reviewer": reviewer,
        "qwen_review_file": "annotations_qwen35b_review.jsonl" if reviewer == "qwen3.6-35b" else "annotations_qwen4b_review_normalized.jsonl",
    })
    result["metadata"] = metadata
    if status == "verified":
        result["old_label"] = human["human_old_label"]
        result["new_label"] = human["human_new_label"]
        result["should_change"] = human["human_should_change"]
        result["action"] = human["human_action"]
        result["metadata"]["human_old_evidence_ids"] = [str(x) for x in human.get("human_old_evidence_ids", [])]
        result["metadata"]["human_new_evidence_ids"] = [str(x) for x in human.get("human_new_evidence_ids", [])]
        result["metadata"]["scorable"] = True
    else:
        # Keep uncertain/excluded examples for audit but remove them from the
        # primary metrics until a human resolves them.
        result["old_label"] = result.get("old_label", "unknown") if result.get("old_label") in LABELS else "unknown"
        result["new_label"] = result.get("new_label", "unknown") if result.get("new_label") in LABELS else "unknown"
        result["metadata"]["scorable"] = False
    return result


def finalize(data: Path, human_path: Path, output: Path, initial_path: Path | None, review_path: Path | None) -> dict[str, Any]:
    natural = _natural_rows(data)
    human = _load_map(human_path)
    initial, reviews = _load_map(initial_path), _load_map(review_path)
    errors = []
    for row_id in natural:
        if row_id not in human:
            errors.append({"id": row_id, "error": "missing human annotation"})
            continue
        valid, error = _valid_human(human[row_id])
        if not valid:
            errors.append({"id": row_id, "error": error})
    if errors:
        raise ValueError(json.dumps({"human_annotation_errors": errors[:20], "n": len(errors)}, ensure_ascii=False))

    output.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    for split in ("train", "dev", "test"):
        target = output / f"{split}.jsonl"
        rows_out = []
        for row in load_jsonl(data / f"{split}.jsonl"):
            if row["id"] in natural:
                row = _apply_human(row, human[row["id"]], initial.get(row["id"], {}), reviews.get(row["id"], {}))
                counts[row["metadata"].get("annotation_status", "unknown")] += 1
            else:
                row = json.loads(json.dumps(row))
                row.setdefault("metadata", {})["annotation_status"] = "controlled_auto"
                counts["controlled_auto"] += 1
            rows_out.append(row)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "natural_n": len(natural), "human_n": len(human), "annotation_status": dict(counts),
        "initial_n": len(initial), "review_n": len(reviews),
        "unresolved_n": sum(1 for row in human.values() if row.get("review_status") != "verified"),
        "source": str(data), "human_file": str(human_path),
    }
    (output / "annotation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"annotation_audit": audit, "counts": dict(counts)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("template", "finalize"), required=True)
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--initial", default="data/revimpact_800/annotations_qwen80b.jsonl")
    parser.add_argument("--review", default="data/revimpact_800/annotations_qwen35b_review.jsonl")
    parser.add_argument("--human", default="data/revimpact_800/annotations_human_final.jsonl")
    parser.add_argument("--output", default="data/revimpact_800_annotated")
    args = parser.parse_args()
    if args.mode == "template":
        print(json.dumps(create_template(Path(args.data), Path(args.initial), Path(args.review), Path(args.human)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(finalize(Path(args.data), Path(args.human), Path(args.output), Path(args.initial), Path(args.review)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
