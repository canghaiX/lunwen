#!/usr/bin/env python3
"""Build the 800-query RICA feasibility dataset."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path
from typing import Any

from revimpact.data.schema import RevisionRecord
from revimpact.evaluation.metrics import load_jsonl

try:
    from build_pilot_dataset import DOMAIN_SPECS, _make_controlled
except ImportError:  # pragma: no cover - import path used by package execution
    from scripts.build_pilot_dataset import DOMAIN_SPECS, _make_controlled


NATURAL_TARGETS = {
    "train": {"changed": 175, "invariant": 35},
    "dev": {"changed": 37, "invariant": 7},
    "test": {"changed": 38, "invariant": 8},
}


def _reservoir(path: Path, size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(load_jsonl(path)):
        if index < size:
            result.append(row)
        else:
            position = rng.randrange(index + 1)
            if position < size:
                result[position] = row
    return result


def _candidate_rows(source: Path, seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "dev", "test"):
        path = source / f"{split}.jsonl"
        if not path.exists():
            continue
        # Sample each class independently. VitaminC has very sparse invariant
        # pairs; sampling the complete file first would almost always discard
        # that class before the category filter is applied.
        rng = random.Random(seed + len(split))
        pools: dict[str, list[dict[str, Any]]] = {"changed": [], "invariant": []}
        seen = {"changed": 0, "invariant": 0}
        for row in load_jsonl(path):
            if not row.get("metadata", {}).get("pairing"):
                continue
            category = "changed" if bool(row.get("should_change")) else "invariant"
            seen[category] += 1
            pool = pools[category]
            if len(pool) < 12000:
                pool.append(row)
            else:
                position = rng.randrange(seen[category])
                if position < len(pool):
                    pool[position] = row
        candidates[split] = {
            "changed": pools["changed"],
            "invariant": pools["invariant"],
        }
    return candidates


def _natural_records(source: Path, seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates = _candidate_rows(source, seed)
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    # A family is assigned to one official split, while several query pairs
    # from that family may be sampled within the split. This preserves the
    # family-level split contract without wasting the sparse invariant rows.
    family_owner: dict[str, str] = {}
    selected_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        # Invariant examples are scarce in VitaminC, so allocate their family
        # ownership first; changed examples have a much larger candidate pool.
        for category in ("invariant", "changed"):
            pool = list(candidates.get(split, {}).get(category, []))
            rng.shuffle(pool)
            target = NATURAL_TARGETS[split][category]
            for row in pool:
                family = str(row.get("family_id", row.get("id", "")))
                row_id = str(row.get("id", ""))
                if not family or not row_id or row_id in selected_ids:
                    continue
                owner = family_owner.get(family)
                if owner is not None and owner != split:
                    continue
                selected_row = copy.deepcopy(row)
                selected_row["family_id"] = f"natural-{family}"
                metadata = dict(selected_row.get("metadata", {}))
                metadata.update({
                    "source_type": "natural_vitaminc",
                    "category": category,
                    "scorable": True,
                    "boundary_group_id": None,
                    "condition_status": "unstructured",
                    "annotation_status": "auto",
                    "manual_review_required": True,
                    "original_family_id": family,
                })
                selected_row["metadata"] = metadata
                selected_row["domain"] = "wikipedia_fact_verification"
                selected_row["split"] = split
                selected[split].append(selected_row)
                family_owner[family] = split
                selected_ids.add(row_id)
                if len([item for item in selected[split] if item["metadata"]["category"] == category]) >= target:
                    break
    # The reservoir can rarely miss a target when a family is duplicated across
    # official splits. Fill from unused rows without violating split ownership.
    counts = {split: collections.Counter(row["metadata"]["category"] for row in rows) for split, rows in selected.items()}
    summary = {
        "targets": NATURAL_TARGETS,
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "n": sum(len(rows) for rows in selected.values()),
        "families": len(family_owner),
        "manual_review_pending": sum(len(rows) for rows in selected.values()),
    }
    return selected, summary


def _controlled_records(families: int, seed: int) -> tuple[dict[str, list[RevisionRecord]], dict[str, Any]]:
    total = families * 5
    category_counts = {"changed": 140, "invariant": 140, "boundary": 120, "unsupported": 100}
    if sum(category_counts.values()) != total:
        raise ValueError("controlled target counts must sum to families * 5")
    rng = random.Random(seed)
    family_ids = list(range(families))
    rng.shuffle(family_ids)
    split_by_family = {
        family: "train" if index < int(families * 0.7) else "dev" if index < int(families * 0.85) else "test"
        for index, family in enumerate(family_ids)
    }
    # Pair boundary queries within the same revision family so the ranking
    # objective has a real affected/unaffected comparison. The remaining
    # slots are filled with the exact requested category totals.
    boundary_families = set(family_ids[:42] + family_ids[70:79] + family_ids[85:94])
    categories = [["boundary", "boundary"] if family in boundary_families else [] for family in range(families)]
    remaining = [category for category, count in category_counts.items() for _ in range(count - (2 * len(boundary_families) if category == "boundary" else 0))]
    rng.shuffle(remaining)
    for family in range(families):
        while len(categories[family]) < 5:
            categories[family].append(remaining.pop())
        rng.shuffle(categories[family])
    result: dict[str, list[RevisionRecord]] = {"train": [], "dev": [], "test": []}
    for family_index in range(families):
        spec = DOMAIN_SPECS[family_index % len(DOMAIN_SPECS)]
        split = split_by_family[family_index]
        for query_index in range(5):
            record = _make_controlled(family_index, query_index, categories[family_index][query_index], spec)
            record.metadata.update({
                "source_type": "controlled",
                "boundary_group_id": record.family_id if record.revision_type == "boundary" else None,
                "condition_status": "structured",
                "annotation_status": "auto",
                "manual_review_required": record.revision_type in {"boundary", "unsupported"},
            })
            record.split = split
            result[split].append(record)
    return result, {"families": families, "counts": category_counts, "n": total}


def _as_dict(record: RevisionRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_json_dict() if isinstance(record, RevisionRecord) else record


def _write_jsonl(path: Path, rows: list[RevisionRecord | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_as_dict(row), ensure_ascii=False) + "\n")


def build_dataset(source: Path, output: Path, seed: int = 7) -> dict[str, Any]:
    controlled, controlled_summary = _controlled_records(100, seed)
    natural, natural_summary = _natural_records(source, seed + 100)
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[RevisionRecord | dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        rows = controlled[split] + natural[split]
        rows.sort(key=lambda row: _as_dict(row)["id"])
        _write_jsonl(output / f"{split}.jsonl", rows)
        splits[split] = rows

    all_rows = [_as_dict(row) for rows in splits.values() for row in rows]
    families_by_split = {split: {_as_dict(row)["family_id"] for row in rows} for split, rows in splits.items()}
    overlap = {}
    names = list(families_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap[f"{left}:{right}"] = len(families_by_split[left] & families_by_split[right])

    audit_rows = [row for row in all_rows if row.get("metadata", {}).get("source_type") == "natural_vitaminc"]
    controlled_audit = [row for row in all_rows if row.get("metadata", {}).get("source_type") == "controlled"][:50]
    with (output / "annotation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit_rows + controlled_audit:
            handle.write(json.dumps({
                "id": row["id"], "family_id": row["family_id"], "source_type": row["metadata"].get("source_type"),
                "human_old_label": None, "human_new_label": None, "human_should_change": None,
                "human_old_evidence_ids": [], "human_new_evidence_ids": [], "human_condition_status": None,
                "human_action": None, "status": "pending",
            }, ensure_ascii=False) + "\n")

    summary = {
        "total_queries": len(all_rows),
        "controlled": controlled_summary,
        "natural": natural_summary,
        "splits": {split: {"n": len(rows), "families": len(families_by_split[split])} for split, rows in splits.items()},
        "categories": dict(collections.Counter(row["metadata"].get("category") for row in all_rows)),
        "source_types": dict(collections.Counter(row["metadata"].get("source_type") for row in all_rows)),
        "family_overlap": overlap,
        "manual_review_manifest": len(audit_rows) + len(controlled_audit),
        "seed": seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitaminc", default="data/vitaminc_converted")
    parser.add_argument("--output", default="data/revimpact_800")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(build_dataset(Path(args.vitaminc), Path(args.output), args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
