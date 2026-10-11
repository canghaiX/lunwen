#!/usr/bin/env python3
"""Build a balanced RevImpact feasibility pilot and a small VitaminC auxiliary set."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from revimpact.data.schema import Action, Condition, Conclusion, EvidenceSpan, Revision, RevisionRecord
from revimpact.evaluation.metrics import load_jsonl


DOMAIN_SPECS = [
    {
        "domain": "industrial_safety",
        "subject": "A-type pump",
        "condition": {"model": "A", "environment": "high_temperature"},
        "other_condition": {"model": "B", "environment": "high_temperature"},
        "measure": "pressure",
        "unit": "MPa",
        "old": 10.0,
        "new": 8.0,
        "main": "The {subject} pressure limit in {environment} is {limit:g} {unit}.",
        "query": "Is {measure} {value:g} {unit} safe for the {subject} in {environment}?",
        "noise": "The B-type pump pressure limit at normal temperature is 12 MPa.",
    },
    {
        "domain": "software_api",
        "subject": "the upload endpoint",
        "condition": {"endpoint": "upload", "region": "global"},
        "other_condition": {"endpoint": "download", "region": "global"},
        "measure": "timeout",
        "unit": "seconds",
        "old": 30.0,
        "new": 20.0,
        "main": "For {region} clients, the {subject} timeout is {limit:g} {unit}.",
        "query": "Is a {value:g}-{unit} timeout allowed for {subject} for {region} clients?",
        "noise": "The download endpoint has a 60 second timeout for global clients.",
    },
    {
        "domain": "public_technical_guidance",
        "subject": "adult outdoor workers",
        "condition": {"group": "adult", "setting": "outdoor"},
        "other_condition": {"group": "child", "setting": "outdoor"},
        "measure": "exposure",
        "unit": "hours",
        "old": 8.0,
        "new": 6.0,
        "main": "The recommended daily {measure} limit for {subject} is {limit:g} {unit}.",
        "query": "Is {value:g} {unit} of daily {measure} acceptable for {subject}?",
        "noise": "For child indoor activities, the recommended exposure limit is 4 hours.",
    },
]


def _conclusion(value: float, limit: float) -> Conclusion:
    return Conclusion.ENTAILED if value <= limit else Conclusion.CONTRADICTED


def _make_controlled(family_index: int, query_index: int, category: str, spec: dict[str, Any]) -> RevisionRecord:
    profile = f"P{family_index:04d}"
    spec = {
        **spec,
        "subject": f"{spec['subject']} profile {profile}",
        "condition": {**spec["condition"], "profile": profile},
        "other_condition": {**spec["other_condition"], "profile": profile},
    }
    old_limit, new_limit = spec["old"], spec["new"]
    condition_slots = dict(spec["condition"])
    affected = True
    scorable = True
    if category == "changed":
        value = (old_limit + new_limit) / 2
        old_label, new_label = _conclusion(value, old_limit), _conclusion(value, new_limit)
    elif category == "invariant":
        if query_index % 2 == 0:
            value = max(0.5, new_limit - 1)
            old_label = new_label = _conclusion(value, new_limit)
        else:
            value = old_limit - 1
            condition_slots = dict(spec["other_condition"])
            old_label = new_label = Conclusion.UNKNOWN
            affected = False
    elif category == "boundary":
        # Keep paired boundary queries in one family: one lies in the changed
        # interval, the other is safely below the new limit and must remain
        # invariant. This gives the ranking loss an actual contrast pair.
        value = old_limit if query_index % 2 == 0 else max(0.5, new_limit - 1)
        old_label, new_label = _conclusion(value, old_limit), _conclusion(value, new_limit)
    else:
        value = (old_limit + new_limit) / 2
        condition_slots = {key: "unreported" for key in spec["condition"]}
        old_label = new_label = Conclusion.UNKNOWN
        scorable = False
        affected = None

    old_main = spec["main"].format(subject=spec["subject"], environment="high temperature", region="global", limit=old_limit, unit=spec["unit"], measure=spec["measure"])
    new_main = spec["main"].format(subject=spec["subject"], environment="high temperature", region="global", limit=new_limit, unit=spec["unit"], measure=spec["measure"])
    old_document = f"{old_main} {spec['noise']}"
    new_document = f"{new_main} {spec['noise']}"
    query = spec["query"].format(subject=spec["subject"], environment="high temperature", region="global", value=value, unit=spec["unit"], measure=spec["measure"])
    family_id = f"pilot-family-{family_index:04d}"
    old_id, new_id = f"{family_id}-old", f"{family_id}-new"
    revision_condition = Condition(slots=dict(spec["condition"]))
    query_condition = Condition(slots={**condition_slots, "value": value})
    action = Action.HUMAN_REVIEW if not scorable else (Action.UPDATE if old_label != new_label else Action.VERIFY)
    return RevisionRecord(
        id=f"{family_id}-q{query_index:02d}", family_id=family_id, domain=spec["domain"],
        old_version="v1", new_version="v2", old_document=old_document, new_document=new_document,
        query=query, condition=query_condition,
        revision=Revision(predicate=spec["measure"], old_value=old_limit, new_value=new_limit,
                          condition=revision_condition, revision_type="numeric_change", source_id=family_id),
        old_label=old_label, new_label=new_label, should_change=old_label != new_label,
        impact_condition=Condition(slots={**spec["condition"], "value": {"min": new_limit, "max": old_limit}}),
        old_evidence=[EvidenceSpan(version="v1", text=old_main, source_id=old_id)],
        new_evidence=[EvidenceSpan(version="v2", text=new_main, source_id=new_id)],
        action=action, revision_type=category, split=None,
        metadata={"source": "controlled_pilot", "category": category, "scorable": scorable,
                  "affected_condition": affected, "domain_template": spec["domain"]},
    )


def _write_jsonl(path: Path, records: list[RevisionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


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


def _build_auxiliary(vitaminc: Path, output: Path, sample_per_split: int, seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"source": str(vitaminc), "sample_per_split": sample_per_split, "splits": {}}
    for split in ("train", "dev", "test"):
        source = vitaminc / f"{split}.jsonl"
        if not source.exists():
            continue
        rows = _reservoir(source, sample_per_split, seed + len(split))
        target = output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][split] = {
            "n": len(rows),
            "changed": sum(bool(row.get("should_change")) for row in rows),
            "unchanged": sum(not bool(row.get("should_change")) for row in rows),
        }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_pilot(output: Path, families: int, queries_per_family: int, seed: int) -> dict[str, Any]:
    if queries_per_family != 8:
        raise ValueError("the feasibility pilot uses exactly 8 query templates per family")
    total = families * queries_per_family
    changed_n = max(1, round(total * 0.35))
    invariant_n = max(1, round(total * 0.35))
    boundary_n = max(1, round(total * 0.20))
    unsupported_n = total - changed_n - invariant_n - boundary_n
    if unsupported_n < 1:
        unsupported_n = 1
        changed_n -= 1
    categories = (["changed"] * changed_n + ["invariant"] * invariant_n + ["boundary"] * boundary_n + ["unsupported"] * unsupported_n)
    rng = random.Random(seed)
    rng.shuffle(categories)
    records: list[RevisionRecord] = []
    for family_index in range(families):
        spec = DOMAIN_SPECS[family_index % len(DOMAIN_SPECS)]
        for query_index in range(queries_per_family):
            category = categories[family_index * queries_per_family + query_index]
            records.append(_make_controlled(family_index, query_index, category, spec))
    family_ids = list(range(families))
    rng.shuffle(family_ids)
    train_cut, dev_cut = int(families * 0.7), int(families * 0.8)
    split_by_family = {family: "train" if position < train_cut else "dev" if position < dev_cut else "test" for position, family in enumerate(family_ids)}
    by_split: dict[str, list[RevisionRecord]] = collections.defaultdict(list)
    for record in records:
        family_index = int(record.family_id.rsplit("-", 1)[-1])
        split = split_by_family[family_index]
        record.split = split
        by_split[split].append(record)
    output.mkdir(parents=True, exist_ok=True)
    for split, split_records in by_split.items():
        _write_jsonl(output / f"{split}.jsonl", split_records)
    summary = {
        "families": families,
        "queries_per_family": queries_per_family,
        "total_queries": len(records),
        "splits": {split: {"n": len(items), "families": len({item.family_id for item in items})} for split, items in by_split.items()},
        "categories": dict(collections.Counter(item.metadata["category"] for item in records)),
        "domains": dict(collections.Counter(item.domain for item in records)),
        "scorable": sum(bool(item.metadata.get("scorable", True)) for item in records),
        "seed": seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitaminc", default="data/vitaminc_converted")
    parser.add_argument("--output", default="data/revimpact_pilot")
    parser.add_argument("--aux-output", default="data/vitaminc_aux")
    parser.add_argument("--families", type=int, default=400)
    parser.add_argument("--queries-per-family", type=int, default=8)
    parser.add_argument("--aux-sample-per-split", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    pilot_summary = build_pilot(Path(args.output), args.families, args.queries_per_family, args.seed)
    aux_summary = _build_auxiliary(Path(args.vitaminc), Path(args.aux_output), args.aux_sample_per_split, args.seed)
    print(json.dumps({"pilot": pilot_summary, "vitaminc_aux": aux_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
