"""Controlled pilot data with changed, invariant, and boundary cases."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from revimpact.data.schema import Action, Condition, Conclusion, EvidenceSpan, Revision, RevisionRecord


def _record(index: int, kind: str, rng: random.Random) -> RevisionRecord:
    old_limit, new_limit = 10.0, 8.0
    if kind == "changed":
        model, environment, pressure = "A", "high_temperature", 9.0
        old_label, new_label = Conclusion.ENTAILED, Conclusion.CONTRADICTED
        action = Action.UPDATE
    elif kind == "invariant":
        model, environment, pressure = rng.choice([
            ("A", "high_temperature", 7.0),
            ("B", "high_temperature", 9.0),
            ("A", "normal_temperature", 9.0),
        ])
        if model == "A" and environment == "high_temperature":
            old_label = new_label = Conclusion.ENTAILED
        else:
            old_label = new_label = Conclusion.UNKNOWN
        action = Action.VERIFY
    elif kind == "boundary":
        model, environment, pressure = "A", "high_temperature", 11.0
        old_label = new_label = Conclusion.CONTRADICTED
        action = Action.VERIFY
    else:
        model, environment, pressure = "B", "normal_temperature", 9.0
        old_label = new_label = Conclusion.UNKNOWN
        action = Action.KEEP
    condition = Condition(slots={"model": model, "environment": environment, "pressure": pressure})
    revision_condition = Condition(slots={"model": "A", "environment": "high_temperature"})
    old_doc = f"Model A in high temperature has a pressure limit of {old_limit:g} MPa."
    new_doc = f"Model A in high temperature has a pressure limit of {new_limit:g} MPa."
    query = f"Is pressure {pressure:g} MPa safe for model {model} in {environment.replace('_', ' ')}?"
    old_evidence = [EvidenceSpan(version="old", text=old_doc)]
    new_evidence = [EvidenceSpan(version="new", text=new_doc)]
    return RevisionRecord(
        id=f"pilot-{index:05d}", family_id=f"family-{index // 4:04d}", domain="industrial_safety",
        old_version="v1", new_version="v2", old_document=old_doc, new_document=new_doc,
        query=query, condition=condition,
        revision=Revision(predicate="pressure_limit", old_value=old_limit, new_value=new_limit,
                          condition=revision_condition, revision_type="numeric_change"),
        old_label=old_label, new_label=new_label, should_change=old_label != new_label,
        impact_condition=Condition(slots={"model": "A", "environment": "high_temperature", "pressure": {"min": 8, "max": 10}}),
        old_evidence=old_evidence, new_evidence=new_evidence, action=action, revision_type=kind,
        metadata={"synthetic": True},
    )


def generate(n_pairs: int = 120, seed: int = 7):
    rng = random.Random(seed)
    kinds = ["changed", "invariant", "boundary", "unsupported"]
    return [_record(i, kinds[i % len(kinds)], rng) for i in range(n_pairs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pilot.jsonl")
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in generate(args.n_pairs, args.seed):
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {args.n_pairs} records to {output}")


if __name__ == "__main__":
    main()
