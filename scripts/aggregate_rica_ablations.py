#!/usr/bin/env python3
"""Aggregate threshold-selected RICA ablation test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="outputs/ablations")
    parser.add_argument("--prefix", default="fixed_")
    parser.add_argument("--output", default="outputs/ablations/ablation_results.json")
    args = parser.parse_args()
    root = Path(args.directory)
    grouped = {}
    for path in sorted(root.glob("*_threshold_test.json")):
        if not path.name.startswith(args.prefix):
            continue
        name = path.name.removesuffix("_threshold_test.json")
        name = name.removeprefix(args.prefix)
        if "_seed" in name:
            name = name.rsplit("_seed", 1)[0]
        value = json.loads(path.read_text(encoding="utf-8"))
        overall = value.get("overall", {})
        grouped.setdefault(name, []).append({
            "path": str(path),
            "threshold": value.get("selected_threshold"),
            "impact_f1": overall.get("impact", {}).get("f1", 0.0),
            "impact_precision": overall.get("impact", {}).get("precision", 0.0),
            "impact_recall": overall.get("impact", {}).get("recall", 0.0),
            "changed_recall": overall.get("changed_accuracy", 0.0),
            "invariant_accuracy": overall.get("invariant_accuracy", 0.0),
            "unchanged_accuracy": overall.get("unchanged_accuracy", 0.0),
            "old_accuracy": overall.get("old_accuracy", 0.0),
            "new_accuracy": overall.get("new_accuracy", 0.0),
        })
    summary = {}
    for name, values in grouped.items():
        metrics = ["impact_f1", "changed_recall", "invariant_accuracy", "unchanged_accuracy", "old_accuracy", "new_accuracy"]
        aggregate = {}
        for metric in metrics:
            numbers = [float(item[metric]) for item in values]
            aggregate[metric] = {"mean": mean(numbers), "std": stdev(numbers) if len(numbers) > 1 else 0.0,
                                 "values": numbers}
        summary[name] = {"n_runs": len(values), "runs": values, "aggregate": aggregate}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
