#!/usr/bin/env python3
"""Aggregate RICA-Select test results across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/rica_selective_summary.json")
    args = parser.parse_args()
    runs = []
    for filename in args.files:
        path = Path(filename)
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        runs.append({
            "path": str(path),
            "impact_f1": result["impact"]["f1"],
            "changed_recall": result["changed_recall"],
            "invariant_accuracy": result["invariant_accuracy"],
            "selective_macro_f1": result["selective"]["macro_f1"],
            "selective_coverage": result["selective"]["coverage"],
            "selective_risk": result["selective"]["risk"],
            "abstain_f1": result["selective"]["per_class_f1"]["abstain"],
        })
    summary = {"n_runs": len(runs), "runs": runs, "aggregate": {}}
    for metric in runs[0]:
        if metric == "path":
            continue
        values = [float(run[metric]) for run in runs]
        summary["aggregate"][metric] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
