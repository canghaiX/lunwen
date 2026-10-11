#!/usr/bin/env python3
"""Summarize threshold and ablation outputs for the RevImpact pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from revimpact.evaluation.baselines import text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _metric(value, path, default=None):
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _family_bootstrap(rows, predictions, samples=1000, seed=7):
    families = {}
    for row, prediction in zip(rows, predictions):
        families.setdefault(str(row.get("family_id", row.get("id"))), []).append((row, prediction))
    groups = list(families.values())
    rng = random.Random(seed)
    values = {"impact_f1": [], "invariant_accuracy": [], "changed_recall": []}
    for _ in range(samples):
        selected = [item for _ in groups for item in rng.choice(groups)]
        boot_rows = [item[0] for item in selected]
        boot_predictions = [item[1] for item in selected]
        result = evaluate_records(boot_rows, boot_predictions)
        values["impact_f1"].append(result["impact"]["f1"])
        values["invariant_accuracy"].append(result["invariant_accuracy"])
        values["changed_recall"].append(result["changed_accuracy"])
    return {key: {"lower": sorted(value)[int(0.025 * len(value))],
                  "upper": sorted(value)[int(0.975 * len(value)) - 1]}
            for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-dev", default="outputs/rica_threshold_sweep_dev.json")
    parser.add_argument("--threshold-test", default="outputs/rica_threshold_test.json")
    parser.add_argument("--ablation", default="outputs/ablations/ablation_results.json")
    parser.add_argument("--data-summary", default="data/revimpact_800/summary.json")
    parser.add_argument("--test-data", default="data/revimpact_800/test.jsonl")
    parser.add_argument("--test-predictions", default="outputs/ablations/fixed_rica_full_threshold_predictions.jsonl")
    parser.add_argument("--baseline-test", default="outputs/ablations/fixed_rica_full_seed7_threshold_test.json")
    parser.add_argument("--output-json", default="outputs/rica_feasibility_report.json")
    parser.add_argument("--output-md", default="outputs/rica_feasibility_report.md")
    args = parser.parse_args()
    dev = _read(Path(args.threshold_dev)) or {}
    test = _read(Path(args.threshold_test)) or {}
    ablation = _read(Path(args.ablation)) or {}
    summary = _read(Path(args.data_summary)) or {}
    selected = dev.get("selection", {}).get("selected", {})
    test_overall = test.get("overall", {})
    baseline_test = _read(Path(args.baseline_test)) or {}
    if baseline_test.get("overall"):
        test = baseline_test
        test_overall = baseline_test["overall"]
    baseline_recall = dev.get("selection", {}).get("baseline_changed_recall")
    text_diff_recall = baseline_recall if baseline_recall is not None else 1.0
    test_invariant = _metric(test_overall, ["invariant_accuracy"], 0.0)
    test_changed = _metric(test_overall, ["changed_accuracy"], 0.0)
    test_impact = _metric(test_overall, ["impact", "f1"], 0.0)
    test_rows = list(load_jsonl(args.test_data)) if Path(args.test_data).exists() else []
    test_predictions = list(load_jsonl(args.test_predictions)) if Path(args.test_predictions).exists() else []
    bootstrap = _family_bootstrap(test_rows, test_predictions) if len(test_rows) == len(test_predictions) else {}
    text_diff_test = evaluate_records(test_rows, [text_diff_prediction(row) for row in test_rows]) if test_rows else {}
    text_diff_invariant = _metric(text_diff_test, ["invariant_accuracy"], 0.0)
    text_diff_changed = _metric(text_diff_test, ["changed_accuracy"], 0.0)
    controlled = test.get("by_source_type", {}).get("controlled", {})
    natural = test.get("by_source_type", {}).get("natural_vitaminc", {})
    full = ablation.get("rica_full", {}).get("aggregate", {})
    dual = ablation.get("dual_path_no_condition", {}).get("aggregate", {})
    full_inv = _metric(full, ["invariant_accuracy", "mean"], None)
    dual_inv = _metric(dual, ["invariant_accuracy", "mean"], None)
    criteria = {
        "family_split_no_leakage": not any(summary.get("family_overlap", {}).values()),
        "invariant_plus_15_points": test_invariant >= text_diff_invariant + 0.15,
        "changed_recall_drop_at_most_5_points": test_changed >= text_diff_changed - 0.05,
        "test_impact_f1_present": bool(test_overall),
        "controlled_result_present": bool(controlled),
        "natural_result_present": bool(natural),
        "full_rica_beats_plain_dual": full_inv is None or dual_inv is None or full_inv > dual_inv,
    }
    report = {
        "status": "pilot_only_silver_labels",
        "selected_threshold": selected.get("threshold"),
        "threshold_selection": dev.get("selection", {}),
        "test": {"impact_f1": test_impact, "invariant_accuracy": test_invariant,
                 "changed_recall": test_changed, "controlled": controlled, "natural": natural},
        "ablations": ablation,
        "bootstrap_95": bootstrap,
        "text_diff_test": text_diff_test,
        "criteria": criteria,
        "continue_recommended": all(criteria.values()),
        "data_summary": summary,
        "limitations": [
            "Labels are automatic or controlled pilot labels until human review is complete.",
            "Bootstrap confidence intervals require family-level resampling and are not inferred from point estimates.",
        ],
    }
    out_json = Path(args.output_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# RevImpact RICA Feasibility Report", "", f"Status: `{report['status']}`", "",
             f"Selected dev threshold: `{report['selected_threshold']}`",
             f"Test Impact F1: `{test_impact:.4f}`",
             f"Test Invariant Accuracy: `{test_invariant:.4f}`",
             f"Test Changed Recall: `{test_changed:.4f}`",
             f"Continue recommendation: `{report['continue_recommended']}`", "", "## Criteria"]
    lines.extend(f"- {key}: `{value}`" for key, value in criteria.items())
    lines.extend(["", "## Limitations", "- Automatic/controlled pilot labels are not final gold labels.",
                  "- Human review is required before reporting paper results."])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "output_md": args.output_md,
                      "continue_recommended": report["continue_recommended"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
