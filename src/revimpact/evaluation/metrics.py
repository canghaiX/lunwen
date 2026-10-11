"""Task metrics that do not depend on a particular model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def load_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _binary_prf(gold, pred):
    tp = sum(a and b for a, b in zip(gold, pred))
    fp = sum((not a) and b for a, b in zip(gold, pred))
    fn = sum(a and (not b) for a, b in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _macro_f1(gold, pred, labels):
    scores = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, pred))
        fp = sum(a != label and b == label for a, b in zip(gold, pred))
        fn = sum(a == label and b != label for a, b in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"macro_f1": sum(scores.values()) / max(len(labels), 1), "per_class_f1": scores}


def _evidence_scores(rows: list[Mapping[str, Any]], predictions: list[Mapping[str, Any]]) -> Dict[str, float]:
    gold_sets = []
    pred_sets = []
    for row, prediction in zip(rows, predictions):
        gold = set()
        for key in ("old_evidence", "new_evidence"):
            for index, span in enumerate(row.get(key, []) or []):
                if isinstance(span, Mapping):
                    gold.add(str(span.get("source_id") or f"{key}:{index}"))
        predicted = {str(item) for item in prediction.get("pred_evidence_ids", [])}
        gold_sets.append(gold)
        pred_sets.append(predicted)
    if not gold_sets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_positive = sum(len(gold & predicted) for gold, predicted in zip(gold_sets, pred_sets))
    predicted_total = sum(len(predicted) for predicted in pred_sets)
    gold_total = sum(len(gold) for gold in gold_sets)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_records(records: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    rows = list(records)
    if predictions is None:
        raise ValueError("predictions are required; pass an explicit baseline or model output")
    preds = list(predictions)
    if len(rows) != len(preds):
        raise ValueError("records and predictions must have equal length")
    if any(row.get("id") != pred.get("id") for row, pred in zip(rows, preds)):
        raise ValueError("prediction IDs must match records in order")
    for pred in preds:
        if "pred_decision" in pred:
            if pred["pred_decision"] not in {"change", "keep", "abstain"}:
                raise ValueError("invalid selective decision")
            if bool(pred.get("pred_should_change", False)) != (pred["pred_decision"] == "change"):
                raise ValueError("pred_should_change is inconsistent with pred_decision")
    scorable_mask = [bool(row.get("metadata", {}).get("scorable", True)) for row in rows]
    scorable_rows = [row for row, keep in zip(rows, scorable_mask) if keep]
    scorable_preds = [pred for pred, keep in zip(preds, scorable_mask) if keep]
    gold_change = [bool(row["should_change"]) for row in scorable_rows]
    pred_change = [bool(pred.get("pred_should_change", False)) for pred in scorable_preds]
    impact = _binary_prf(gold_change, pred_change)
    old_acc = sum(row["old_label"] == pred.get("pred_old_label") for row, pred in zip(scorable_rows, scorable_preds)) / max(len(scorable_rows), 1)
    new_acc = sum(row["new_label"] == pred.get("pred_new_label") for row, pred in zip(scorable_rows, scorable_preds)) / max(len(scorable_rows), 1)
    old_gold = [row["old_label"] for row in scorable_rows]
    new_gold = [row["new_label"] for row in scorable_rows]
    old_pred = [pred.get("pred_old_label", "unknown") for pred in scorable_preds]
    new_pred = [pred.get("pred_new_label", "unknown") for pred in scorable_preds]
    labels = ["entailed", "contradicted", "unknown"]
    invariant = [not item for item in gold_change]
    invariant_acc = sum(pred.get("pred_decision", "keep" if not pred.get("pred_should_change", False) else "change") == "keep"
                        for pred, keep in zip(scorable_preds, invariant) if keep) / max(sum(invariant), 1)
    changed_rows = [index for index, value in enumerate(gold_change) if value]
    unchanged_rows = [index for index, value in enumerate(gold_change) if not value]
    changed_accuracy = sum(pred_change[index] for index in changed_rows) / max(len(changed_rows), 1)
    unchanged_accuracy = invariant_acc
    unsupported_rows = [row for row, keep in zip(rows, scorable_mask) if not keep]
    unsupported_preds = [pred for pred, keep in zip(preds, scorable_mask) if not keep]
    unsupported_accuracy = (
        sum(not bool(pred.get("pred_should_change", False)) for pred in unsupported_preds) / len(unsupported_rows)
        if unsupported_rows else None
    )
    # Selective maintenance metrics are reported when a predictor emits an
    # explicit change/keep/abstain decision. They include unsupported rows only
    # for abstention auditing and do not alter the historical binary metrics.
    decision_gold = [
        "abstain" if not bool(row.get("metadata", {}).get("scorable", True))
        else ("change" if bool(row["should_change"]) else "keep")
        for row in rows
    ]
    decision_pred = [str(pred.get("pred_decision", "change" if pred.get("pred_should_change", False) else "keep"))
                     for pred in preds]
    decision_labels = ["change", "keep", "abstain"]
    decision_f1 = _macro_f1(decision_gold, decision_pred, decision_labels)
    answered = [(g, p) for g, p in zip(decision_gold, decision_pred) if p != "abstain"]
    answered_errors = sum(g != p for g, p in answered)
    selective = {
        "accuracy": sum(g == p for g, p in zip(decision_gold, decision_pred)) / max(len(decision_gold), 1),
        "macro_f1": decision_f1["macro_f1"],
        "per_class_f1": decision_f1["per_class_f1"],
        "coverage": len(answered) / max(len(decision_gold), 1),
        "risk": answered_errors / max(len(answered), 1),
        "human_review_rate": sum(p == "abstain" for p in decision_pred) / max(len(decision_pred), 1),
        "keep_accuracy": invariant_acc,
        "changed_abstention_rate": sum(g == "change" and p == "abstain" for g, p in zip(decision_gold, decision_pred)) / max(sum(g == "change" for g in decision_gold), 1),
        "unchanged_abstention_rate": sum(g == "keep" and p == "abstain" for g, p in zip(decision_gold, decision_pred)) / max(sum(g == "keep" for g in decision_gold), 1),
        "counts": {label: sum(g == label for g in decision_gold) for label in decision_labels},
        "predicted_counts": {label: sum(p == label for p in decision_pred) for label in decision_labels},
    }
    return {
        "n": len(rows),
        "scorable_n": len(scorable_rows),
        "unsupported_n": len(unsupported_rows),
        "impact": impact,
        "old_accuracy": old_acc,
        "old_classification": _macro_f1(old_gold, old_pred, labels),
        "new_accuracy": new_acc,
        "new_classification": _macro_f1(new_gold, new_pred, labels),
        "unknown_old_accuracy": sum(g == "unknown" and p == "unknown" for g, p in zip(old_gold, old_pred)) / max(sum(g == "unknown" for g in old_gold), 1),
        "unknown_new_accuracy": sum(g == "unknown" and p == "unknown" for g, p in zip(new_gold, new_pred)) / max(sum(g == "unknown" for g in new_gold), 1),
        "changed_accuracy": changed_accuracy,
        "unchanged_accuracy": unchanged_accuracy,
        "invariant_accuracy": invariant_acc,
        "changed_recall": changed_accuracy,
        "false_update_rate": sum(pred_change[index] for index in unchanged_rows) / len(unchanged_rows) if unchanged_rows else None,
        "missed_update_rate": 1.0 - changed_accuracy if changed_rows else None,
        "unsupported_update_rate": (sum(bool(pred.get("pred_should_change")) for pred in unsupported_preds) / len(unsupported_preds)
                                    if unsupported_preds else None),
        "unsupported_human_review_rate": (sum(pred.get("pred_action") == "human_review" for pred in unsupported_preds) / len(unsupported_preds)
                                          if unsupported_preds and all("pred_action" in pred for pred in unsupported_preds) else None),
        "evidence": _evidence_scores(scorable_rows, scorable_preds),
        "unsupported_accuracy": unsupported_accuracy,
        "selective": selective,
    }
