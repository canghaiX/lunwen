#!/usr/bin/env python3
"""Select an RICA impact threshold on dev and evaluate it once on test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from revimpact.evaluation.baselines import text_diff_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl
from revimpact.models.rica import RICAAdapter
from train_rica import LABELS, condition_features, make_collator, move_batch


def _complete(row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    value = dict(prediction)
    value.setdefault("id", row["id"])
    value.setdefault("pred_old_label", "unknown")
    value.setdefault("pred_new_label", "unknown")
    value.setdefault("pred_should_change", False)
    value.setdefault("pred_evidence_ids", [])
    return value


@torch.no_grad()
def predict_scores(
    rows: list[dict[str, Any]], checkpoint: Path, model_name: str,
    device: str, batch_size: int, max_length: int,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    plain_dual = bool(state.get("ablation", {}).get("plain_dual", False))
    model = RICAAdapter(model_name, condition_feature_dim=20, plain_dual=plain_dual,
                        selective=state.get("selective", False),
                        joint_decision="decision_head.weight" in state["state_dict"]).to(device)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError(f"Invalid adapter checkpoint: {missing=}, {unexpected=}")
    model.eval()
    no_condition_features = bool(state.get("ablation", {}).get("no_condition_features", False))
    allowed = ("id", "old_document", "new_document", "query", "condition", "revision")
    eligible = [{key: row[key] for key in allowed if key in row} for row in rows]
    predictions = {
        row["id"]: {
            "id": row["id"], "pred_old_label": "unknown", "pred_new_label": "unknown",
            "pred_should_change": False, "pred_evidence_ids": [],
            "impact_confidence": 0.0,
        }
        for row in rows
    }
    # Inference must not construct or consume any gold label fields, including
    # rows marked unsupported. The evaluator masks unsupported metrics later.
    collate = make_collator(tokenizer, max_length, include_targets=False)
    for start in range(0, len(eligible), batch_size):
        batch = move_batch(collate(eligible[start:start + batch_size]), torch.device(device))
        if no_condition_features:
            batch["condition_features"] = torch.zeros_like(batch["condition_features"])
        output = model(
            batch["old"]["input_ids"], batch["old"]["attention_mask"],
            batch["new"]["input_ids"], batch["new"]["attention_mask"],
            batch["condition_features"],
        )
        old_prob = output["old_logits"].softmax(-1)
        new_prob = output["new_logits"].softmax(-1)
        impact_prob = output["impact_logits"].softmax(-1)[:, 1]
        decision_prob = (output["decision_logits"].softmax(-1)
                         if "decision_logits" in output else None)
        for index, row in enumerate(batch["rows"]):
            predictions[row["id"]].update({
                "pred_old_label": list(LABELS)[int(old_prob[index].argmax())],
                "pred_new_label": list(LABELS)[int(new_prob[index].argmax())],
                "impact_confidence": float(impact_prob[index]),
                "confidence": float(min(old_prob[index].max(), new_prob[index].max())),
                "abstain_confidence": (float(output["abstain_logits"][index].softmax(-1)[1])
                                       if "abstain_logits" in output else 0.0),
                "decision_confidence": (float(decision_prob[index].max())
                                         if decision_prob is not None else 0.0),
                "decision_probabilities": (decision_prob[index].tolist() if decision_prob is not None else None),
                "pred_decision_head": (["change", "keep", "abstain"][int(decision_prob[index].argmax())]
                                        if decision_prob is not None else None),
                "pred_action": "human_review" if float(min(old_prob[index].max(), new_prob[index].max())) < 0.55 else "verify",
            })
    return [predictions[row["id"]] for row in rows]


def _threshold_predictions(predictions: list[dict[str, Any]], threshold: float):
    return [
        {**prediction, "pred_should_change": bool(prediction.get("impact_confidence", 0.0) >= threshold)}
        for prediction in predictions
    ]


def _groups(rows, predictions, key: str):
    values = sorted({str(row.get("metadata", {}).get(key, row.get(key, "unknown"))) for row in rows})
    output = {}
    for value in values:
        selected = [i for i, row in enumerate(rows)
                    if str(row.get("metadata", {}).get(key, row.get(key, "unknown"))) == value]
        output[value] = evaluate_records([rows[i] for i in selected], [predictions[i] for i in selected])
    return output


def _run(rows, raw, threshold):
    predictions = _threshold_predictions(raw, threshold)
    return {
        "threshold": threshold,
        "overall": evaluate_records(rows, predictions),
        "by_category": _groups(rows, predictions, "category"),
        "by_source_type": _groups(rows, predictions, "source_type"),
        "by_domain": _groups(rows, predictions, "domain_template"),
        "by_condition_status": _groups(rows, predictions, "condition_status"),
        "predictions": predictions,
    }


def _select(dev_rows, dev_raw, thresholds):
    baseline = [_complete(row, text_diff_prediction(row)) for row in dev_rows]
    baseline_recall = evaluate_records(dev_rows, baseline)["changed_accuracy"]
    candidates = []
    for threshold in thresholds:
        result = _run(dev_rows, dev_raw, threshold)
        overall = result["overall"]
        candidates.append({
            "threshold": threshold,
            "changed_recall": overall["changed_accuracy"],
            "invariant_accuracy": overall["invariant_accuracy"],
            "impact_f1": overall["impact"]["f1"],
            "eligible": overall["changed_accuracy"] >= baseline_recall - 0.05,
        })
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        selected = max(candidates, key=lambda item: (item["changed_recall"], item["invariant_accuracy"], item["threshold"]))
        reason = "no threshold met the five-point changed-recall constraint"
    else:
        selected = max(eligible, key=lambda item: (item["invariant_accuracy"], item["threshold"], item["impact_f1"]))
        reason = "max invariant accuracy subject to changed-recall constraint"
    return {"baseline_changed_recall": baseline_recall, "candidates": candidates,
            "selected": selected, "selection_reason": reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--checkpoint", default="outputs/rica.pt")
    parser.add_argument("--model", default=os.environ.get("REVIMPACT_MODEL_PATH", "/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output", default="outputs/rica_threshold_sweep_dev.json")
    parser.add_argument("--test-output", default="outputs/rica_threshold_test.json")
    parser.add_argument("--selected-output", default="outputs/rica_threshold_selected.json")
    parser.add_argument("--predictions-output", default="outputs/rica_threshold_predictions.jsonl")
    args = parser.parse_args()
    root = Path(args.data)
    splits = {split: list(load_jsonl(root / f"{split}.jsonl")) for split in ("dev", "test")}
    raw = {"dev": predict_scores(splits["dev"], Path(args.checkpoint), args.model, args.device,
                                 args.batch_size, args.max_length)}
    thresholds = [round(value, 2) for value in [0.10 + 0.05 * i for i in range(17)]]
    selection = _select(splits["dev"], raw["dev"], thresholds)
    selected_threshold = float(selection["selected"]["threshold"])
    # Commit the dev choice before computing any test predictions.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selected_output).write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw["test"] = predict_scores(splits["test"], Path(args.checkpoint), args.model, args.device,
                                  args.batch_size, args.max_length)
    dev_result = {"data": str(root), "thresholds": thresholds, "selection": selection,
                  "results": [{key: value for key, value in _run(splits["dev"], raw["dev"], threshold).items()
                               if key != "predictions"} for threshold in thresholds]}
    test_result = _run(splits["test"], raw["test"], selected_threshold)
    test_result["data"] = str(root)
    test_result["selection_source"] = "dev"
    test_result["selected_threshold"] = selected_threshold
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dev_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.selected_output).write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.test_output).write_text(json.dumps({key: value for key, value in test_result.items() if key != "predictions"},
                                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with Path(args.predictions_output).open("w", encoding="utf-8") as handle:
        for prediction in test_result["predictions"]:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    for split, predictions in raw.items():
        with output.with_name(output.stem + f"_{split}_scores.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_threshold": selected_threshold,
                      "dev": selection["selected"], "test": test_result["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
