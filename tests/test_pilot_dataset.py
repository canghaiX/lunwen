import json

from scripts.build_pilot_dataset import build_pilot
from scripts.evaluate_pilot import _diagnostics
from revimpact.evaluation.baselines import text_diff_prediction, condition_rule_prediction
from revimpact.evaluation.metrics import evaluate_records, load_jsonl


def test_balanced_pilot_has_family_split_and_categories(tmp_path):
    output = tmp_path / "pilot"
    summary = build_pilot(output, families=12, queries_per_family=8, seed=3)
    assert summary["total_queries"] == 96
    assert set(summary["categories"]) == {"changed", "invariant", "boundary", "unsupported"}
    rows = {split: list(load_jsonl(output / f"{split}.jsonl")) for split in ("train", "dev", "test")}
    assert not ({row["family_id"] for row in rows["train"]} & {row["family_id"] for row in rows["dev"]})
    assert not ({row["family_id"] for row in rows["train"]} & {row["family_id"] for row in rows["test"]})
    assert all(row["old_evidence"] and row["new_evidence"] for split in rows.values() for row in split)


def test_unsupported_is_excluded_from_primary_metrics(tmp_path):
    output = tmp_path / "pilot"
    build_pilot(output, families=12, queries_per_family=8, seed=3)
    rows = list(load_jsonl(output / "test.jsonl"))
    predictions = [condition_rule_prediction(row) for row in rows]
    result = evaluate_records(rows, predictions)
    assert result["unsupported_n"] > 0
    assert result["scorable_n"] < result["n"]
    assert result["impact"]["f1"] == 1.0
    assert result["unsupported_accuracy"] == 1.0


def test_diagnostics_distinguish_within_family_repeats():
    row = {
        "id": "x", "family_id": "f", "old_document": "old", "new_document": "new", "query": "q",
    }
    result = _diagnostics({"train": [row, {**row, "id": "y"}], "test": []})
    assert result["documents"]["train"]["within_family_repeated_pairs"] == 1
    assert result["documents"]["train"]["cross_family_duplicate_pairs"] == 0


def test_text_diff_does_not_read_gold():
    row = {"id": "x", "old_document": "same", "new_document": "same", "should_change": True}
    assert text_diff_prediction(row)["pred_should_change"] is False
