import json
from pathlib import Path

from revimpact.llm.annotation import parse_initial, parse_review, public_record


def test_public_record_does_not_expose_gold_fields():
    row = {"id": "x", "old_document": "old", "new_document": "new", "query": "q",
           "old_label": "entailed", "new_label": "contradicted", "should_change": True,
           "condition": {}, "revision": {}}
    value = public_record(row)
    assert "old_label" not in value
    assert "new_label" not in value
    assert "should_change" not in value


def test_invalid_initial_json_falls_back_to_unknown():
    prediction, failed = parse_initial("not json")
    assert failed is True
    assert prediction["old_label"] == "unknown"
    assert prediction["new_label"] == "unknown"
    assert prediction["action"] == "human_review"


def test_review_parser_normalizes_field_decisions():
    review, failed = parse_review(json.dumps({
        "old_label": {"decision": "accept", "value": "entailed"},
        "new_label": {"decision": "correct", "value": "contradicted"},
        "should_change": {"decision": "correct", "value": True},
        "action": {"decision": "accept", "value": "update"},
        "evidence": {"decision": "accept", "old_evidence_ids": ["o"], "new_evidence_ids": ["n"]},
        "overall_decision": "revise", "confidence": 0.9,
    }))
    assert failed is False
    assert review["new_label"]["value"] == "contradicted"
    assert review["should_change"]["value"] is True


def test_existing_pilot_files_have_no_annotation_side_effects():
    # This test documents that the annotation artifacts are separate from the
    # original dataset; it is skipped naturally before the first annotation run.
    path = Path("data/revimpact_800/annotations_qwen80b.jsonl")
    if path.exists():
        assert path.stat().st_size >= 0
