import json

from revimpact.data.adapters.vitaminc import convert_file, row_to_record


def test_vitaminc_paired_row():
    row = {
        "id": "x1", "page_id": "Ada_Lovelace", "claim": "Ada was a mathematician",
        "original_evidence": "Ada studied mathematics.", "edited_evidence": "Ada studied mathematics and computing.",
        "original_label": "supports", "edited_label": "supports",
    }
    record = row_to_record(row, split="test")
    assert record.id == "x1"
    assert record.family_id == "Ada_Lovelace"
    assert record.old_label.value == "entailed"
    assert record.new_label.value == "entailed"
    assert record.should_change is False
    assert record.metadata["source"] == "vitaminc"


def test_vitaminc_single_label_is_conservative(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps({"uid": "x2", "claim": "The claim", "evidence": "Some evidence", "label": "REFUTES"}) + "\n", encoding="utf-8")
    record = convert_file(path, split="test")[0]
    assert record.old_label.value == "unknown"
    assert record.new_label.value == "contradicted"
    assert record.metadata["has_explicit_old_new"] is False


def test_vitaminc_tsv_aliases(tmp_path):
    path = tmp_path / "dev.tsv"
    path.write_text("statement\tcontext\tgold_label\nA claim\tAn article\tSUPPORTS\n", encoding="utf-8")
    record = convert_file(path, split="dev")[0]
    assert record.query == "A claim"
    assert record.new_label.value == "entailed"


def test_vitaminc_json_data_layout(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"data": [{"claim": "C", "evidence": "E", "label": "SUPPORTS"}]}), encoding="utf-8")
    record = convert_file(path, split="train")[0]
    assert record.query == "C"
    assert record.new_label.value == "entailed"


def test_vitaminc_case_pairs_are_old_new(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        {"unique_id": "c_1", "case_id": "c", "claim": "C", "evidence": "old", "label": "SUPPORTS", "page": "P", "revision_type": "real"},
        {"unique_id": "c_2", "case_id": "c", "claim": "C", "evidence": "new", "label": "REFUTES", "page": "P", "revision_type": "real"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    record = convert_file(path, split="test")[0]
    assert record.old_document == "old"
    assert record.new_document == "new"
    assert record.old_label.value == "entailed"
    assert record.new_label.value == "contradicted"
    assert record.should_change is True
    assert record.metadata["pairing"] == "case_id_claim_adjacent"
