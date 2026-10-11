"""VitaminC to :class:`RevisionRecord` conversion.

VitaminC releases have appeared as JSON/JSONL/TSV/Parquet files and minor
release variants use different names for the same fields. This adapter keeps
the original row in metadata, applies conservative field discovery, and never
invent an old label when a row only contains the edited claim/evidence.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from revimpact.data.schema import Action, Conclusion, Condition, EvidenceSpan, Revision, RevisionRecord

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".tsv", ".parquet"}
FIELD_ALIASES = {
    "id": ("id", "uid", "unique_id", "example_id", "instance_id"),
    "claim": ("claim", "statement", "hypothesis", "text", "query"),
    "label": ("label", "gold_label", "verdict", "annotation"),
    "evidence": ("evidence", "context", "article", "document", "sentence"),
    "old_evidence": (
        "old_evidence", "original_evidence", "original_context", "pre_edit_evidence",
        "before_evidence", "evidence_old", "source_evidence",
    ),
    "new_evidence": (
        "new_evidence", "edited_evidence", "revised_evidence", "post_edit_evidence",
        "after_evidence", "evidence_new", "counterfactual_evidence",
    ),
    "old_label": ("old_label", "original_label", "pre_edit_label", "label_old"),
    "new_label": ("new_label", "edited_label", "revised_label", "post_edit_label", "label_new"),
    "family": ("family_id", "page_id", "article_id", "document_id", "title", "page"),
    "split": ("split", "subset", "partition"),
    "change": ("should_change", "changed", "label_changed", "is_changed", "revision_changed"),
    "revision_type": ("revision_type", "edit_type", "evidence_type", "change_type"),
}


def discover_files(path: str | Path) -> dict[str, list[Path]]:
    """Return supported files grouped by official split when detectable."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    groups: dict[str, list[Path]] = {}
    for file in files:
        lowered = file.stem.lower()
        split = next((candidate for candidate in ("train", "dev", "validation", "test") if re.search(rf"(^|[-_.]){candidate}($|[-_.])", lowered)), "all")
        groups.setdefault(split, []).append(file)
    return groups


def _rows_from_file(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    yield row if isinstance(row, dict) else {"value": row}
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            # Common release layout: {"data": [...]} or split-keyed JSON.
            value = value.get("data", value.get("examples", value))
            if isinstance(value, dict):
                # Preserve a split-keyed object as rows while keeping the
                # originating split available to the outer filename logic.
                if all(isinstance(item, list) for item in value.values()):
                    flattened = []
                    for key, items in value.items():
                        flattened.extend({**item, "_source_split": key} if isinstance(item, dict) else {"value": item, "_source_split": key} for item in items)
                    value = flattened
                else:
                    value = [value]
        for row in value:
            yield row if isinstance(row, dict) else {"value": row}
        return
    if suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t")
        return
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pandas/pyarrow in the container") from exc
        for row in pd.read_parquet(path).to_dict(orient="records"):
            yield row
        return
    raise ValueError(f"unsupported VitaminC file: {path}")


def _first(row: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
        if alias.lower() in lowered and lowered[alias.lower()] not in (None, ""):
            return lowered[alias.lower()]
    return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "sentence", "content", "evidence"):
            if key in value:
                return _text(value[key])
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _label(value: Any) -> Conclusion:
    if value is None:
        return Conclusion.UNKNOWN
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"supports", "support", "supported", "true", "entailment", "entailed", "yes"}:
        return Conclusion.ENTAILED
    if normalized in {"refutes", "refute", "refuted", "false", "contradiction", "contradicted", "no"}:
        return Conclusion.CONTRADICTED
    return Conclusion.UNKNOWN


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "changed", "change"}:
        return True
    if normalized in {"false", "0", "no", "unchanged", "same"}:
        return False
    return None


def _family(row: Mapping[str, Any], fallback: str) -> str:
    value = _first(row, FIELD_ALIASES["family"], fallback)
    return _text(value).strip() or fallback


def row_to_record(row: Mapping[str, Any], *, split: str = "all", source_file: str = "", index: int = 0) -> RevisionRecord:
    claim = _text(_first(row, FIELD_ALIASES["claim"], ""))
    old_document = _text(_first(row, FIELD_ALIASES["old_evidence"], ""))
    new_document = _text(_first(row, FIELD_ALIASES["new_evidence"], ""))
    shared_evidence = _text(_first(row, FIELD_ALIASES["evidence"], ""))
    if not old_document:
        old_document = shared_evidence
    if not new_document:
        new_document = shared_evidence
    old_value = _first(row, FIELD_ALIASES["old_label"])
    new_value = _first(row, FIELD_ALIASES["new_label"])
    raw_label = _first(row, FIELD_ALIASES["label"])
    old_label = _label(old_value) if old_value is not None else Conclusion.UNKNOWN
    new_label = _label(new_value) if new_value is not None else _label(raw_label)
    explicit_change = _bool(_first(row, FIELD_ALIASES["change"]))
    has_pair = bool(_first(row, FIELD_ALIASES["old_evidence"]) or _first(row, FIELD_ALIASES["new_evidence"]) or old_value is not None or new_value is not None)
    inferred_change = old_label != new_label if has_pair and old_label != Conclusion.UNKNOWN and new_label != Conclusion.UNKNOWN else None
    should_change = explicit_change if explicit_change is not None else bool(inferred_change)
    revision_type = _text(_first(row, FIELD_ALIASES["revision_type"], "evidence_edit")) or "evidence_edit"
    if old_document == new_document:
        revision_type = "unchanged"
    row_id = _text(_first(row, FIELD_ALIASES["id"], f"vitaminc-{split}-{index:07d}"))
    family_id = _family(row, row_id)
    action = Action.UPDATE if should_change else (Action.VERIFY if has_pair else Action.HUMAN_REVIEW)
    evidence_old = [EvidenceSpan(version="original", text=old_document, source_id=family_id)] if old_document else []
    evidence_new = [EvidenceSpan(version="edited", text=new_document, source_id=family_id)] if new_document else []
    return RevisionRecord(
        id=row_id, family_id=family_id, domain="wikipedia_fact_verification",
        old_version="original", new_version="edited", old_document=old_document, new_document=new_document,
        query=claim, condition=Condition(),
        revision=Revision(predicate="evidence", old_value=old_document, new_value=new_document,
                          condition=Condition(), revision_type=revision_type, source_id=family_id),
        old_label=old_label, new_label=new_label, should_change=should_change,
        old_evidence=evidence_old, new_evidence=evidence_new, action=action,
        revision_type=revision_type, split=split,
        metadata={"source": "vitaminc", "derived": True, "source_file": source_file, "raw": dict(row),
                  "has_explicit_old_new": has_pair, "explicit_change": explicit_change},
    )


def _pair_to_record(old_row: Mapping[str, Any], new_row: Mapping[str, Any], *, split: str, source_file: str, index: int) -> RevisionRecord:
    """Convert an adjacent VitaminC contrastive pair to one old/new record."""
    claim = _text(_first(old_row, FIELD_ALIASES["claim"], _first(new_row, FIELD_ALIASES["claim"], "")))
    old_document = _text(_first(old_row, FIELD_ALIASES["evidence"], ""))
    new_document = _text(_first(new_row, FIELD_ALIASES["evidence"], ""))
    old_label = _label(_first(old_row, FIELD_ALIASES["label"]))
    new_label = _label(_first(new_row, FIELD_ALIASES["label"]))
    case_id = _text(_first(old_row, ("case_id",), _first(old_row, FIELD_ALIASES["family"], f"case-{index}")))
    page = _text(_first(old_row, ("page", "title"), case_id))
    old_id = _text(_first(old_row, FIELD_ALIASES["id"], f"old-{index}"))
    new_id = _text(_first(new_row, FIELD_ALIASES["id"], f"new-{index}"))
    revision_type = _text(_first(old_row, FIELD_ALIASES["revision_type"], "evidence_edit")) or "evidence_edit"
    should_change = old_label != new_label
    return RevisionRecord(
        id=f"{case_id}:{old_id}->{new_id}", family_id=page, domain="wikipedia_fact_verification",
        old_version=old_id, new_version=new_id, old_document=old_document, new_document=new_document,
        query=claim, condition=Condition(),
        revision=Revision(predicate="evidence", old_value=old_document, new_value=new_document,
                          condition=Condition(), revision_type=revision_type, source_id=case_id),
        old_label=old_label, new_label=new_label, should_change=should_change,
        old_evidence=[EvidenceSpan(version="original", text=old_document, source_id=old_id)],
        new_evidence=[EvidenceSpan(version="edited", text=new_document, source_id=new_id)],
        action=Action.UPDATE if should_change else Action.VERIFY,
        revision_type=revision_type, split=split,
        metadata={"source": "vitaminc", "derived": True, "source_file": source_file,
                  "pairing": "case_id_claim_adjacent", "raw": {"old": dict(old_row), "new": dict(new_row)},
                  "case_id": case_id},
    )


def convert_file(path: str | Path, *, split: str = "all") -> list[RevisionRecord]:
    file_path = Path(path)
    rows = list(_rows_from_file(file_path))
    if rows and all("case_id" in row and "claim" in row and "evidence" in row for row in rows):
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((_text(row.get("case_id")), _text(row.get("claim"))), []).append(row)
        records: list[RevisionRecord] = []
        index = 0
        for group in groups.values():
            for offset in range(0, len(group) - 1, 2):
                records.append(_pair_to_record(group[offset], group[offset + 1], split=split, source_file=str(file_path), index=index))
                index += 1
            if len(group) % 2:
                records.append(row_to_record(group[-1], split=split, source_file=str(file_path), index=index))
                index += 1
        return records
    return [row_to_record(row, split=split, source_file=str(file_path), index=index) for index, row in enumerate(rows)]


def convert_path(path: str | Path) -> dict[str, list[RevisionRecord]]:
    groups = discover_files(path)
    output: dict[str, list[RevisionRecord]] = {}
    for split, files in groups.items():
        records: list[RevisionRecord] = []
        for file in files:
            records.extend(convert_file(file, split=split))
        output[split] = records
    return output
