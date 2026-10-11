"""Prediction baselines used by the pilot evaluator."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from revimpact.conditions.logic import ConditionRelation, condition_overlap
from revimpact.data.schema import Conclusion, Condition, Revision


def _threshold_label(condition: Condition, revision: Revision, value: Any) -> str:
    relation = condition_overlap(condition, revision.condition)
    if relation == ConditionRelation.DISJOINT:
        return Conclusion.UNKNOWN.value
    try:
        pressure = float(value)
        limit = float(revision.new_value)
    except (TypeError, ValueError):
        return Conclusion.UNKNOWN.value
    return (Conclusion.ENTAILED if pressure <= limit else Conclusion.CONTRADICTED).value


def condition_rule_prediction(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Predict from structured revision/condition fields without gold labels."""
    condition = Condition.model_validate(row.get("condition", {}))
    revision = Revision.model_validate(row["revision"])
    relation = condition_overlap(condition, revision.condition)
    value = next(
        (condition.slots[key] for key in ("value", "pressure", "timeout", "hours", "exposure", "limit") if key in condition.slots),
        None,
    )
    try:
        old_limit = float(revision.old_value)
        pressure = float(value)
        old_label = (Conclusion.ENTAILED if pressure <= old_limit else Conclusion.CONTRADICTED).value \
            if relation != ConditionRelation.DISJOINT else Conclusion.UNKNOWN.value
    except (TypeError, ValueError):
        old_label = Conclusion.UNKNOWN.value
    new_label = _threshold_label(condition, revision, value)
    return {
        "id": row["id"],
        "pred_old_label": old_label,
        "pred_new_label": new_label,
        "pred_should_change": old_label != new_label,
        "pred_evidence_ids": _evidence_ids(row),
    }


def _evidence_ids(row: Mapping[str, Any]) -> list[str]:
    """Select the available evidence spans without consulting their labels."""
    ids = []
    for key in ("old_evidence", "new_evidence"):
        for index, span in enumerate(row.get(key, []) or []):
            source_id = span.get("source_id") if isinstance(span, Mapping) else None
            ids.append(str(source_id or f"{key}:{index}"))
    return ids


def text_diff_prediction(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Weak baseline: any text change is treated as a changed conclusion."""
    changed = row["old_document"] != row["new_document"]
    return {"id": row["id"], "pred_should_change": changed, "pred_evidence_ids": _evidence_ids(row)}


def evidence_overlap_prediction(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Lexical evidence baseline with no access to gold labels."""
    claim_tokens = {token.lower() for token in row.get("query", "").split() if len(token) > 2}

    def score(text: str) -> float:
        tokens = {token.lower() for token in text.split()}
        return len(claim_tokens & tokens) / max(len(claim_tokens), 1)

    old_score = score(row.get("old_document", ""))
    new_score = score(row.get("new_document", ""))
    changed = abs(old_score - new_score) > 0.05
    return {
        "id": row["id"],
        "pred_should_change": changed,
        "pred_evidence_ids": _evidence_ids(row),
        "confidence": abs(old_score - new_score),
    }
