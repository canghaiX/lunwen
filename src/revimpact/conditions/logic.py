"""Deterministic condition and interval operations used by RICE."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from revimpact.data.schema import Condition


class ConditionRelation(str, Enum):
    MATCH = "match"
    DISJOINT = "disjoint"
    OVERLAP = "overlap"
    UNKNOWN = "unknown"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


def as_interval(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping):
        lo = _number(value.get("min", value.get("low", float("-inf"))))
        hi = _number(value.get("max", value.get("high", float("inf"))))
        if lo is None and "min" not in value and "low" not in value:
            lo = float("-inf")
        if hi is None and "max" not in value and "high" not in value:
            hi = float("inf")
        if lo is not None and hi is not None:
            return lo, hi
        return None
    number = _number(value)
    return None if number is None else (number, number)


def interval_relation(left: Any, right: Any) -> ConditionRelation:
    """Return the relation between a query value and a revision value."""
    a, b = as_interval(left), as_interval(right)
    if a is None or b is None:
        if left == right:
            return ConditionRelation.MATCH
        # Two explicit categorical values are mutually exclusive. Unknown is
        # reserved for missing or non-comparable values, handled by the caller.
        if left is not None and right is not None:
            return ConditionRelation.DISJOINT
        return ConditionRelation.UNKNOWN
    if a[1] < b[0] or b[1] < a[0]:
        return ConditionRelation.DISJOINT
    if a == b:
        return ConditionRelation.MATCH
    return ConditionRelation.OVERLAP


def normalize_condition(value: Condition | Mapping[str, Any] | None) -> Condition:
    if value is None:
        return Condition()
    if isinstance(value, Condition):
        return value
    slots = value.get("slots", value)
    return Condition(
        slots=dict(slots),
        valid_from=value.get("valid_from"),
        valid_to=value.get("valid_to"),
    )


def condition_overlap(query: Condition | Mapping[str, Any], revision: Condition | Mapping[str, Any]) -> ConditionRelation:
    """Compare all shared condition slots.

    A missing slot is unknown rather than disjoint. This conservative behavior
    avoids silently declaring an affected knowledge item safe.
    """
    q, r = normalize_condition(query), normalize_condition(revision)
    relations = []
    for key, revision_value in r.slots.items():
        if key not in q.slots:
            relations.append(ConditionRelation.UNKNOWN)
        else:
            relations.append(interval_relation(q.slots[key], revision_value))
    if not relations:
        return ConditionRelation.MATCH
    if any(item == ConditionRelation.DISJOINT for item in relations):
        return ConditionRelation.DISJOINT
    if all(item == ConditionRelation.MATCH for item in relations):
        return ConditionRelation.MATCH
    if any(item == ConditionRelation.OVERLAP for item in relations):
        return ConditionRelation.OVERLAP
    return ConditionRelation.UNKNOWN
