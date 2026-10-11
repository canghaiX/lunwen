from revimpact.conditions.logic import ConditionRelation, condition_overlap, interval_relation
from revimpact.data.schema import Condition


def test_numeric_boundary_overlap():
    assert interval_relation(9, {"min": 8, "max": 10}) == ConditionRelation.OVERLAP
    assert interval_relation(7, {"min": 8, "max": 10}) == ConditionRelation.DISJOINT


def test_missing_slot_is_conservative():
    query = Condition(slots={"model": "A"})
    revision = Condition(slots={"model": "A", "environment": "high_temperature"})
    assert condition_overlap(query, revision) == ConditionRelation.UNKNOWN


def test_disjoint_model_does_not_propagate():
    query = Condition(slots={"model": "B", "environment": "high_temperature"})
    revision = Condition(slots={"model": "A", "environment": "high_temperature"})
    assert condition_overlap(query, revision) == ConditionRelation.DISJOINT
