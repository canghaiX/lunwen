from revimpact.data.schema import Action, Conclusion, Condition, Revision
from revimpact.models.rice import decide_maintenance


def test_changed_case_updates():
    output = decide_maintenance(
        Conclusion.ENTAILED, Conclusion.CONTRADICTED,
        Condition(slots={"model": "A", "environment": "high_temperature", "pressure": 9}),
        Revision(predicate="limit", old_value=10, new_value=8,
                 condition=Condition(slots={"model": "A", "environment": "high_temperature"}),
                 revision_type="numeric_change"),
    )
    assert output.should_change is True
    assert output.action == Action.UPDATE


def test_unaffected_case_keeps_answer():
    output = decide_maintenance(
        Conclusion.ENTAILED, Conclusion.ENTAILED,
        Condition(slots={"model": "B", "environment": "high_temperature"}),
        Revision(predicate="limit", old_value=10, new_value=8,
                 condition=Condition(slots={"model": "A", "environment": "high_temperature"}),
                 revision_type="numeric_change"),
    )
    assert output.should_change is False
    assert output.action == Action.VERIFY
