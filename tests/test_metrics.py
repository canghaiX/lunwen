from revimpact.data.synthetic import generate
from revimpact.evaluation.metrics import evaluate_records
from revimpact.evaluation.baselines import condition_rule_prediction


def test_pilot_metrics_run():
    rows = [item.to_json_dict() for item in generate(8, seed=1)]
    result = evaluate_records(rows, [condition_rule_prediction(row) for row in rows])
    assert result["n"] == 8
    assert result["impact"]["f1"] == 1.0
    assert result["old_accuracy"] == 1.0
    assert result["new_accuracy"] == 1.0
    assert result["invariant_accuracy"] == 1.0
