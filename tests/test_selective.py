import torch
import pytest

from revimpact.evaluation.metrics import evaluate_records
from revimpact.evaluation.selective import apply_policy, policy_grid, select_policy
from revimpact.models.rica import rica_loss


def test_selective_loss_excludes_unsupported_from_main_heads():
    outputs = {
        "old_logits": torch.randn(3, 3, requires_grad=True),
        "new_logits": torch.randn(3, 3, requires_grad=True),
        "impact_logits": torch.randn(3, 2, requires_grad=True),
        "abstain_logits": torch.randn(3, 2, requires_grad=True),
        "decision_logits": torch.randn(3, 3, requires_grad=True),
        "relation_logits": torch.randn(3, 4, requires_grad=True),
        "old_evidence_logits": torch.randn(3, 4, requires_grad=True),
        "new_evidence_logits": torch.randn(3, 4, requires_grad=True),
    }
    mask = torch.tensor([[1, 0, -1, -1]] * 3, dtype=torch.float32)
    loss, parts = rica_loss(
        outputs, torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]), torch.tensor([0, 1, 0]),
        target_relation=torch.tensor([0, 1, 2]), old_evidence_mask=mask, new_evidence_mask=mask,
        boundary_groups=["g", "g", None], target_scorable=torch.tensor([True, True, False]),
        target_abstain=torch.tensor([0, 0, 1]), selective=True,
        target_decision=torch.tensor([0, 1, 2]),
        decision_class_weight=torch.tensor([0.6, 1.2, 2.4]),
    )
    assert torch.isfinite(loss)
    assert parts["abstention"].item() > 0
    assert parts["decision"].item() > 0
    loss.backward()
    assert torch.equal(outputs["old_logits"].grad[2], torch.zeros(3))
    assert torch.equal(outputs["new_logits"].grad[2], torch.zeros(3))
    assert torch.equal(outputs["impact_logits"].grad[2], torch.zeros(2))
    assert outputs["decision_logits"].grad[2].abs().sum() > 0


def test_selective_metrics_report_coverage_and_risk():
    rows = [
        {"id": "a", "old_label": "entailed", "new_label": "contradicted", "should_change": True,
         "metadata": {"scorable": True}},
        {"id": "b", "old_label": "unknown", "new_label": "unknown", "should_change": False,
         "metadata": {"scorable": False}},
    ]
    result = evaluate_records(rows, [
        {"id": "a", "pred_decision": "change", "pred_should_change": True},
        {"id": "b", "pred_decision": "abstain", "pred_should_change": False},
    ])
    assert result["selective"]["coverage"] == 0.5
    assert result["selective"]["risk"] == 0.0


def test_abstaining_is_not_a_correct_keep():
    row = {"id": "keep", "old_label": "entailed", "new_label": "entailed",
           "should_change": False, "metadata": {"scorable": True}}
    result = evaluate_records([row], [{"id": "keep", "pred_decision": "abstain",
                                       "pred_should_change": False}])
    assert result["invariant_accuracy"] == 0
    assert result["false_update_rate"] == 0
    assert result["selective"]["unchanged_abstention_rate"] == 1


def test_decision_head_policies_and_missing_head():
    scores = [{"id": "x", "decision_probabilities": [0.6, 0.3, 0.1]}]
    assert apply_policy(scores, {"mode": "head"})[0]["pred_decision"] == "change"
    assert apply_policy(scores, {"mode": "calibrated_head", "keep_bias": 1.0})[0]["pred_decision"] == "keep"
    with pytest.raises(ValueError, match="trained three-class"):
        apply_policy([{"id": "x"}], {"mode": "head"})
    assert len(policy_grid("head")) == 1


def test_dev_selection_does_not_sacrifice_recall_when_infeasible():
    rows = [{"id": str(i), "old_label": "entailed", "new_label": "contradicted",
             "should_change": True} for i in range(2)]
    scores = [{"id": "0", "impact_confidence": 0.7, "abstain_confidence": 0.0},
              {"id": "1", "impact_confidence": 0.3, "abstain_confidence": 0.0}]
    policies = [{"mode": "threshold", "impact_threshold": value, "abstain_threshold": 0.5}
                for value in (0.5, 0.8)]
    selection = select_policy(rows, scores, policies, minimum_changed_recall=0.95)
    assert not selection["constraint_satisfied"]
    assert selection["selected"]["changed_recall"] == 0.5


def test_inference_collator_ignores_all_gold_fields():
    from scripts.train_rica import make_collator

    class TinyTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
                    "offset_mapping": torch.zeros(len(texts), 4, 2, dtype=torch.long)}

    class InputOnly(dict):
        def get(self, key, default=None):
            if key not in {"id", "old_document", "new_document", "query", "condition", "revision"}:
                raise AssertionError(f"read target: {key}")
            return super().get(key, default)

    row = InputOnly(id="x", old_document="old", new_document="new", query="q", condition={})
    batch = make_collator(TinyTokenizer(), 4, include_targets=False)([row])
    assert not any(key.startswith("target_") for key in batch)
