"""RICE decision layer and optional differentiable multi-task objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from revimpact.conditions.logic import ConditionRelation, condition_overlap
from revimpact.data.schema import Action, Conclusion, Condition, Revision


@dataclass
class RICEOutput:
    old_label: Conclusion
    new_label: Conclusion
    should_change: bool
    relation: ConditionRelation
    action: Action
    confidence: float


def decide_maintenance(
    old_label: Conclusion,
    new_label: Conclusion,
    query_condition: Condition,
    revision: Revision,
    *,
    evidence_sufficient: bool = True,
    confidence: float = 1.0,
    threshold: float = 0.7,
) -> RICEOutput:
    relation = condition_overlap(query_condition, revision.condition)
    should_change = old_label != new_label and relation != ConditionRelation.DISJOINT
    if confidence < threshold or relation == ConditionRelation.UNKNOWN or not evidence_sufficient:
        action = Action.HUMAN_REVIEW
    elif should_change:
        action = Action.UPDATE
    elif revision.revision_type not in {"unchanged", "unknown"}:
        action = Action.VERIFY
    else:
        action = Action.KEEP
    return RICEOutput(old_label, new_label, should_change, relation, action, confidence)


def rice_loss(
    old_logits,
    new_logits,
    change_logits,
    target_old,
    target_new,
    target_change,
    *,
    invariant_mask=None,
    lambda_change: float = 1.0,
    lambda_invariance: float = 1.0,
):
    """Compute the core RICE loss for torch tensors.

    Evidence and condition heads can be added by callers as extra terms. The
    function is kept independent of a particular encoder so frozen LLM and
    cross-encoder experiments share the same objective.
    """
    import torch.nn.functional as F

    loss = F.cross_entropy(old_logits, target_old) + F.cross_entropy(new_logits, target_new)
    loss = loss + lambda_change * F.cross_entropy(change_logits, target_change)
    if invariant_mask is not None and invariant_mask.any():
        old_prob = old_logits[invariant_mask].softmax(-1)
        new_prob = new_logits[invariant_mask].softmax(-1)
        loss = loss + lambda_invariance * F.kl_div(new_prob.log(), old_prob.detach(), reduction="batchmean")
    return loss


class RICEModel:
    """Factory for the trainable RICE multi-task model.

    The class is a lazy wrapper so the dependency-light rule pilot remains
    usable without importing or loading Transformer weights. The returned
    module shares one encoder and has old/new entailment heads, a change head,
    and a condition-overlap head.
    """

    @staticmethod
    def from_pretrained(model_name: str, condition_feature_dim: int = 8):
        import torch
        from torch import nn
        from transformers import AutoModel

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(
                    model_name, local_files_only=True, trust_remote_code=True,
                    device_map="auto", torch_dtype="auto"
                )
                hidden = getattr(self.encoder.config, "hidden_size", None)
                if hidden is None:
                    hidden = getattr(self.encoder.config, "hidden_sizes", [768])[-1]
                self.condition = nn.Sequential(nn.Linear(condition_feature_dim, hidden), nn.GELU())
                self.old_head = nn.Linear(hidden, 3)
                self.new_head = nn.Linear(hidden, 3)
                self.change_head = nn.Linear(hidden, 2)
                self.relation_head = nn.Linear(hidden, 4)

            def forward(self, input_ids, attention_mask=None, condition_features=None):
                encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = getattr(encoded, "last_hidden_state", encoded[0])[:, 0]
                if condition_features is not None:
                    condition_device = next(self.condition.parameters()).device
                    condition = self.condition(condition_features.to(condition_device).float()).to(hidden.device, dtype=hidden.dtype)
                    hidden = hidden + condition
                head_device = next(self.old_head.parameters()).device
                head_hidden = hidden.to(head_device).float()
                return {
                    "old_logits": self.old_head(head_hidden),
                    "new_logits": self.new_head(head_hidden),
                    "change_logits": self.change_head(head_hidden),
                    "relation_logits": self.relation_head(head_hidden),
                }

        return _Module()
