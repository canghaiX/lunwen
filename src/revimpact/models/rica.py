"""Revision- and Invariance-Conditioned Adapter (RICA).

The language model is used as a frozen encoder.  All trainable parameters are
in the revision fusion and task heads, which keeps the first feasibility run
small enough for a single local GPU and makes the contribution auditable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LABEL_COUNT = 3
RELATION_COUNT = 4
DECISION_COUNT = 3  # change, keep, abstain


def boundary_rank_loss(
    impact_logits: Any,
    boundary_groups: Sequence[str | None],
    target_change: Any,
    margin: float = 0.2,
    target_scorable: Any | None = None,
):
    """Rank affected examples above unaffected examples in each family."""
    import torch
    import torch.nn.functional as F

    scores = impact_logits.softmax(-1)[:, 1]
    losses = []
    for group in sorted({item for item in boundary_groups if item}):
        indices = [i for i, item in enumerate(boundary_groups) if item == group]
        changed = [i for i in indices if bool(target_change[i].item()) and
                   (target_scorable is None or bool(target_scorable[i].item()))]
        invariant = [i for i in indices if not bool(target_change[i].item()) and
                     (target_scorable is None or bool(target_scorable[i].item()))]
        for positive in changed:
            for negative in invariant:
                losses.append(F.relu(margin - scores[positive] + scores[negative]))
    if not losses:
        return impact_logits.sum() * 0.0
    return torch.stack(losses).mean()


def rica_loss(
    outputs: Mapping[str, Any],
    target_old: Any,
    target_new: Any,
    target_change: Any,
    *,
    target_relation: Any | None = None,
    old_evidence_mask: Any | None = None,
    new_evidence_mask: Any | None = None,
    boundary_groups: Sequence[str | None] | None = None,
    lambda_impact: float = 1.0,
    lambda_consistency: float = 0.5,
    lambda_boundary: float = 0.5,
    lambda_evidence: float = 0.25,
    condition_gate: Any | None = None,
    target_scorable: Any | None = None,
    target_abstain: Any | None = None,
    target_decision: Any | None = None,
    lambda_abstain: float = 0.5,
    lambda_decision: float = 1.0,
    selective: bool = False,
    decision_class_weight: Any | None = None,
):
    """RICA loss with optional abstention supervision.

    Unsupported rows can train the abstention head, while their placeholder
    entailment/change labels are excluded from the corresponding losses.
    """
    import torch
    import torch.nn.functional as F

    old_logits = outputs["old_logits"]
    new_logits = outputs["new_logits"]
    impact_logits = outputs["impact_logits"]
    scorable = (target_scorable.bool() if target_scorable is not None
                else torch.ones_like(target_old, dtype=torch.bool))
    if scorable.any():
        loss_old = F.cross_entropy(old_logits[scorable], target_old[scorable])
        loss_new = F.cross_entropy(new_logits[scorable], target_new[scorable])
    else:
        loss_old = old_logits.sum() * 0.0
        loss_new = new_logits.sum() * 0.0
    # The pilot intentionally contains many changed examples (natural
    # VitaminC is heavily skewed), so weight the invariant class to prevent a
    # degenerate always-update gate.
    impact_weight = torch.tensor([1.6, 1.0], device=impact_logits.device, dtype=impact_logits.dtype)
    loss_impact = (F.cross_entropy(impact_logits[scorable], target_change[scorable], weight=impact_weight)
                   if scorable.any() else impact_logits.sum() * 0.0)
    total = loss_old + loss_new + lambda_impact * loss_impact
    components = {
        "old": loss_old.detach(), "new": loss_new.detach(), "impact": loss_impact.detach()
    }

    abstention = old_logits.sum() * 0.0
    if target_abstain is not None and "abstain_logits" in outputs:
        abstention = F.cross_entropy(outputs["abstain_logits"], target_abstain)
        total = total + lambda_abstain * abstention
    components["abstention"] = abstention.detach()

    decision = old_logits.sum() * 0.0
    if target_decision is not None and "decision_logits" in outputs:
        # Explicit tri-state supervision is the primary selective objective;
        # unsupported rows map to abstain and are not collapsed into keep.
        decision_weight = (decision_class_weight.to(device=old_logits.device,
                                                     dtype=old_logits.dtype)
                           if decision_class_weight is not None else
                           torch.ones(DECISION_COUNT, device=old_logits.device,
                                      dtype=old_logits.dtype))
        decision = F.cross_entropy(outputs["decision_logits"], target_decision,
                                   weight=decision_weight)
        total = total + lambda_decision * decision
    components["decision"] = decision.detach()

    # Conditions that are disjoint from the revision are required to preserve
    # the old distribution. On affected changed examples, the distributions
    # should move enough to avoid a trivial always-invariant solution.
    old_prob = old_logits.softmax(-1)
    new_prob = new_logits.softmax(-1)
    gate = condition_gate if condition_gate is not None else (target_change > 0)
    gate = gate.bool() & scorable
    invariant = ~gate.bool() & scorable
    if selective:
        invariant = (target_change == 0) & scorable
    consistency = old_logits.sum() * 0.0
    if invariant.any():
        if selective:
            midpoint = (old_prob[invariant] + new_prob[invariant]) / 2
            consistency = (F.kl_div(midpoint.clamp_min(1e-8).log(), old_prob[invariant], reduction="batchmean")
                           + F.kl_div(midpoint.clamp_min(1e-8).log(), new_prob[invariant], reduction="batchmean")) / 2
        else:
            consistency = F.kl_div(new_prob[invariant].clamp_min(1e-8).log(), old_prob[invariant].detach(), reduction="batchmean")
    affected = gate.bool() & (target_old != target_new)
    if affected.any():
        distribution_distance = (old_prob[affected] - new_prob[affected]).abs().sum(-1) / 2.0
        consistency = consistency + F.relu(0.15 - distribution_distance).mean()
    total = total + lambda_consistency * consistency
    components["consistency"] = consistency.detach()

    if target_relation is not None:
        relation = F.cross_entropy(outputs["relation_logits"], target_relation)
        total = total + 0.25 * relation
        components["relation"] = relation.detach()

    boundary = old_logits.sum() * 0.0
    if boundary_groups is not None:
        boundary = boundary_rank_loss(impact_logits, boundary_groups, target_change, target_scorable=scorable)
        total = total + lambda_boundary * boundary
    components["boundary"] = boundary.detach()

    evidence = old_logits.sum() * 0.0
    evidence_terms = []
    for key, target in (("old_evidence_logits", old_evidence_mask), ("new_evidence_logits", new_evidence_mask)):
        if target is None or key not in outputs:
            continue
        logits = outputs[key]
        valid = (target >= 0) & scorable.unsqueeze(-1)
        if valid.any():
            evidence_terms.append(F.binary_cross_entropy_with_logits(logits[valid], target[valid].float()))
    if evidence_terms:
        evidence = torch.stack(evidence_terms).mean()
        total = total + lambda_evidence * evidence
    components["evidence"] = evidence.detach()
    components["total"] = total.detach()
    return total, components


class RICAAdapter:
    """Frozen Transformer encoder plus revision-aware trainable heads."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _RICA(nn.Module):
            def __init__(self, model_name: str, condition_feature_dim: int = 20, dropout: float = 0.1,
                         plain_dual: bool = False, selective: bool = False,
                         joint_decision: bool = True):
                super().__init__()
                from transformers import AutoModel

                self.model_name = model_name
                self.plain_dual = plain_dual
                self.selective = selective
                self.joint_decision = selective and joint_decision
                self.encoder = AutoModel.from_pretrained(
                    model_name, local_files_only=True, trust_remote_code=True
                )
                hidden = int(getattr(self.encoder.config, "hidden_size", 768))
                self.hidden_size = hidden
                self.encoder.requires_grad_(False)
                self.encoder.eval()
                self.condition_projection = nn.Sequential(
                    nn.Linear(condition_feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
                )
                self.fusion = nn.Sequential(
                    nn.Linear(hidden * (2 if plain_dual else 5), hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden)
                )
                self.old_head = nn.Linear(hidden, LABEL_COUNT)
                self.new_head = nn.Linear(hidden, LABEL_COUNT)
                self.impact_head = nn.Linear(hidden, 2)
                if selective:
                    self.abstain_head = nn.Linear(hidden, 2)
                if self.joint_decision:
                    self.decision_head = nn.Linear(hidden, DECISION_COUNT)
                self.relation_head = nn.Linear(hidden, RELATION_COUNT)
                self.old_evidence_head = nn.Linear(hidden, 1)
                self.new_evidence_head = nn.Linear(hidden, 1)

            def train(self, mode: bool = True):
                super().train(mode)
                self.encoder.eval()  # the encoder remains frozen in every phase
                return self

            def _encode(self, input_ids, attention_mask):
                with __import__("torch").no_grad():
                    result = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    tokens = result.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
                    pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                return pooled.float(), tokens.float()

            def forward(
                self,
                old_input_ids,
                old_attention_mask,
                new_input_ids,
                new_attention_mask,
                condition_features,
            ):
                import torch

                old_pooled, old_tokens = self._encode(old_input_ids, old_attention_mask)
                new_pooled, new_tokens = self._encode(new_input_ids, new_attention_mask)
                if self.plain_dual:
                    fused = self.fusion(torch.cat([old_pooled, new_pooled], dim=-1))
                else:
                    condition = self.condition_projection(condition_features.float())
                    fused = self.fusion(torch.cat([
                        old_pooled, new_pooled, new_pooled - old_pooled,
                        (new_pooled - old_pooled).abs() if self.selective else old_pooled * new_pooled, condition,
                    ], dim=-1))
                outputs = {
                    "old_logits": self.old_head(old_pooled if self.selective else fused),
                    "new_logits": self.new_head(new_pooled if self.selective else fused),
                    "impact_logits": self.impact_head(fused),
                    "relation_logits": self.relation_head(fused),
                    "old_evidence_logits": self.old_evidence_head(old_tokens).squeeze(-1),
                    "new_evidence_logits": self.new_evidence_head(new_tokens).squeeze(-1),
                    "old_tokens": old_tokens,
                    "new_tokens": new_tokens,
                }
                if self.selective:
                    outputs["abstain_logits"] = self.abstain_head(fused)
                if self.joint_decision:
                    outputs["decision_logits"] = self.decision_head(fused)
                return outputs

        instance = object.__new__(_RICA)
        _RICA.__init__(instance, *args, **kwargs)
        return instance


# Explicit alias makes checkpoint metadata and imports stable if the class is
# later moved from an adapter implementation to a standard nn.Module class.
RICAModel = RICAAdapter
