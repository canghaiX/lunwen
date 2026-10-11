import json
from pathlib import Path

import torch

from revimpact.models.rica import boundary_rank_loss, rica_loss


def test_rica_loss_and_boundary_rank_are_finite():
    outputs = {
        "old_logits": torch.randn(4, 3, requires_grad=True),
        "new_logits": torch.randn(4, 3, requires_grad=True),
        "impact_logits": torch.randn(4, 2, requires_grad=True),
        "relation_logits": torch.randn(4, 4, requires_grad=True),
        "old_evidence_logits": torch.randn(4, 5, requires_grad=True),
        "new_evidence_logits": torch.randn(4, 5, requires_grad=True),
    }
    mask = torch.tensor([[1, 0, -1, -1, -1]] * 4, dtype=torch.float32)
    loss, parts = rica_loss(
        outputs, torch.tensor([0, 1, 2, 0]), torch.tensor([0, 2, 2, 0]), torch.tensor([0, 1, 0, 1]),
        target_relation=torch.tensor([0, 1, 2, 3]), old_evidence_mask=mask, new_evidence_mask=mask,
        boundary_groups=["g", "g", "g", None],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert "consistency" in parts and "boundary" in parts


def test_rica_pilot_has_family_isolation_and_expected_categories():
    root = Path("data/revimpact_800")
    if not root.exists():
        return
    rows = {split: [json.loads(line) for line in (root / f"{split}.jsonl").open()] for split in ("train", "dev", "test")}
    families = {split: {row["family_id"] for row in values} for split, values in rows.items()}
    assert not families["train"] & families["dev"]
    assert not families["train"] & families["test"]
    assert not families["dev"] & families["test"]
    categories = {row["metadata"]["category"] for values in rows.values() for row in values}
    assert {"changed", "invariant", "boundary", "unsupported"} <= categories
    assert sum(len(values) for values in rows.values()) == 800
