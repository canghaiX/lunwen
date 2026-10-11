#!/usr/bin/env python3
"""Train the lightweight RICA heads on the 800-query feasibility split."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from revimpact.conditions.logic import ConditionRelation, condition_overlap
from revimpact.data.schema import Conclusion
from revimpact.evaluation.metrics import load_jsonl
from revimpact.models.rica import RICAAdapter, rica_loss


LABELS = {item.value: index for index, item in enumerate(Conclusion)}
RELATIONS = {item.value: index for index, item in enumerate(ConditionRelation)}


def condition_features(row: dict[str, Any]) -> list[float]:
    """Encode query conditions without reading any gold category or label."""
    slots = row.get("condition", {}).get("slots", {})
    model, env, region, mode = (str(slots.get(key, "")).lower() for key in ("model", "environment", "region", "mode"))
    endpoint, group = str(slots.get("endpoint", "")).lower(), str(slots.get("group", "")).lower()
    numeric = next((slots[key] for key in ("value", "pressure", "timeout", "hours", "exposure") if key in slots), None)
    try:
        numeric = float(numeric)
    except (TypeError, ValueError):
        numeric = 0.0
    interval = slots.get("range", slots.get("interval", {}))
    lo = interval.get("min", interval.get("low", 0.0)) if isinstance(interval, dict) else 0.0
    hi = interval.get("max", interval.get("high", 0.0)) if isinstance(interval, dict) else 0.0
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        lo, hi = 0.0, 0.0
    return [
        float(model == "a"), float(model == "b"), float(env in {"high_temperature", "high", "hot"}),
        float(env in {"normal_temperature", "normal", "room"}), float(region == "eu"),
        float(region in {"us", "usa"}), float(region in {"global", "worldwide"}),
        float(mode == "outdoor"), float(mode == "indoor"), float(endpoint == "upload"),
        float(endpoint == "download"), float(group == "adult"), float(group == "child"),
        float(numeric != 0.0), numeric / 100.0, lo / 100.0, hi / 100.0,
        float(bool(row.get("condition", {}).get("valid_from"))),
        float(bool(row.get("condition", {}).get("valid_to"))),
        float(bool(slots)),
    ]


def prompt(row: dict[str, Any], version: str) -> str:
    document = row["old_document"] if version == "old" else row["new_document"]
    condition = json.dumps(row.get("condition", {}), ensure_ascii=False, sort_keys=True)
    return f"DOCUMENT ({version}):\n{document}\nQUERY: {row['query']}\nCONDITION: {condition}"


def _evidence_mask(tokenizer, prompts: list[str], rows: list[dict[str, Any]], version: str, encoded):
    masks = torch.full(encoded["input_ids"].shape, -1.0)
    for index, (text, row) in enumerate(zip(prompts, rows)):
        spans = row.get(f"{version}_evidence", []) or []
        evidence = spans[0].get("text", "") if spans and isinstance(spans[0], dict) else ""
        start = text.find(evidence) if evidence else -1
        offsets = encoded["offset_mapping"][index]
        if start < 0:
            continue
        end = start + len(evidence)
        for token_index, (left, right) in enumerate(offsets):
            if right > left and right > start and left < end:
                masks[index, token_index] = 1.0
            elif encoded["attention_mask"][index, token_index]:
                masks[index, token_index] = 0.0
    return masks


class RICADataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], include_unsupported: bool = False):
        self.rows = [row for row in rows if include_unsupported or row.get("metadata", {}).get("scorable", True)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class BoundaryBatchSampler:
    """Keep each revision's boundary queries together, without dropping rows."""

    def __init__(self, dataset, batch_size, seed):
        self.rows, self.batch_size, self.seed = dataset.rows, batch_size, seed
        self.epoch = 0

    def __iter__(self):
        groups = {}
        for index, row in enumerate(self.rows):
            group = row.get("metadata", {}).get("boundary_group_id") or f"singleton:{index}"
            groups.setdefault(group, []).append(index)
        units = list(groups.values())
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        rng.shuffle(units)
        batch = []
        for unit in units:
            if len(unit) > self.batch_size:
                raise ValueError("boundary group exceeds batch size")
            if len(batch) + len(unit) > self.batch_size:
                yield batch
                batch = []
            batch.extend(unit)
        if batch:
            yield batch


def make_collator(tokenizer, max_length: int, include_targets: bool = True):
    def collate(rows: list[dict[str, Any]]):
        old_prompts = [prompt(row, "old") for row in rows]
        new_prompts = [prompt(row, "new") for row in rows]
        old = tokenizer(old_prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
        new = tokenizer(new_prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
        old_mask = _evidence_mask(tokenizer, old_prompts, rows, "old", old) if include_targets else None
        new_mask = _evidence_mask(tokenizer, new_prompts, rows, "new", new) if include_targets else None
        old.pop("offset_mapping", None)
        new.pop("offset_mapping", None)
        batch = {
            "old": old, "new": new, "condition_features": torch.tensor([condition_features(row) for row in rows], dtype=torch.float32),
            "rows": rows,
        }
        if include_targets:
            relation = [RELATIONS[condition_overlap(row.get("condition", {}), row.get("revision", {}).get("condition", {})).value] for row in rows]
            batch.update({
                "target_old": torch.tensor([LABELS[row["old_label"]] for row in rows]),
                "target_new": torch.tensor([LABELS[row["new_label"]] for row in rows]),
                "target_change": torch.tensor([int(row["should_change"]) for row in rows]),
                "target_relation": torch.tensor(relation), "old_evidence_mask": old_mask,
                "new_evidence_mask": new_mask,
                "boundary_groups": [row.get("metadata", {}).get("boundary_group_id") for row in rows],
                "target_scorable": torch.tensor([row.get("metadata", {}).get("scorable", True) for row in rows]),
                "target_abstain": torch.tensor([int(not row.get("metadata", {}).get("scorable", True)) for row in rows]),
                "target_decision": torch.tensor([
                    2 if not row.get("metadata", {}).get("scorable", True)
                    else (0 if row["should_change"] else 1) for row in rows
                ]),
            })
        return batch
    return collate


def move_batch(batch: dict[str, Any], device):
    for key in ("old", "new"):
        batch[key] = {name: value.to(device) for name, value in batch[key].items()}
    for key in ("condition_features", "target_old", "target_new", "target_change", "target_relation", "old_evidence_mask", "new_evidence_mask", "target_scorable", "target_abstain", "target_decision"):
        if key in batch:
            batch[key] = batch[key].to(device)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/revimpact_800")
    parser.add_argument("--model", default=os.environ.get("REVIMPACT_MODEL_PATH", "/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--output", default="outputs/rica.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--no-condition-gate", action="store_true")
    parser.add_argument("--no-condition-features", action="store_true")
    parser.add_argument("--plain-dual", action="store_true")
    parser.add_argument("--selective", action="store_true", help="Train RICA-Select with unsupported supervision and isolated version heads")
    parser.add_argument("--consistency-weight", type=float, default=None)
    parser.add_argument("--decision-weight", type=float, default=1.0)
    parser.add_argument("--abstain-weight", type=float, default=0.5)
    parser.add_argument("--no-decision-head", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = list(load_jsonl(Path(args.data) / "train.jsonl"))
    dataset = RICADataset(train_rows, include_unsupported=args.selective)
    decision_class_weight = None
    if args.selective:
        counts = torch.bincount(torch.tensor([
            2 if not row.get("metadata", {}).get("scorable", True)
            else (0 if row["should_change"] else 1) for row in dataset.rows
        ]), minlength=3).float()
        decision_class_weight = counts.sum() / (3.0 * counts.clamp_min(1.0))
    train_loader = DataLoader(dataset, batch_sampler=BoundaryBatchSampler(dataset, args.batch_size, args.seed),
                              collate_fn=make_collator(tokenizer, args.max_length))
    model = RICAAdapter(args.model, condition_feature_dim=20, plain_dual=args.plain_dual,
                        selective=args.selective, joint_decision=not args.no_decision_head).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total, steps, boundary_pairs = 0.0, 0, 0
        component_sums = {}
        for batch in train_loader:
            batch = move_batch(batch, device)
            condition_features_input = (torch.zeros_like(batch["condition_features"])
                                        if args.no_condition_features else batch["condition_features"])
            output = model(batch["old"]["input_ids"], batch["old"]["attention_mask"],
                           batch["new"]["input_ids"], batch["new"]["attention_mask"], condition_features_input)
            loss, parts = rica_loss(output, batch["target_old"], batch["target_new"], batch["target_change"],
                                    target_relation=None if args.plain_dual else batch["target_relation"], old_evidence_mask=batch["old_evidence_mask"],
                                    new_evidence_mask=batch["new_evidence_mask"],
                                    boundary_groups=None if args.no_boundary or args.plain_dual else batch["boundary_groups"],
                                    lambda_consistency=0.0 if args.no_consistency or args.plain_dual else (args.consistency_weight if args.consistency_weight is not None else (0.1 if args.selective else 0.5)),
                                    lambda_boundary=0.0 if args.no_boundary or args.plain_dual else 0.5,
                                    condition_gate=None if args.no_condition_gate or args.selective else batch["target_relation"] != RELATIONS[ConditionRelation.DISJOINT.value],
                                    target_scorable=batch["target_scorable"],
                                    target_abstain=batch["target_abstain"] if args.selective else None,
                                    target_decision=batch["target_decision"] if args.selective else None,
                                    lambda_abstain=args.abstain_weight,
                                    lambda_decision=args.decision_weight,
                                    selective=args.selective,
                                    decision_class_weight=decision_class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()); steps += 1
            for key, value in parts.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
            for group in {g for g in batch["boundary_groups"] if g}:
                indices = [i for i, g in enumerate(batch["boundary_groups"]) if g == group]
                positive = sum(int(batch["target_change"][i]) for i in indices)
                boundary_pairs += positive * (len(indices) - positive)
        record = {"epoch": epoch + 1, "train_loss": total / max(steps, 1), "train_n": len(train_loader.dataset),
                  "components": {key: value / max(steps, 1) for key, value in component_sums.items()},
                  "boundary_pairs_available": boundary_pairs}
        history.append(record)
        print(json.dumps(record))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value for key, value in model.state_dict().items() if not key.startswith("encoder.")},
                "adapter_only": True, "model": args.model, "condition_feature_dim": 20,
                "selective": args.selective,
                "joint_decision": args.selective and not args.no_decision_head,
                "decision_class_weight": (decision_class_weight.tolist()
                                           if decision_class_weight is not None else None),
                "training_config": vars(args),
                "max_length": args.max_length, "history": history,
                "ablation": {"no_consistency": args.no_consistency, "no_boundary": args.no_boundary,
                "no_condition_gate": args.no_condition_gate,
                "no_condition_features": args.no_condition_features, "plain_dual": args.plain_dual}}, output)
    print(json.dumps({"saved": str(output), "history": history}))


if __name__ == "__main__":
    main()
