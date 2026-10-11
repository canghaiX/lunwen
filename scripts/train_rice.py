"""Minimal training entry point for the local RICE multi-task head.

This intentionally trains on JSONL gold records and a frozen/partially
trainable local encoder. A production run should add a real tokenizer-based
collator and document-family split; the script provides the reproducible pilot
scaffold and verifies the model/loss interface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from revimpact.data.schema import Conclusion
from revimpact.evaluation.metrics import load_jsonl
from revimpact.models.rice import RICEModel, rice_loss


def _label(value: str) -> int:
    return {item.value: index for index, item in enumerate(Conclusion)}[value]


def _condition_features(row):
    slots = row.get("condition", {}).get("slots", {})
    values = [
        1.0 if slots.get("model") == "A" else 0.0,
        1.0 if slots.get("model") == "B" else 0.0,
        1.0 if slots.get("environment") == "high_temperature" else 0.0,
        1.0 if slots.get("environment") == "normal_temperature" else 0.0,
        float(slots.get("pressure", 0.0)) / 100.0,
        1.0 if row.get("revision_type") == "changed" else 0.0,
        1.0 if row.get("revision_type") == "boundary" else 0.0,
        1.0 if row.get("revision_type") == "unsupported" else 0.0,
    ]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=os.environ.get("REVIMPACT_MODEL_PATH", "/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--output", default="outputs/rice.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    rows = list(load_jsonl(args.input))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = RICEModel.from_pretrained(args.model)
    model_device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for row in rows:
            text = f"OLD: {row['old_document']} NEW: {row['new_document']} QUERY: {row['query']}"
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            batch = {key: value.to(model_device) for key, value in batch.items()}
            features = torch.tensor([_condition_features(row)], device=model_device)
            output = model(**batch, condition_features=features)
            target_old = torch.tensor([_label(row["old_label"])])
            target_new = torch.tensor([_label(row["new_label"])])
            target_change = torch.tensor([int(row["should_change"])])
            loss = rice_loss(output["old_logits"], output["new_logits"], output["change_logits"],
                             target_old, target_new, target_change)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(json.dumps({"epoch": epoch + 1, "loss": total / max(len(rows), 1)}))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
