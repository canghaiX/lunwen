#!/usr/bin/env python3
"""Optional Qwen local-LLM baseline for converted VitaminC JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from revimpact.data.schema import Conclusion
from revimpact.evaluation.metrics import load_jsonl


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        lowered = text.lower()
        labels = ["contradicted", "entailed", "unknown"]
        found = [label for label in labels if label in lowered]
        return {"pred_old_label": found[0] if found else "unknown", "pred_new_label": found[1] if len(found) > 1 else "unknown", "pred_should_change": "should_change" in lowered and ("true" in lowered or "yes" in lowered), "confidence": 0.0, "raw_output": text[:1000]}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"pred_old_label": "unknown", "pred_new_label": "unknown", "pred_should_change": False, "confidence": 0.0, "raw_output": text[:1000]}
    labels = {item.value for item in Conclusion}
    old_label = value.get("old_label", "unknown") if value.get("old_label") in labels else "unknown"
    new_label = value.get("new_label", "unknown") if value.get("new_label") in labels else "unknown"
    return {
        "pred_old_label": old_label,
        "pred_new_label": new_label,
        "pred_should_change": bool(value.get("should_change", old_label != new_label)),
        "pred_evidence_ids": value.get("evidence_ids", []),
        "confidence": float(value.get("confidence", 0.0) or 0.0),
        "raw_output": text[:1000],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model_kwargs = {"local_files_only": True, "trust_remote_code": True, "torch_dtype": "auto"}
    try:
        import accelerate  # noqa: F401
        model_kwargs["device_map"] = "auto"
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if "device_map" not in model_kwargs and torch.cuda.is_available():
        model = model.to("cuda")
    predictions = []
    for row in load_jsonl(args.input):
        device = next(model.parameters()).device
        prompt = (
            "Compare the original and edited evidence for the claim. Return only JSON with "
            "old_label, new_label, should_change, evidence_ids, confidence. Labels must be "
            "entailed, contradicted, or unknown.\n"
            f"CLAIM: {row['query']}\nORIGINAL: {row['old_document']}\nEDITED: {row['new_document']}"
        )
        messages = [
            {"role": "system", "content": "You are a fact verification classifier. Return only valid JSON. /no_think"},
            {"role": "user", "content": prompt},
        ]
        use_template = hasattr(tokenizer, "apply_chat_template")
        if use_template:
            try:
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                        return_tensors="pt", enable_thinking=False)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(device)
                else:
                    inputs = {key: value.to(device) for key, value in inputs.items()}
            except (ImportError, RuntimeError, ValueError):
                use_template = False
        if not use_template:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append({"id": row["id"], **_parse(text)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
