from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate_records, load_jsonl
from .baselines import condition_rule_prediction, text_diff_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--predictions")
    parser.add_argument("--baseline", choices=["condition_rule", "text_diff"], default="condition_rule")
    args = parser.parse_args()
    rows = list(load_jsonl(args.input))
    if args.predictions:
        predictions = list(load_jsonl(args.predictions))
    elif args.baseline == "condition_rule":
        predictions = [condition_rule_prediction(row) for row in rows]
    else:
        predictions = [text_diff_prediction(row) for row in rows]
    result = evaluate_records(rows, predictions)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
