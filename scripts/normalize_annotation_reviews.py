#!/usr/bin/env python3
"""Re-parse saved reviewer raw outputs after parser improvements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revimpact.evaluation.metrics import load_jsonl
from revimpact.llm.annotation import parse_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for row in load_jsonl(args.input):
        normalized, failed = parse_review(row.get("raw_output", ""))
        row["review"] = normalized
        row["parse_failed"] = failed
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "output": str(output), "parse_failed": sum(row["parse_failed"] for row in rows)}))


if __name__ == "__main__":
    main()
