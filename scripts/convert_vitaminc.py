#!/usr/bin/env python3
"""Convert VitaminC files into split-wise RevImpact JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revimpact.data.adapters.vitaminc import convert_path, discover_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/vitaminc")
    parser.add_argument("--output", default="data/vitaminc_converted")
    args = parser.parse_args()
    groups = convert_path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"input": str(args.input), "files": {key: [str(item) for item in value] for key, value in discover_files(args.input).items()}, "splits": {}}
    for split, records in groups.items():
        target = output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
        unknown_old = sum(record.old_label.value == "unknown" for record in records)
        unknown_new = sum(record.new_label.value == "unknown" for record in records)
        summary["splits"][split] = {"n": len(records), "unknown_old": unknown_old, "unknown_new": unknown_new}
    (output / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
