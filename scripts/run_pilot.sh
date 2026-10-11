#!/usr/bin/env bash
set -euo pipefail
python -m revimpact.data.synthetic --output data/pilot.jsonl --n-pairs "${1:-120}" --seed 7
python -m revimpact.evaluation.cli --input data/pilot.jsonl --output outputs/pilot.json
cat outputs/pilot.json
