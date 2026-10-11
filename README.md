# RevImpact

RevImpact is a reproducible pilot for **Revision-Aware Conditional Entailment**. Given an old document, a revised document, a query, and its conditions, the system predicts old/new conclusions, whether the conclusion should change, supporting evidence, and a selective-maintenance action.

The repository deliberately has a dependency-light path that runs without model weights. It provides a controlled pilot generator, condition and interval reasoning, deterministic baselines, metrics, and tests. Optional Transformers code can be added on top of the same JSONL schema for frozen LLM extraction and a learned RICE head.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m revimpact.data.synthetic --output data/pilot.jsonl --n-pairs 120 --seed 7
python -m revimpact.evaluation.cli --input data/pilot.jsonl --output outputs/pilot.json
pytest -q
```

## Docker environment

The service and container are both named `lunwen`. The Dockerfile uses the
preloaded local `inferkernellab:cuda` image so it does not need Docker Hub
access in this environment. It mounts `/data/models` read-only; the default
local model is `/models/Qwen/Qwen3-0.6B` and can be changed with
`REVIMPACT_MODEL_PATH`.

```bash
docker build -t lunwen:latest .
docker run --rm -it --gpus all --name lunwen \
  -v "$PWD:/workspace/lunwen" -v /data/models:/models:ro \
  -w /workspace/lunwen lunwen:latest bash
```

Inside the container, run `bash scripts/run_pilot.sh`. For the optional local
LLM extractor, use `LocalLLMRevisionExtractor` with the mounted model path.

The pilot is a development harness, not the final ACL dataset. For the paper, replace or augment it with versioned public documents and human annotations while preserving the schema in `src/revimpact/data/schema.py`.

## Task schema

Each JSONL record stores `old_document`, `new_document`, `query`, structured `condition`, old/new labels (`entailed`, `contradicted`, `unknown`), `should_change`, evidence spans, a revision type, and an action (`keep`, `verify`, `update`, `invalidate`, `human_review`). The split field is assigned by document family rather than by individual query.

## Research implementation

`src/revimpact/revision/extract.py` contains a conservative rule extractor for pilot data and a JSON contract for an optional LLM extractor. `src/revimpact/conditions/logic.py` implements typed slots, interval overlap, version validity, and condition matching. `src/revimpact/models/rice.py` implements the deterministic RICE decision layer and an optional PyTorch loss for old/new entailment, change, boundary, evidence, and invariance heads. The latter is intentionally modular so a frozen encoder or a 7B LLM can be plugged in without changing evaluation.

The first real-data milestone is 100 revision pairs and 500 manually checked queries. Do not claim novelty or final ACL performance from the synthetic pilot alone.

## VitaminC

Put the downloaded VitaminC files under `data/vitaminc/` (the container maps
the same directory). The adapter accepts JSON, JSONL, TSV and Parquet and
recognizes common VitaminC aliases such as `claim`, `evidence`, `label`,
`original_evidence`, and `edited_evidence`.

```bash
python scripts/convert_vitaminc.py --input data/vitaminc --output data/vitaminc_converted
python scripts/evaluate_vitaminc.py --data data/vitaminc_converted --output outputs/vitaminc_baselines.json
```

An optional local-Qwen prompting baseline writes predictions separately and
can then be evaluated with `--predictions` on the general evaluation CLI:

```bash
python scripts/predict_vitaminc_llm.py \
  --input data/vitaminc_converted/test.jsonl \
  --output outputs/vitaminc.test.llm.predictions.jsonl \
  --model /models/Qwen/Qwen3-0.6B
python -m revimpact.evaluation.cli \
  --input data/vitaminc_converted/test.jsonl \
  --predictions outputs/vitaminc.test.llm.predictions.jsonl
```

The conversion keeps every source row in `metadata.raw`. Rows without reliable
old/new labels are conservatively marked `unknown`; they are not silently
converted into a changed example. The evaluation reports classification,
changed/unchanged accuracy, unknown-label accuracy, and family/document
overlap diagnostics.

## Balanced feasibility pilot

VitaminC is highly skewed toward changed evidence, so the first task signal
check uses a balanced controlled pilot alongside a sampled VitaminC auxiliary
set:

```bash
python scripts/build_pilot_dataset.py \
  --vitaminc data/vitaminc_converted \
  --output data/revimpact_pilot \
  --aux-output data/vitaminc_aux \
  --families 400 --queries-per-family 8 --seed 7

python scripts/evaluate_pilot.py \
  --data data/revimpact_pilot \
  --output outputs/revimpact_pilot_results.json \
  --model /models/Qwen/Qwen3-0.6B --llm-samples 200
```

The pilot contains changed, invariant, boundary, and unsupported/conflict
queries across three domains. Unsupported rows carry
`metadata.scorable=false`; they are excluded from the primary impact score and
reported separately. Predictions are written beside the result JSON as one
JSONL file per split and baseline. The auxiliary set is evaluated with
`scripts/evaluate_vitaminc.py` and is not a substitute for the balanced pilot.

## RICA 800-query feasibility run

The RICA milestone combines 500 controlled hard cases with 300 natural
VitaminC revision pairs. Controlled boundary cases are paired inside a family
(one affected and one invariant query), while all splits are family-isolated.
The generated manifest contains pending human-review fields; it is not a
claim that these automatic labels have already been double annotated.

```bash
python scripts/build_revimpact_800.py \
  --vitaminc data/vitaminc_converted --output data/revimpact_800 --seed 7
python scripts/train_rica.py \
  --data data/revimpact_800 --model /models/Qwen/Qwen3-0.6B \
  --output outputs/rica.pt --epochs 2 --batch-size 8 --max-length 256
python scripts/evaluate_rica.py \
  --data data/revimpact_800 --checkpoint outputs/rica.pt \
  --model /models/Qwen/Qwen3-0.6B --output outputs/rica_results.json
```

`src/revimpact/models/rica.py` freezes the Qwen encoder and trains only the
revision fusion, condition projection, old/new entailment, impact, boundary,
and evidence heads. `train_rica.py` exposes `--no-consistency`,
`--no-boundary`, and `--no-condition-gate` for ablations. The deterministic
`condition_rule_oracle` is reported as an upper bound and must not be used as
the learned-method result.

## Two-stage real-document annotation

The 300 natural VitaminC records can be annotated without exposing their
existing labels to the models. Run Qwen3-Next on all eight GPUs; the command
is resumable and writes raw outputs:

```bash
python scripts/annotate_real_docs.py \
  --data data/revimpact_800 \
  --model /models/Qwen/Qwen3-Next-80B-A3B-Instruct \
  --output data/revimpact_800/annotations_qwen80b.jsonl
```

Start the Qwen3.6 audit server in a separate container/session:

```bash
scripts/launch_qwen36.sh
python scripts/review_annotations.py \
  --data data/revimpact_800 \
  --initial data/revimpact_800/annotations_qwen80b.jsonl \
  --output data/revimpact_800/annotations_qwen35b_review.jsonl
```

Create the human-review file, fill every row with `verified`, `uncertain`, or
`excluded`, and then create the final dataset:

```bash
python scripts/finalize_annotations.py --mode template
# edit data/revimpact_800/annotations_human_final.jsonl
python scripts/finalize_annotations.py --mode finalize
python scripts/evaluate_annotated_rica.py \
  --data data/revimpact_800_annotated \
  --output outputs/annotated_rica_results.json
```

Only `verified` human rows are scorable. Model annotations remain silver labels
and are retained for agreement and error analysis.

The first Qwen3-Next run produced 300/300 parseable JSON annotations. The
reported confidence field is not calibrated (the model frequently omits it),
so it must not be used as a gold-quality filter. Qwen3.6 review requires a
separate serving environment compatible with `qwen3_5_moe`; the base RICA
image intentionally keeps its Transformers/kernels stack unchanged.

When that serving backend is unavailable, the compatible local `Qwen3-4B` can
run the same audit prompt as a fallback reviewer. It writes a separate silver
review file and never becomes a gold label:

```bash
python scripts/review_annotations.py \
  --backend transformers \
  --model /models/Qwen/Qwen3-4B \
  --reviewer qwen3-4b \
  --initial data/revimpact_800/annotations_qwen80b.jsonl \
  --output data/revimpact_800/annotations_qwen4b_review.jsonl
python scripts/normalize_annotation_reviews.py \
  --input data/revimpact_800/annotations_qwen4b_review.jsonl \
  --output data/revimpact_800/annotations_qwen4b_review_normalized.jsonl
```

## RICA-Select pilot

The selective variant adds an explicit `change/keep/abstain` decision. It
trains the abstention head on unsupported rows while excluding their placeholder
`unknown` labels from old/new entailment and impact losses:

```bash
python scripts/train_rica.py \
  --data data/revimpact_800 \
  --model /models/Qwen/Qwen3-0.6B \
  --output outputs/rica_selective_seed7.pt \
  --epochs 3 --batch-size 16 --max-length 384 --seed 7 --selective

python scripts/evaluate_rica_selective.py \
  --data data/revimpact_800 \
  --checkpoint outputs/rica_selective_seed7.pt
```

Thresholds are selected on `dev`; the test file is evaluated once. The
calibrated two-score policy is the pilot default; the explicit three-class
decision head can be run with `--threshold-mode head` as an ablation. Aggregate
multiple seeds with `scripts/aggregate_rica_selective.py`. Existing RICA
checkpoints and the binary evaluator remain compatible.
