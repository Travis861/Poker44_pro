# Public Benchmark and W&B

This document describes the public benchmark flow intended for miner training/reference use.

## Purpose

The public benchmark exists to give miners:

- a reproducible labeled dataset for local training and offline evaluation;
- a schema aligned with the sanitized validator payloads miners receive at inference time;
- an artifact that can be published to Weights & Biases (W&B) without exposing validator-private evaluation data.

## Data Boundary

The public benchmark is built only from:

- the public human corpus committed in the repo:
  `hands_generator/human_hands/poker_hands_combined.json.gz`
- offline-generated bot chunks derived from the public corpus

It does **not** use:

- `POKER44_HUMAN_JSON_PATH`
- validator-private human datasets
- `data/validator_mixed_chunks.json`
- live validator batches sent to miners

## Output

The benchmark builder produces a labeled dataset with:

- `train` / `validation` split per chunk
- `is_bot` ground-truth label per chunk
- sanitized hands matching the miner-visible schema
- aggregate dataset statistics
- dataset hash for versioning

Default output path:

`data/public_miner_benchmark.json.gz`

## Build Locally

```bash
python scripts/publish/publish_public_benchmark.py --skip-wandb
```

Example with explicit output path:

```bash
python scripts/publish/publish_public_benchmark.py \
  --skip-wandb \
  --output-path /tmp/p44-public-benchmark.json.gz
```

## Publish to W&B

Offline test:

```bash
WANDB_MODE=offline python scripts/publish/publish_public_benchmark.py --offline
```

Online publish:

```bash
export WANDB_API_KEY=...

python scripts/publish/publish_public_benchmark.py \
  --wandb-project poker44-miner-benchmarks \
  --wandb-entity <your-team>
```

## What W&B Publishes

The publish script logs:

- the benchmark artifact file
- dataset hash
- chunk counts and split counts
- shortcut-rule accuracy
- aggregate benchmark metadata

It does not publish validator-private human data or live validator evaluation chunks.

## Relationship to Validator Live Evaluation

This benchmark is a public training/reference artifact, not a copy of production validator evaluation.

Validators in production still evaluate miners using:

- private human-hand data available only on validator infrastructure
- dynamically generated mixed batches
- previously unseen hands delivered in real time

The public benchmark is meant to help miners train and validate against the task definition and payload shape, without revealing the live validator evaluation stream.
