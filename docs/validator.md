# 🔐 Poker44 Validator Guide

Production validator guide for Poker44 subnet `126`.

---

## Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+
- Registered validator hotkey on netuid `126`

---

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use the helper script:

```bash
./scripts/validator/main/setup.sh
```

---

## Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name p44_cold --subtensor.network finney
```

---

## Required Environment

Mandatory:

- `POKER44_HUMAN_JSON_PATH` (private local human dataset JSON)

Optional tuning:

- `POKER44_DATASET_REFRESH_SECONDS` (default `3600`)
- `POKER44_POLL_INTERVAL_SECONDS` (default `300`)
- `POKER44_REWARD_WINDOW` (default `40`)
- `POKER44_CHUNK_COUNT` (default `40`)
- `POKER44_MIN_HANDS_PER_CHUNK` (default `60`)
- `POKER44_MAX_HANDS_PER_CHUNK` (default `120`)
- `POKER44_HUMAN_RATIO` (default `0.5`)
- `POKER44_MINERS_PER_CYCLE` (default `24`; set `0` or a negative value to query all eligible miners)
- `POKER44_TARGET_MINER_UIDS` (comma-separated UIDs, useful for controlled local tests)
- `--neuron.timeout` (default `60s`, validator -> miner query timeout)
- `--wandb.off` (disable Weights & Biases logging)
- `--wandb.offline` (log to local offline Weights & Biases files only)
- `--wandb.project_name` (default `poker44-validators`)
- `--wandb.entity` (optional W&B entity/team)
- `--wandb.notes` (optional run notes)
- `POKER44_VALIDATOR_RUNTIME_REPORT_URL` (optional override; defaults to `https://api.poker44.net/internal/validators/runtime`)
- `POKER44_VALIDATOR_RUNTIME_REPORT_TIMEOUT_SECONDS` (default `5`)

---

## Run Validator

### PM2 command

```bash
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
POKER44_CHUNK_COUNT=40 \
POKER44_REWARD_WINDOW=40 \
POKER44_POLL_INTERVAL_SECONDS=300 \
POKER44_MINERS_PER_CYCLE=24 \
POKER44_SYNC_DIRECT_SCORE_UPDATE=false \
POKER44_SYNC_RESET_BUFFERS_ON_WINDOW_CHANGE=false \
pm2 start python --name poker44_validator -- \
  ./neurons/validator.py \
  --netuid 126 \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --subtensor.network finney \
  --neuron.timeout 60 \
  --logging.debug
```

### Script

Script path: `scripts/validator/run/run_vali.sh`

```bash
chmod +x ./scripts/validator/run/run_vali.sh
./scripts/validator/run/run_vali.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `POKER44_HUMAN_JSON_PATH`

The script is environment-driven. Example:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
POKER44_CHUNK_COUNT=40 \
POKER44_REWARD_WINDOW=40 \
POKER44_POLL_INTERVAL_SECONDS=300 \
POKER44_MINERS_PER_CYCLE=24 \
POKER44_SYNC_DIRECT_SCORE_UPDATE=false \
POKER44_SYNC_RESET_BUFFERS_ON_WINDOW_CHANGE=false \
NEURON_TIMEOUT=60 \
./scripts/validator/run/run_vali.sh
```

PM2:

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

Startup logs now include:

- validator UID and hotkey
- subnet code version
- `VALIDATOR_DEPLOY_VERSION`
- git branch / short commit / dirty state

The validator also writes a local runtime snapshot to:

- `$(full_path)/validator_runtime.json`
- `$(full_path)/network_snapshot.json`

Those snapshots are updated automatically. The runtime snapshot includes the current version/deploy metadata, sync mode flags, latest `set_weights` result, and basic score-state counters. The network snapshot includes the validator's current metagraph-derived view of Subnet 126 for Poker44 platform dashboards.

By default, both snapshots are pushed automatically to Poker44's central collector using hotkey-signed requests. Override `POKER44_VALIDATOR_RUNTIME_REPORT_URL` or `POKER44_VALIDATOR_NETWORK_SNAPSHOT_REPORT_URL` only if you need to point the validator at a different collector.

---

## Auto-Update

Poker44 supports optional validator auto-update through a separate PM2 watcher process.

How it works:

- The watcher checks `origin/main` periodically.
- It reads `VALIDATOR_DEPLOY_VERSION` from `poker44/__init__.py`.
- It updates only when the remote deploy version is newer than the local one.
- On update, it pulls the repo, reinstalls dependencies, and restarts the validator PM2 process.

Files:

- `scripts/validator/update/auto_update_validator.sh`
- `scripts/validator/update/update_validator.sh`
- `scripts/validator/update/update_full.sh`

Recommended environment for the watcher:

- `PROCESS_NAME` (default `poker44_validator`)
- `WALLET_NAME`
- `WALLET_HOTKEY`
- `SUBTENSOR_PARAM` (default `--subtensor.network finney`)
- `VALIDATOR_ENV_DIR` (default `validator_env`)
- `VALIDATOR_EXTRA_ARGS`
- `SLEEP_INTERVAL` (default `600`)
- `TARGET_BRANCH` (default `main`)

Start the watcher:

```bash
chmod +x scripts/validator/update/auto_update_validator.sh
pm2 start --name poker44_auto_update \
  --interpreter /bin/bash \
  scripts/validator/update/auto_update_validator.sh
pm2 save
```

Typical one-time setup:

```bash
PROCESS_NAME=poker44_validator \
WALLET_NAME=p44_cold \
WALLET_HOTKEY=p44_validator \
SUBTENSOR_PARAM="--subtensor.network finney" \
VALIDATOR_ENV_DIR=validator_env \
SLEEP_INTERVAL=600 \
pm2 start --name poker44_auto_update \
  --interpreter /bin/bash \
  scripts/validator/update/auto_update_validator.sh
```

Manual update:

```bash
chmod +x scripts/validator/update/update_validator.sh
./scripts/validator/update/update_validator.sh
```

Stop or inspect:

```bash
pm2 logs poker44_auto_update
pm2 restart poker44_auto_update --update-env
pm2 stop poker44_auto_update
pm2 delete poker44_auto_update
```

The auto-update watcher now logs:

- local vs remote `VALIDATOR_DEPLOY_VERSION`
- local vs remote git commit
- updated commit after a successful pull

Notes:

- Auto-update is optional.
- Validators still control whether they enable the watcher.
- Deploys are gated by `VALIDATOR_DEPLOY_VERSION`, not by every commit on `main`.

---

## Runtime Behavior

Per cycle, validator:

1. Builds mixed labeled chunks from private human data + generated bot data.
2. Sanitizes payloads before sending to miners.
3. Queries miners, records any returned `model_manifest`, and scores returned `risk_scores`.
4. Updates internal scores and attempts `set_weights` on-chain.

## Model Manifest Registry

Poker44 validators now persist miner model metadata separately from scoring. This does not
change the reward loop. It records what each miner claims to be running.

Current behavior:

- manifests are optional for backward compatibility;
- when present, they are normalized and stored under the validator state directory as
  `model_manifests.json`;
- validator compliance state is persisted separately in `compliance_registry.json`;
- validator anti-leakage tracking also persists `suspicion_registry.json` and
  `served_chunk_registry.json`;
- a manifest change is detected by digest and logged once per update;
- miners are classified as `transparent` or `opaque` based on minimum manifest fields;
- missing or incomplete disclosure fields create suspicion events per UID;
- served chunk fingerprints are tracked to monitor repeated exposure of evaluation payloads;
- `set_weights` still depends on prediction quality, not on manifest presence.

For the broader threat model and planned controls around leaked private data, memorization,
and hardcoded miners, see [Anti-Leakage Policy](./anti-leakage.md).

Optional W&B integration:

- Logs only aggregated validator telemetry.
- Includes dataset hash, dataset statistics, forward-cycle summaries, reward summaries, and `set_weights` status.
- Does not publish live chunks, private human data, or the validator's mixed evaluation dataset contents.
- Public benchmark publication for miners is documented separately in [Public benchmark + W&B](./public-benchmark.md).

Default production cadence:

- dataset refresh: every `3600s`
- query loop: every `300s` unless overridden
- miner fanout: `24` miners per cycle by default, rotating across the eligible set

Validated starting profile:

- `POKER44_CHUNK_COUNT=40`
- `POKER44_REWARD_WINDOW=40`
- `--neuron.timeout 60`
- `POKER44_MINERS_PER_CYCLE=24`
- `POKER44_SYNC_DIRECT_SCORE_UPDATE=false`
- `POKER44_SYNC_RESET_BUFFERS_ON_WINDOW_CHANGE=false`

These defaults were validated as a practical starting point for production-like runs:

- `80` chunks with the current heuristic miners caused validator query timeouts;
- `40` chunks with `60s` timeout completed successfully;
- setting `POKER44_REWARD_WINDOW=40` allows miners to receive non-zero weights from the first completed cycle.
- querying the full eligible set in one cycle degraded useful miner responses; rotating a subset per cycle was more stable.
- increasing fanout modestly while keeping persistent scoring reduces sampling noise without reintroducing the worst timeout behavior.

---

## Production Checklist

- Validator logs show forward cycles and eligible miner UIDs.
- Miners return non-empty `risk_scores` with expected chunk count.
- Validator logs periodic successful weight submissions:
  - `set_weights on chain successfully!`

---

## Help

- Open a GitHub issue for bugs or missing behavior.
