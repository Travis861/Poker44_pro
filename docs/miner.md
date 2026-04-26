# 🛠️ Poker44 Miner Guide

This guide covers the production-facing miner flow for Poker44 subnet `126`.

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
./scripts/miner/setup.sh
```

---

## Wallet and Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey --wallet.name my_cold --wallet.hotkey my_poker44_hotkey

btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

---

## Run Miner

Script path: `scripts/miner/run/run_miner.sh`

```bash
chmod +x ./scripts/miner/run/run_miner.sh
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `AXON_PORT`
- `ALLOWED_VALIDATOR_HOTKEYS` for the recommended Swarm-like allowlist mode

If `ALLOWED_VALIDATOR_HOTKEYS` is left empty, the script falls back to `--blacklist.force_validator_permit`.
If DNS or websocket reliability is poor, you can also set `CHAIN_ENDPOINT` to override the
default public RPC endpoint while keeping `NETWORK=finney`.

The script is environment-driven. Example:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

PM2:

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 stop poker44_miner
pm2 delete poker44_miner
```

Direct CLI example with explicit validator allowlist:

```bash
python neurons/miner.py \
  --netuid 126 \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --subtensor.network finney \
  --axon.port 8091 \
  --blacklist.allowed_validator_hotkeys <validator_hotkey_1> <validator_hotkey_2>
```

---

## Request/Response Contract

Miners receive `DetectionSynapse(chunks=...)`, where:

- `chunks` is a list of chunks.
- each chunk is a list of hands.
- return exactly one `risk_score` per chunk.

Expected output fields:

- `risk_scores: List[float]` in `[0, 1]`
- `predictions: List[bool]` (optional but recommended)
- `model_manifest: Dict[str, Any]` (recommended; published automatically by the reference miner)

Important: validator payloads are sanitized to remove label/identity leakage before querying miners.

### Model Manifest

Poker44 miners can publish a lightweight `model_manifest` without changing the current
remote-inference evaluation flow. The validator still scores returned `risk_scores`; the
manifest is for transparency and traceability.

Recommended manifest fields:

- `open_source`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `framework`
- `license`
- `training_data_statement`
- `training_data_sources`
- `private_data_attestation`
- `artifact_url` and `artifact_sha256` when a downloadable checkpoint exists
- `implementation_sha256`

Minimum fields for `transparent` compliance:

- `open_source=true`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `training_data_statement`
- `private_data_attestation`

If these fields are missing, the validator can still score the miner today, but the miner is
classified as `opaque` rather than `transparent`.

The reference miner reads these environment variables when available:

- `POKER44_MODEL_OPEN_SOURCE`
- `POKER44_MODEL_REPO_URL`
- `POKER44_MODEL_REPO_COMMIT`
- `POKER44_MODEL_NAME`
- `POKER44_MODEL_VERSION`
- `POKER44_MODEL_FRAMEWORK`
- `POKER44_MODEL_LICENSE`
- `POKER44_MODEL_ARTIFACT_URL`
- `POKER44_MODEL_ARTIFACT_SHA256`
- `POKER44_MODEL_CARD_URL`
- `POKER44_MODEL_TRAINING_DATA_STATEMENT`
- `POKER44_MODEL_TRAINING_DATA_SOURCES`
- `POKER44_MODEL_PRIVATE_DATA_ATTESTATION`
- `POKER44_MODEL_INFERENCE_MODE`
- `POKER44_MODEL_NOTES`

For the rationale behind these disclosures, see [Anti-Leakage Policy](./anti-leakage.md).

Startup behavior of the reference miner:

- logs the published `model_manifest`
- logs current `transparent` / `opaque` status and missing fields
- logs benchmark/doc paths useful for miner preparation
- logs the public benchmark command miners can use locally

---

## Production Access Policy

Poker44 miners support two production access modes.

Recommended mode, similar to Swarm:

- `--blacklist.allowed_validator_hotkeys <validator_hotkey...>`

Meaning:

- only the listed validator hotkeys may query your miner;
- requests must be signed and pass Bittensor's default request verification;
- this does not depend on `validator_permit=True` being visible on the metagraph.

Fallback mode:

- `--blacklist.force_validator_permit`

Meaning:

- requests from non-permitted peers are rejected;
- access depends on the caller having `validator_permit=True` on the metagraph.

Operational note:

- if `--blacklist.allowed_validator_hotkeys` is set, the miner uses the allowlist policy;
- if no allowlist is set, the miner falls back to the `validator_permit` policy;
- in both cases, your miner must stay reachable and correctly served on-chain.

---

## Training Data (Miner Side)

Public human corpus:

`hands_generator/human_hands/poker_hands_combined.json.gz`

Bot generation:

```bash
python3 hands_generator/bot_hands/generate_poker_data.py
```

Output:

`hands_generator/bot_hands/bot_hands.json`

Validators evaluate with private human data (`POKER44_HUMAN_JSON_PATH`), not with the public training corpus.

Optional public benchmark artifact generation:

```bash
python scripts/publish/publish_public_benchmark.py --skip-wandb
```

This produces a labeled benchmark built only from the public human corpus and offline-generated
bot chunks, with the same sanitized hand schema miners see at inference time. It does not use
the validator's private human dataset and does not expose the live validator batches.

See also:

- [Public benchmark + W&B](./public-benchmark.md)

---

## Health Checklist

- Miner hotkey registered on netuid `126`.
- Axon served and visible on-chain.
- Validator queries are accepted.
- Miner returns non-empty `risk_scores` with correct chunk count.

---

## Reward Optimization

Validator scoring favors miners that are robust in the live forward cycle, not just miners
that respond once. Practical priorities:

1. Optimize your miner model.
   Train for accurate chunk-level bot-risk prediction, minimize false positives, and calibrate
   probabilities so returned `risk_scores` are useful to validators over time.
2. Be transparent with your model manifest.
   Fill out the recommended `model_manifest` fields so the validator can classify your miner as
   `transparent` rather than `opaque`.
3. Maximize availability and speed.
   Validators score miners in cycles with a timeout budget. Keep your axon online and your
   responses comfortably below the timeout.
4. Handle every chunk correctly.
   Validators query chunked windows. Returning incomplete or malformed `risk_scores` can zero
   out otherwise good cycles.
5. Avoid anti-leakage triggers.
   Do not reuse stale outputs or produce suspiciously manipulated results. Keep the miner
   aligned with the manifest, sanitized payload shape, and compliance expectations.
6. Maintain steady uptime.
   Validator fanout samples only part of the miner set each cycle. Stable uptime increases the
   number of cycles in which your miner can be queried and rewarded.
