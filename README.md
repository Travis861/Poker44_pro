<div align="center">
  <h1>🂡 <strong>Poker44</strong> — Poker Bot Detection Subnet</h1>
  <img src="poker44/assets/logopoker44.png" alt="Poker44 logo" style="width:320px;">
  <p>
    <a href="docs/validator.md">🔐 Validator Guide</a> &bull;
    <a href="docs/miner.md">🛠️ Miner Guide</a> &bull;
    <a href="docs/anti-leakage.md">🛡️ Anti-Leakage</a> &bull;
    <a href="docs/roadmap.md">🗺️ Roadmap</a>
  </p>
</div>

---

## Official Links

- X: https://x.com/poker44subnet
- Web: https://poker44.net
- Whitepaper: https://poker44.net/Poker44_Whitepaper.pdf

---

## What is Poker44?

Poker44 is a Bittensor subnet focused on one problem: detecting bots in online poker with objective, reproducible evaluation.

Validators build labeled evaluation windows (human vs bot behavior), query miners, score predictions, and publish weights on-chain.  
Miners compete by returning robust bot-risk predictions that generalize to evolving bot behavior.

Poker44 is security infrastructure, not a poker room.

---

## Vision

### Short-Mid Term (Subnet Operating Model)

Poker44 currently uses a hybrid operating model to generate high-quality labeled datasets for miner evaluation.  
The immediate direction is to consolidate this into a decentralized runtime path where gameplay/integrity services are executed on validator infrastructure, with attested execution and reproducible evaluation loops.

### Mid-Long Term (Global Decentralized Platform)

Beyond the current hybrid stage, Poker44 targets a fully decentralized poker integrity platform:

- integrity and model-evaluation loop coordinated through the subnet,
- transparent, verifiable settlement through smart contracts,
- global trust-minimized operation with auditable behavior validation.

In short: today’s hybrid platform is the data/evaluation engine; the destination is a global decentralized platform with on-chain settlement guarantees.

---

## Target Outcome

The subnet is designed to support production anti-bot workflows where suspicious behavior is detected early and reviewed with evidence.

<div align="center">
  <img src="poker44/assets/bot_detected.png" alt="Example bot detection overlay on a poker table" style="max-width:900px;width:100%;">
</div>

---

## How the Subnet Works (V0)

### Validators

- Build mixed labeled chunks from private human hands plus generated bot hands.
- Query miners with standardized chunk payloads.
- Score miner outputs and set weights on-chain.

### Miners

- Receive chunked poker behavior payloads.
- Return `risk_scores` and predicted labels for each chunk.
- Publish a lightweight `model_manifest` describing the implementation behind the miner.
- Compete on accuracy, calibration, low false positives, and robustness over time.

---

## Data Model

### Public training data for miners

The repo includes a compressed human corpus:

`hands_generator/human_hands/poker_hands_combined.json.gz`

Intended use:

- Use it as human base data.
- Generate bot hands with `hands_generator/bot_hands/generate_poker_data.py`.
- Train your own model and features.
- Optionally build and publish a public benchmark artifact with:
  `python scripts/publish/publish_public_benchmark.py --skip-wandb`

### Validator evaluation data

Validators should not rely on the public corpus for evaluation.  
Set `POKER44_HUMAN_JSON_PATH` to a private local human dataset.

The public benchmark builder uses only the repo public human corpus plus offline-generated
bot chunks. It does not use validator-private human data and does not publish the live
validator evaluation dataset.

### Open-Source Miner Standard

Poker44 now supports a lightweight `model_manifest` attached to normal miner responses.
This does not change validator scoring or on-chain `set_weights`. It adds traceability for
miners that want to make their models open source while the subnet keeps the current
remote-inference evaluation loop.

Recommended manifest fields:

- public repo URL
- repo commit or tag for the production version
- model name and version
- framework
- license
- optional artifact URL / artifact SHA256

---

## Quick Start

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then follow:

- [Validator setup](docs/validator.md)
- [Miner setup](docs/miner.md)
- [Public benchmark + W&B](docs/public-benchmark.md)

Validator operators can also enable optional auto-update support. See [Validator setup](docs/validator.md).

Validated starting profile for production-like operation:

- `POKER44_CHUNK_COUNT=40`
- `POKER44_REWARD_WINDOW=40`
- `POKER44_POLL_INTERVAL_SECONDS=300`
- `--neuron.timeout 60`

Production-facing miner run flow:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

If `ALLOWED_VALIDATOR_HOTKEYS` is not set, the miner falls back to
`--blacklist.force_validator_permit`. If public RPC reliability is poor, set
`CHAIN_ENDPOINT` to a more stable websocket endpoint or local node.

---

## Miner Priorities for Reward

Validator logic rewards miners that consistently produce useful bot-risk scores under the
production query loop. In practice, miner operators should prioritize:

- high-quality, well-calibrated predictions on mixed human/bot chunks;
- a complete `model_manifest` with transparent training-data disclosures;
- low-latency, always-online axon availability;
- correct handling of every requested chunk in a cycle;
- compliant behavior that avoids anti-leakage and suspicious-output triggers;
- stable uptime so the miner is available when validator fanout samples the network.

---

## Repository Links

- Validator docs: [`docs/validator.md`](docs/validator.md)
- Miner docs: [`docs/miner.md`](docs/miner.md)
- Anti-leakage policy: [`docs/anti-leakage.md`](docs/anti-leakage.md)
- Open-sourced roadmap: [`docs/opensourced_roadmap.md`](docs/opensourced_roadmap.md)
- Roadmap: [`docs/roadmap.md`](docs/roadmap.md)

---

## License

MIT — see [`LICENSE`](LICENSE).
