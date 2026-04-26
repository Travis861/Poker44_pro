#!/bin/bash

# Poker44 Miner Startup Script
set -euo pipefail

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-}"
HOTKEY="${HOTKEY:-wolf_miner_2}"
NETWORK="${NETWORK:-finney}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-wolf_miner}"
AXON_PORT="${AXON_PORT:-8091}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

if [ -z "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  ALLOWED_VALIDATOR_HOTKEYS="5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126 5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD 5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k 5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126 5C8R8ifnxswxhSsRiRhkriRAThdryCpkP6ScZXUotJhsuNZD"
fi

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

if [ -z "$WALLET_NAME" ]; then
    echo "Error: WALLET_NAME must be set before starting the miner"
    exit 1
fi

if [ -z "$HOTKEY" ]; then
    echo "Error: HOTKEY must be set before starting the miner"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$CHAIN_ENDPOINT" ]; then
  MINER_ARGS+=(--subtensor.chain_endpoint "$CHAIN_ENDPOINT")
fi

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$CHAIN_ENDPOINT" ]; then
    echo "Chain endpoint override: $CHAIN_ENDPOINT"
fi
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
