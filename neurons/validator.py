# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Poker44 validator entrypoint wired into the base Bittensor neuron."""
# neuron/validator.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__, VALIDATOR_DEPLOY_VERSION
from poker44.base.validator import BaseValidatorNeuron
from poker44.utils.config import config
from poker44.utils.network_snapshot import collect_network_snapshot
from poker44.utils.runtime_info import (
    build_signed_runtime_request,
    collect_runtime_info,
    post_runtime_snapshot,
    write_runtime_snapshot,
)
from poker44.utils.wandb_helper import ValidatorWandbHelper
from poker44.validator.forward import forward as forward_cycle
from poker44.validator.integrity import (
    load_json_registry,
    normalize_uid_key_registry,
)
from hands_generator.mixed_dataset_provider import (
    DEFAULT_OUTPUT_PATH,
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
)

load_dotenv()
os.makedirs("./logs", exist_ok=True)
bt.logging.set_trace()
bt.logging(debug=True, trace=False, logging_dir="./logs", record_log=True)

DEFAULT_VALIDATOR_RUNTIME_REPORT_URL = (
    "https://api.poker44.net/internal/validators/runtime"
)
DEFAULT_NETWORK_SNAPSHOT_REPORT_URL = (
    "https://api.poker44.net/internal/network/snapshots"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class Validator(BaseValidatorNeuron):
    """Poker44 validator neuron wired into the BaseValidator scaffold."""

    def __init__(self):
        cfg = config(Validator)
        self.poll_interval = None
        self.reward_window = None
        self.synced_window_mode = None
        self.sync_all_miners = None
        self.sync_direct_score_update = None
        self.coverage_round_index = 0
        self.coverage_round_expected_uids = []
        self.coverage_round_seen_uids = set()
        self.coverage_round_reward_sums = {}
        self.coverage_round_reward_counts = {}
        self.coverage_round_pending_set_weights = False
        self.coverage_round_completed_at_step = None
        super().__init__(config=cfg)
        bt.logging.info(f"🚀 Poker44 Validator v{__version__} started")

        self.forward_count = 0
        self.settings = cfg

        human_json_env = os.getenv("POKER44_HUMAN_JSON_PATH")
        if not human_json_env:
            raise RuntimeError(
                "POKER44_HUMAN_JSON_PATH must point to the private local human-hand JSON used by validators."
            )

        human_json_path = Path(human_json_env).expanduser().resolve()
        mixed_output_path = Path(
            os.getenv("POKER44_MIXED_DATASET_PATH", str(DEFAULT_OUTPUT_PATH))
        ).expanduser().resolve()
        refresh_seconds = int(
            os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(60 * 60))
        )
        chunk_count = int(os.getenv("POKER44_CHUNK_COUNT", "40"))
        min_hands_per_chunk = int(os.getenv("POKER44_MIN_HANDS_PER_CHUNK", "60"))
        max_hands_per_chunk = int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK", "120"))
        human_ratio = float(os.getenv("POKER44_HUMAN_RATIO", "0.5"))
        dataset_seed_env = os.getenv("POKER44_DATASET_SEED")
        dataset_seed = int(dataset_seed_env) if dataset_seed_env is not None else None
        self.chunk_batch_size = chunk_count
        self.dataset_cfg = MixedDatasetConfig(
            human_json_path=human_json_path,
            output_path=mixed_output_path,
            chunk_count=chunk_count,
            min_hands_per_chunk=min_hands_per_chunk,
            max_hands_per_chunk=max_hands_per_chunk,
            human_ratio=human_ratio,
            refresh_seconds=refresh_seconds,
            seed=dataset_seed,
        )
        self.provider = TimedMixedDatasetProvider(self.dataset_cfg)
        bt.logging.info(
            f"📁 Using mixed dataset provider | human_json={human_json_path} output={mixed_output_path} "
            f"chunks={chunk_count} hands_range=[{min_hands_per_chunk},{max_hands_per_chunk}] "
            f"ratio={human_ratio} refresh_s={refresh_seconds}"
        )
        bt.logging.info("🧭 Dataset generation is deterministic per refresh window.")
        configured_poll_interval = getattr(cfg, "poll_interval_seconds", refresh_seconds)
        self.poll_interval = int(
            os.getenv("POKER44_POLL_INTERVAL_SECONDS", str(configured_poll_interval))
        )
        self.reward_window = int(os.getenv("POKER44_REWARD_WINDOW", "40"))
        self.synced_window_mode = _env_bool("POKER44_SYNCED_WINDOW_MODE", True)
        self.sync_all_miners = _env_bool("POKER44_SYNC_ALL_MINERS", False)
        # Keep synchronized evaluation windows, but default to persistent scoring so
        # short outages and uneven request coverage do not reset miner rankings.
        self.sync_direct_score_update = _env_bool(
            "POKER44_SYNC_DIRECT_SCORE_UPDATE",
            False,
        )
        self.sync_reset_buffers_on_window_change = _env_bool(
            "POKER44_SYNC_RESET_BUFFERS_ON_WINDOW_CHANGE",
            False,
        )
        self.current_eval_window_id: Optional[int] = None
        self.coverage_round_index = int(getattr(self, "coverage_round_index", 0))
        self.coverage_round_expected_uids: List[int] = list(
            getattr(self, "coverage_round_expected_uids", [])
        )
        self.coverage_round_seen_uids: set[int] = set(
            getattr(self, "coverage_round_seen_uids", set())
        )
        self.coverage_round_reward_sums: Dict[int, float] = dict(
            getattr(self, "coverage_round_reward_sums", {})
        )
        self.coverage_round_reward_counts: Dict[int, int] = dict(
            getattr(self, "coverage_round_reward_counts", {})
        )
        self.coverage_round_pending_set_weights = bool(
            getattr(self, "coverage_round_pending_set_weights", False)
        )
        self.coverage_round_completed_at_step = getattr(
            self, "coverage_round_completed_at_step", None
        )
        self.prediction_buffer = {}
        self.label_buffer = {}
        state_dir = Path(self.config.neuron.full_path)
        self.model_manifest_path = state_dir / "model_manifests.json"
        self.compliance_registry_path = state_dir / "compliance_registry.json"
        self.suspicion_registry_path = state_dir / "suspicion_registry.json"
        self.served_chunk_registry_path = state_dir / "served_chunk_registry.json"
        self.model_manifest_registry = load_json_registry(self.model_manifest_path)
        if self.model_manifest_registry:
            self.model_manifest_registry = normalize_uid_key_registry(
                self.model_manifest_registry
            )
        self.compliance_registry = load_json_registry(
            self.compliance_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.suspicion_registry = load_json_registry(
            self.suspicion_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.served_chunk_registry = load_json_registry(
            self.served_chunk_registry_path,
            default={"chunk_index": {}, "recent_cycles": [], "summary": {}},
        )
        self.wandb_helper = ValidatorWandbHelper(
            config=cfg,
            validator_uid=self.resolve_uid(self.wallet.hotkey.ss58_address),
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            netuid=cfg.netuid,
        )
        self.wandb_helper.log_validator_startup(
            dataset_cfg=self.dataset_cfg,
            poll_interval=self.poll_interval,
            reward_window=self.reward_window,
            runtime_info=self.runtime_info,
        )
        bt.logging.info(
            "🪟 Validator sync mode | "
            f"synced_window_mode={self.synced_window_mode} "
            f"sync_all_miners={self.sync_all_miners} "
            f"direct_score_update={self.sync_direct_score_update} "
            f"reset_buffers_on_window_change={self.sync_reset_buffers_on_window_change}"
        )
        bt.logging.info(
            "🧾 Validator runtime | "
            f"uid={self.resolve_uid(self.wallet.hotkey.ss58_address)} "
            f"hotkey={self.wallet.hotkey.ss58_address} "
            f"version={__version__} "
            f"deploy_version={VALIDATOR_DEPLOY_VERSION} "
            f"git_branch={self.runtime_info.get('git_branch', '')} "
            f"git_commit={self.runtime_info.get('git_commit_short', '')} "
            f"git_dirty={self.runtime_info.get('git_dirty', False)}"
        )
        self._write_runtime_snapshot(status="started")

    def resolve_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    @property
    def runtime_snapshot_path(self) -> Path:
        return Path(self.config.neuron.full_path) / "validator_runtime.json"

    @property
    def network_snapshot_path(self) -> Path:
        return Path(self.config.neuron.full_path) / "network_snapshot.json"

    @property
    def runtime_info(self) -> dict:
        info = getattr(self, "_runtime_info", None)
        if info is None:
            info = collect_runtime_info()
            self._runtime_info = info
        return info

    def _write_runtime_snapshot(self, *, status: str, extra: Optional[dict] = None) -> None:
        payload = {
            "status": status,
            "validator_uid": self.resolve_uid(self.wallet.hotkey.ss58_address),
            "hotkey": self.wallet.hotkey.ss58_address,
            "version": __version__,
            "deploy_version": VALIDATOR_DEPLOY_VERSION,
            "netuid": self.config.netuid,
            "poll_interval": getattr(self, "poll_interval", None),
            "reward_window": getattr(self, "reward_window", None),
            "synced_window_mode": getattr(self, "synced_window_mode", None),
            "sync_all_miners": getattr(self, "sync_all_miners", None),
            "sync_direct_score_update": getattr(self, "sync_direct_score_update", None),
            "coverage_round_index": getattr(self, "coverage_round_index", None),
            "coverage_round_expected_count": len(getattr(self, "coverage_round_expected_uids", []) or []),
            "coverage_round_seen_count": len(getattr(self, "coverage_round_seen_uids", set()) or set()),
            "coverage_round_pending_set_weights": getattr(
                self, "coverage_round_pending_set_weights", None
            ),
            "runtime": self.runtime_info,
        }
        self.deploy_version = VALIDATOR_DEPLOY_VERSION
        if extra:
            payload.update(extra)
        write_runtime_snapshot(self.runtime_snapshot_path, payload)
        report_url = str(
            os.getenv(
                "POKER44_VALIDATOR_RUNTIME_REPORT_URL",
                DEFAULT_VALIDATOR_RUNTIME_REPORT_URL,
            )
        ).strip()
        if report_url:
            timeout_seconds = float(
                os.getenv("POKER44_VALIDATOR_RUNTIME_REPORT_TIMEOUT_SECONDS", "5")
            )
            signed_request = build_signed_runtime_request(
                wallet=self.wallet,
                url=report_url,
                payload=payload,
            )
            ok, message = post_runtime_snapshot(
                url=report_url,
                payload=payload,
                timeout_seconds=timeout_seconds,
                **signed_request,
            )
            if ok:
                bt.logging.debug(
                    f"Validator runtime snapshot reported successfully to collector: {report_url}"
                )
            else:
                bt.logging.warning(
                    "Validator runtime snapshot report failed | "
                    f"url={report_url} message={message}"
                )

        network_snapshot = collect_network_snapshot(self)
        write_runtime_snapshot(self.network_snapshot_path, network_snapshot)
        network_report_url = str(
            os.getenv(
                "POKER44_VALIDATOR_NETWORK_SNAPSHOT_REPORT_URL",
                DEFAULT_NETWORK_SNAPSHOT_REPORT_URL,
            )
        ).strip()
        if network_report_url:
            timeout_seconds = float(
                os.getenv("POKER44_VALIDATOR_NETWORK_SNAPSHOT_TIMEOUT_SECONDS", "5")
            )
            signed_request = build_signed_runtime_request(
                wallet=self.wallet,
                url=network_report_url,
                payload=network_snapshot,
            )
            ok, message = post_runtime_snapshot(
                url=network_report_url,
                payload=network_snapshot,
                timeout_seconds=timeout_seconds,
                **signed_request,
            )
            if ok:
                bt.logging.debug(
                    f"Validator network snapshot reported successfully to collector: {network_report_url}"
                )
            else:
                bt.logging.warning(
                    "Validator network snapshot report failed | "
                    f"url={network_report_url} message={message}"
                )

    async def forward(self, synapse=None):  # type: ignore[override]
        return await forward_cycle(self)

    def begin_coverage_round(self, expected_uids: List[int], *, reason: str) -> None:
        ordered_uids = sorted(int(uid) for uid in expected_uids)
        self.coverage_round_index += 1
        self.coverage_round_expected_uids = ordered_uids
        self.coverage_round_seen_uids = set()
        self.coverage_round_reward_sums = {uid: 0.0 for uid in ordered_uids}
        self.coverage_round_reward_counts = {uid: 0 for uid in ordered_uids}
        self.coverage_round_pending_set_weights = False
        self.coverage_round_completed_at_step = None
        bt.logging.info(
            f"Started coverage round #{self.coverage_round_index} with "
            f"{len(ordered_uids)} eligible miner(s) ({reason})."
        )

    def ensure_coverage_round(self, expected_uids: List[int], *, reason: str) -> None:
        normalized_expected = sorted(int(uid) for uid in expected_uids)
        if not self.coverage_round_expected_uids:
            self.begin_coverage_round(normalized_expected, reason=reason)
            return
        previous_expected = set(self.coverage_round_expected_uids)
        current_expected = set(normalized_expected)
        if previous_expected == current_expected:
            self.coverage_round_expected_uids = normalized_expected
            return

        added = sorted(current_expected - previous_expected)
        removed = sorted(previous_expected - current_expected)

        self.coverage_round_expected_uids = normalized_expected
        self.coverage_round_seen_uids &= current_expected
        self.coverage_round_reward_sums = {
            uid: float(self.coverage_round_reward_sums.get(uid, 0.0))
            for uid in normalized_expected
        }
        self.coverage_round_reward_counts = {
            uid: int(self.coverage_round_reward_counts.get(uid, 0))
            for uid in normalized_expected
        }

        bt.logging.info(
            f"Coverage round #{self.coverage_round_index} reconciled eligible set "
            f"(+{len(added)} / -{len(removed)}) ({reason})."
        )

        if current_expected and len(self.coverage_round_seen_uids) >= len(current_expected):
            self.coverage_round_pending_set_weights = True
            self.coverage_round_completed_at_step = int(getattr(self, "step", 0))
            bt.logging.info(
                f"Coverage round #{self.coverage_round_index} complete after eligible-set reconciliation; "
                "weights are now eligible to publish."
            )

    def record_round_cycle(self, *, sampled_uids: List[int], reward_map: Dict[int, float]) -> None:
        if self.coverage_round_pending_set_weights:
            return

        for uid in sampled_uids:
            normalized_uid = int(uid)
            if normalized_uid not in self.coverage_round_reward_sums:
                continue
            self.coverage_round_seen_uids.add(normalized_uid)
            self.coverage_round_reward_sums[normalized_uid] += float(
                reward_map.get(normalized_uid, 0.0)
            )
            self.coverage_round_reward_counts[normalized_uid] += 1

        seen = len(self.coverage_round_seen_uids)
        expected = len(self.coverage_round_expected_uids)
        bt.logging.info(
            f"Coverage round #{self.coverage_round_index}: seen "
            f"{seen}/{expected} eligible miner(s)."
        )

        if expected > 0 and seen >= expected:
            self.coverage_round_pending_set_weights = True
            self.coverage_round_completed_at_step = int(getattr(self, "step", 0))
            bt.logging.info(
                f"Coverage round #{self.coverage_round_index} complete; "
                "weights are now eligible to publish."
            )

    def build_scores_from_coverage_round(self) -> np.ndarray:
        round_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        for uid in self.coverage_round_expected_uids:
            count = int(self.coverage_round_reward_counts.get(uid, 0))
            if count <= 0:
                continue
            avg_reward = float(self.coverage_round_reward_sums.get(uid, 0.0)) / float(count)
            round_scores[int(uid)] = max(0.0, avg_reward)
        return round_scores

    def should_set_weights(self) -> bool:  # type: ignore[override]
        if not self.coverage_round_pending_set_weights:
            return False
        return super().should_set_weights()

    def set_weights(self) -> bool:  # type: ignore[override]
        if not self.coverage_round_pending_set_weights:
            bt.logging.info(
                f"Skipping set_weights: coverage round #{self.coverage_round_index} "
                "has not yet covered all eligible miners."
            )
            return False

        self.scores = self.build_scores_from_coverage_round()
        success = super().set_weights()
        if success:
            self.begin_coverage_round(
                self.coverage_round_expected_uids,
                reason="previous round published on-chain",
            )
        return success

    def __del__(self) -> None:
        wandb_helper = getattr(self, "wandb_helper", None)
        if wandb_helper is not None:
            try:
                wandb_helper.finish()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover - manual execution
    validator = Validator()
    validator.run()
