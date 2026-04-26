import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from poker44_ml.inference import Poker44Model


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info("Poker44 ML miner started")
        self.max_hands_per_chunk_eval = max(
            0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
        )
        self.query_log_preview = os.getenv("POKER44_LOG_QUERY_PREVIEW", "0") == "1"

        repo_root = Path(__file__).resolve().parents[1]
        model_path = repo_root / "models" / "poker44_xgb_calibrated.joblib"
        self.predictor = Poker44Model(str(model_path))

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44-xgb-calibrated",
                "model_version": "1",
                "framework": self.predictor.metadata.get("framework", "xgboost+sklearn"),
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": "Chunk-level tabular model with calibrated probabilities.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public human corpus plus offline-generated bot hands."
                ),
                "training_data_sources": ["public_human_corpus", "generated_bot_hands"],
                "private_data_attestation": (
                    "This miner does not train on validator-private human data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root=repo_root, model_path=model_path)

    def _log_manifest_startup(self, repo_root: Path, model_path: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(f"Loaded model from {model_path}")
        bt.logging.info(
            f"Model metadata: feature_count={len(self.predictor.feature_names)} "
            f"calibration={self.predictor.metadata.get('calibration', 'unknown')} "
            f"framework={self.predictor.metadata.get('framework', 'unknown')}"
        )
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )
        bt.logging.info(
            "Public benchmark command: "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb"
        )
        bt.logging.info(
            "Purpose: train, validate and refine miner models against the public benchmark "
            "while Poker44 moves toward more dynamic evaluation."
        )
        bt.logging.info(
            f"Fast inference settings | max_hands_per_chunk_eval={self.max_hands_per_chunk_eval} "
            f"query_log_preview={self.query_log_preview}"
        )

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _compress_chunk(self, chunk: list[dict]) -> list[dict]:
        if self.max_hands_per_chunk_eval <= 0 or len(chunk) <= self.max_hands_per_chunk_eval:
            return chunk
        if self.max_hands_per_chunk_eval == 1:
            return [chunk[len(chunk) // 2]]

        last_index = len(chunk) - 1
        slots = self.max_hands_per_chunk_eval - 1
        indices = {
            min(last_index, round(i * last_index / slots))
            for i in range(self.max_hands_per_chunk_eval)
        }
        return [chunk[index] for index in sorted(indices)]

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = list(synapse.chunks or [])
        caller = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        chunk_sizes = [len(chunk or []) for chunk in chunks]
        eval_chunks = [self._compress_chunk(list(chunk or [])) for chunk in chunks]
        eval_chunk_sizes = [len(chunk) for chunk in eval_chunks]
        bt.logging.info(
            f"Received validator query | caller={caller} "
            f"chunk_count={len(chunks)} "
            f"chunk_size_range={[min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else [0, 0]} "
            f"eval_chunk_size_range={[min(eval_chunk_sizes), max(eval_chunk_sizes)] if eval_chunk_sizes else [0, 0]}"
        )
        started = time.perf_counter()
        try:
            raw_scores = self.predictor.predict_chunk_scores(eval_chunks)
        except Exception as err:
            bt.logging.error(f"Predictor failure | caller={caller} error={err}")
            raw_scores = [0.5] * len(chunks)

        scores = [round(self._clamp_score(score), 6) for score in raw_scores[: len(chunks)]]
        if len(scores) < len(chunks):
            deficit = len(chunks) - len(scores)
            bt.logging.warning(
                f"Score count mismatch | caller={caller} expected={len(chunks)} got={len(scores)} "
                f"padding_with=0.5 count={deficit}"
            )
            scores.extend([0.5] * deficit)
        elif len(raw_scores) > len(chunks):
            bt.logging.warning(
                f"Score count mismatch | caller={caller} expected={len(chunks)} got={len(raw_scores)} "
                "truncating extras"
            )

        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_chunk_ms = elapsed_ms / max(len(chunks), 1)
        message = (
            f"Scored {len(chunks)} chunks in {elapsed_ms:.2f} ms "
            f"({per_chunk_ms:.2f} ms/chunk) "
            f"score_range={[min(scores), max(scores)] if scores else [0.0, 0.0]}"
        )
        if self.query_log_preview:
            message += (
                f" score_preview={scores[:5]} "
                f"prediction_preview={synapse.predictions[:5]}"
            )
        bt.logging.info(message)
        bt.logging.success(
            f"Validator response sent successfully | caller={caller} "
            f"chunk_count={len(chunks)} "
            f"response_count={len(scores)} "
            f"elapsed_ms={elapsed_ms:.2f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        blocked, reason = self.common_blacklist(synapse)
        caller = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        if blocked:
            bt.logging.warning(f"Blacklisted request | caller={caller} reason={reason}")
        return blocked, reason

    async def priority(self, synapse: DetectionSynapse) -> float:
        priority = self.caller_priority(synapse)
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("ML miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
