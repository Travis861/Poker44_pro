"""Public benchmark dataset builder for miner training/reference artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hands_generator.mixed_dataset_provider import (
    DEFAULT_HUMAN_JSON_PATH,
    _best_single_rule_accuracy,
    _chunk_features_for_shortcut_rule,
    build_mixed_labeled_chunks,
    MixedDatasetConfig,
)
from poker44.validator.sanitization import sanitize_hand_for_miner, sanitized_chunk_signature


DEFAULT_PUBLIC_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "public_miner_benchmark.json.gz"
)


@dataclass
class PublicBenchmarkConfig:
    human_json_path: Path = DEFAULT_HUMAN_JSON_PATH
    output_path: Path = DEFAULT_PUBLIC_BENCHMARK_PATH
    chunk_count: int = 40
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    seed: int = 44
    validation_ratio: float = 0.25


def _compute_payload_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assign_split(index: int, total: int, validation_ratio: float) -> str:
    validation_cutoff = max(1, int(round(total * validation_ratio)))
    return "validation" if index < validation_cutoff else "train"


def _sanitize_chunk(hands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [sanitize_hand_for_miner(hand) for hand in hands]


def _public_chunk_stats(
    labeled_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    shortcut_acc, shortcut_rule = _best_single_rule_accuracy(labeled_chunks)
    signatures = [sanitized_chunk_signature(chunk["hands"]) for chunk in labeled_chunks]
    avg_signature = [0.0] * 9
    if signatures:
        for sig in signatures:
            for idx, value in enumerate(sig):
                avg_signature[idx] += float(value)
        avg_signature = [value / len(signatures) for value in avg_signature]

    train_chunks = sum(1 for chunk in labeled_chunks if chunk.get("split") == "train")
    validation_chunks = sum(
        1 for chunk in labeled_chunks if chunk.get("split") == "validation"
    )
    bot_chunks = sum(1 for chunk in labeled_chunks if chunk.get("is_bot", False))
    human_chunks = len(labeled_chunks) - bot_chunks
    total_hands = sum(len(chunk.get("hands", [])) for chunk in labeled_chunks)

    return {
        "chunk_count": len(labeled_chunks),
        "train_chunks": train_chunks,
        "validation_chunks": validation_chunks,
        "human_chunks": human_chunks,
        "bot_chunks": bot_chunks,
        "total_hands": total_hands,
        "shortcut_rule_accuracy": shortcut_acc,
        "shortcut_rule": shortcut_rule,
        "avg_signature": {
            "calls": avg_signature[0],
            "checks": avg_signature[1],
            "raises": avg_signature[2],
            "folds": avg_signature[3],
            "actions": avg_signature[4],
            "streets": avg_signature[5],
            "players": avg_signature[6],
            "action_amount": avg_signature[7],
            "pot_after": avg_signature[8],
        },
        "feature_snapshot_example": (
            _chunk_features_for_shortcut_rule(labeled_chunks[0]["hands"])
            if labeled_chunks
            else {}
        ),
    }


def build_public_benchmark(
    cfg: PublicBenchmarkConfig,
) -> Tuple[Dict[str, Any], str]:
    mixed_cfg = MixedDatasetConfig(
        human_json_path=cfg.human_json_path,
        output_path=cfg.output_path,
        chunk_count=int(cfg.chunk_count),
        min_hands_per_chunk=int(cfg.min_hands_per_chunk),
        max_hands_per_chunk=int(cfg.max_hands_per_chunk),
        human_ratio=float(cfg.human_ratio),
        refresh_seconds=3600,
        seed=int(cfg.seed),
    )
    mixed_chunks, _, mixed_stats = build_mixed_labeled_chunks(mixed_cfg, window_id=int(cfg.seed))
    labeled_chunks: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(mixed_chunks):
        labeled_chunks.append(
            {
                "chunk_id": f"chunk_{idx + 1}",
                "split": _assign_split(
                    idx,
                    len(mixed_chunks),
                    float(cfg.validation_ratio),
                ),
                "is_bot": bool(chunk.get("is_bot", False)),
                "hands": _sanitize_chunk(chunk.get("hands", [])),
            }
        )

    stats = _public_chunk_stats(labeled_chunks)
    stats["source_shortcut_rule_accuracy"] = mixed_stats.get("shortcut_rule_accuracy", 0.0)
    payload = {
        "version": 1,
        "source": "public_corpus_only",
        "description": (
            "Public miner benchmark built only from the repo public human corpus and "
            "offline-generated bot chunks. No validator-private human data is included."
        ),
        "config": {
            "chunk_count": int(cfg.chunk_count),
            "min_hands_per_chunk": int(cfg.min_hands_per_chunk),
            "max_hands_per_chunk": int(cfg.max_hands_per_chunk),
            "human_ratio": float(cfg.human_ratio),
            "seed": int(cfg.seed),
            "validation_ratio": float(cfg.validation_ratio),
            "human_json_path": str(cfg.human_json_path),
        },
        "stats": stats,
        "labeled_chunks": labeled_chunks,
    }
    dataset_hash = _compute_payload_hash(payload)
    payload["dataset_hash"] = dataset_hash
    return payload, dataset_hash


def save_public_benchmark(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wb") as handle:
            handle.write(encoded)
    else:
        output_path.write_bytes(encoded)
