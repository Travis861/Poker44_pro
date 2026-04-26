from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent / "Poker44-subnet"
DEFAULT_HUMAN_PATHS = (
    REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz",
    PROJECT_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz",
)
DEFAULT_BOT_PATHS = (
    REPO_ROOT / "data" / "generated_bot_hands.json",
    PROJECT_ROOT / "data" / "generated_bot_hands.json",
)
DEFAULT_BENCHMARK_PATHS = (
    REPO_ROOT / "data" / "public_miner_benchmark.json.gz",
    PROJECT_ROOT / "data" / "public_miner_benchmark.json.gz",
)


def load_json_or_gz(path: str | Path) -> Any:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def extract_rows_from_labeled_chunks(
    labeled_chunks: list[dict[str, Any]],
    *,
    split_filter: str | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for chunk in labeled_chunks:
        if split_filter is not None and chunk.get("split") != split_filter:
            continue
        hands = chunk.get("hands") or []
        if not hands:
            continue
        features = chunk_features(hands)
        features["label"] = 1.0 if bool(chunk.get("is_bot", False)) else 0.0
        features["hand_count"] = float(len(hands))
        rows.append(features)
    return rows


def load_public_benchmark_rows(
    benchmark_path: str | Path,
    *,
    split_filter: str | None = None,
) -> list[dict[str, float]]:
    payload = load_json_or_gz(benchmark_path)
    labeled_chunks = payload.get("labeled_chunks", []) if isinstance(payload, dict) else []
    return extract_rows_from_labeled_chunks(labeled_chunks, split_filter=split_filter)


def resolve_existing_path(preferred: str | Path | None, fallbacks: tuple[Path, ...]) -> Path:
    candidates = [Path(preferred)] if preferred else []
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate paths exist: {joined}")


def build_chunks(
    hands: list[dict[str, Any]],
    label: int,
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    seed: int = 42,
    stride: int | None = None,
    repeats: int = 1,
) -> list[dict[str, float]]:
    minimum = min_chunk_size if min_chunk_size is not None else max(20, chunk_size // 2)
    stride = stride if stride is not None else max(8, chunk_size // 2)
    rows: list[dict[str, float]] = []
    seen_signatures: set[tuple[int, int, int]] = set()
    for repeat in range(max(1, repeats)):
        rng = random.Random(seed + repeat)
        shuffled = list(hands)
        rng.shuffle(shuffled)
        last_start = max(1, len(shuffled) - minimum + 1)
        for index in range(0, last_start, stride):
            chunk = shuffled[index : index + chunk_size]
            if len(chunk) < minimum:
                continue
            signature = (repeat, index, len(chunk))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            features = chunk_features(chunk)
            features["label"] = float(label)
            features["hand_count"] = float(len(chunk))
            rows.append(features)
    return rows


def build_training_dataframe(
    human_hands: list[dict[str, Any]],
    bot_hands: list[dict[str, Any]],
    chunk_size: int = 80,
    min_chunk_size: int | None = None,
    seed: int = 42,
    stride: int | None = None,
    repeats: int = 3,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    rows.extend(
        build_chunks(
            human_hands,
            label=0,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            seed=seed,
            stride=stride,
            repeats=repeats,
        )
    )
    rows.extend(
        build_chunks(
            bot_hands,
            label=1,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            seed=seed + 1,
            stride=stride,
            repeats=repeats,
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chunk-level Poker44 training dataset.")
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--bot-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "training_chunks.csv"))
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_path = resolve_existing_path(args.human_path, DEFAULT_HUMAN_PATHS)
    bot_path = resolve_existing_path(args.bot_path, DEFAULT_BOT_PATHS)

    human_hands = load_json_or_gz(human_path)
    bot_hands = load_json_or_gz(bot_path)
    df = build_training_dataframe(
        human_hands=human_hands,
        bot_hands=bot_hands,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        stride=args.stride,
        repeats=args.repeats,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(df, handle)
    else:
        if not df:
            raise RuntimeError("No rows were generated for dataset export.")
        fieldnames = sorted(df[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(df)

    print(f"Saved {len(df)} chunk rows to {output_path}")
    print(f"Human source: {human_path}")
    print(f"Bot source:   {bot_path}")


if __name__ == "__main__":
    main()
