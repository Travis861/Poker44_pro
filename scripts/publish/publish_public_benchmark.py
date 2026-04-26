#!/usr/bin/env python3
"""Build and optionally publish a public miner benchmark artifact to W&B."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hands_generator.public_benchmark import (
    DEFAULT_PUBLIC_BENCHMARK_PATH,
    DEFAULT_HUMAN_JSON_PATH,
    PublicBenchmarkConfig,
    build_public_benchmark,
    save_public_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-json-path", type=Path, default=DEFAULT_HUMAN_JSON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PUBLIC_BENCHMARK_PATH)
    parser.add_argument("--chunk-count", type=int, default=40)
    parser.add_argument("--min-hands-per-chunk", type=int, default=60)
    parser.add_argument("--max-hands-per-chunk", type=int, default=120)
    parser.add_argument("--human-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--wandb-project", type=str, default="poker44-miner-benchmarks")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--artifact-name", type=str, default="public-miner-benchmark")
    parser.add_argument("--artifact-type", type=str, default="dataset")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PublicBenchmarkConfig(
        human_json_path=args.human_json_path,
        output_path=args.output_path,
        chunk_count=args.chunk_count,
        min_hands_per_chunk=args.min_hands_per_chunk,
        max_hands_per_chunk=args.max_hands_per_chunk,
        human_ratio=args.human_ratio,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    payload, dataset_hash = build_public_benchmark(cfg)
    save_public_benchmark(cfg.output_path, payload)
    print(f"saved={cfg.output_path}")
    print(f"dataset_hash={dataset_hash}")

    if args.skip_wandb:
        return

    import wandb

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_QUIET", "true")

    init_kwargs = {
        "project": args.wandb_project,
        "job_type": "publish_public_benchmark",
        "config": payload["config"],
        "notes": payload["description"],
        "settings": wandb.Settings(quiet=True),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    try:
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=payload["description"],
            metadata={
                "dataset_hash": dataset_hash,
                "source": payload["source"],
                **payload["stats"],
            },
        )
        artifact.add_file(str(cfg.output_path), name=cfg.output_path.name)
        run.log_artifact(artifact)
        run.log(
            {
                "public_benchmark/dataset_hash": dataset_hash,
                "public_benchmark/chunk_count": payload["stats"]["chunk_count"],
                "public_benchmark/train_chunks": payload["stats"]["train_chunks"],
                "public_benchmark/validation_chunks": payload["stats"]["validation_chunks"],
                "public_benchmark/shortcut_rule_accuracy": payload["stats"]["shortcut_rule_accuracy"],
            }
        )
    finally:
        run.finish(quiet=True)


if __name__ == "__main__":
    main()
