from __future__ import annotations

import argparse
from pathlib import Path

from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    DEFAULT_BENCHMARK_PATHS,
    DEFAULT_BOT_PATHS,
    DEFAULT_HUMAN_PATHS,
    build_training_dataframe,
    load_json_or_gz,
    load_public_benchmark_rows,
    resolve_existing_path,
)
from training.evaluate import evaluate_predictions, format_metrics

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    CalibratedClassifierCV = None
    RandomForestClassifier = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random-Forest-first Poker44 miner model tuned for balanced accuracy."
    )
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--bot-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--max-features",
        choices=("sqrt", "log2", "all"),
        default="sqrt",
    )
    parser.add_argument("--benchmark-weight", type=float, default=2.0)
    parser.add_argument(
        "--calibration",
        choices=("auto", "isotonic", "sigmoid", "none"),
        default="auto",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("balanced", "low_fpr"),
        default="balanced",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_xgb_calibrated.joblib"),
    )
    return parser.parse_args()


def choose_calibration(method: str, train_size: int) -> str | None:
    if method == "none":
        return None
    if method == "auto":
        return "isotonic" if train_size >= 1200 else "sigmoid"
    return method


def model_selection_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced":
        return (
            0.45 * metrics["balanced_accuracy_best"]
            + 0.20 * metrics["balanced_accuracy_0_5"]
            + 0.20 * metrics["roc_auc"]
            + 0.10 * metrics["pr_auc"]
            - 0.15 * metrics["log_loss"]
            - 0.10 * metrics["brier_score"]
            - 0.15 * metrics["fpr_at_threshold_0_5"]
            - 0.10 * metrics["fpr_at_recall"]
        )
    return (
        0.25 * metrics["balanced_accuracy_best"]
        + 0.20 * metrics["roc_auc"]
        + 0.10 * metrics["pr_auc"]
        - 0.15 * metrics["log_loss"]
        - 0.10 * metrics["brier_score"]
        - 0.60 * metrics["fpr_at_threshold_0_5"]
        - 0.80 * metrics["fpr_at_recall"]
    )


def train_model(args: argparse.Namespace) -> tuple[object, list[str], dict[str, float]]:
    if joblib is None:
        raise RuntimeError(
            "Training dependencies are missing. Install scikit-learn and joblib first."
        )
    if CalibratedClassifierCV is None or RandomForestClassifier is None or train_test_split is None:
        raise RuntimeError("scikit-learn is required to train and calibrate the miner model.")

    human_path = resolve_existing_path(args.human_path, DEFAULT_HUMAN_PATHS)
    bot_path = resolve_existing_path(args.bot_path, DEFAULT_BOT_PATHS)
    human_hands = load_json_or_gz(human_path)
    bot_hands = load_json_or_gz(bot_path)
    benchmark_path = None
    try:
        benchmark_path = resolve_existing_path(args.benchmark_path, DEFAULT_BENCHMARK_PATHS)
    except FileNotFoundError:
        benchmark_path = None

    raw_rows = build_training_dataframe(
        human_hands=human_hands,
        bot_hands=bot_hands,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        stride=args.stride,
        repeats=args.repeats,
        seed=args.seed,
    )
    benchmark_train_rows: list[dict[str, float]] = []
    benchmark_validation_rows: list[dict[str, float]] = []
    if benchmark_path is not None:
        benchmark_train_rows = load_public_benchmark_rows(benchmark_path, split_filter="train")
        benchmark_validation_rows = load_public_benchmark_rows(
            benchmark_path, split_filter="validation"
        )

    rows = list(raw_rows) + list(benchmark_train_rows)
    if not rows:
        raise RuntimeError("Training dataframe is empty. Verify your human/bot hand sources.")

    feature_names = sorted(key for key in rows[0].keys() if key != "label")
    all_X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    all_y = [int(row["label"]) for row in rows]
    all_weights = [
        float(args.benchmark_weight) if index >= len(raw_rows) else 1.0
        for index in range(len(rows))
    ]

    if benchmark_validation_rows:
        X_train = all_X
        y_train = all_y
        train_weights = all_weights
        X_test = [
            [float(row.get(name, 0.0)) for name in feature_names]
            for row in benchmark_validation_rows
        ]
        y_test = [int(row["label"]) for row in benchmark_validation_rows]
    else:
        X_train, X_test, y_train, y_test, train_weights, _ = train_test_split(
            all_X,
            all_y,
            all_weights,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=all_y,
        )

    requested_calibration = choose_calibration(args.calibration, len(X_train))
    candidate_calibrations = (
        [requested_calibration]
        if args.calibration != "auto"
        else ["isotonic", "sigmoid", None]
    )
    selected_max_features = None if args.max_features == "all" else args.max_features
    candidate_configs = [
        {
            "n_estimators": max(700, args.n_estimators),
            "max_depth": args.max_depth if args.max_depth > 0 else None,
            "min_samples_leaf": max(1, args.min_samples_leaf),
            "max_features": selected_max_features,
        },
        {
            "n_estimators": max(900, args.n_estimators + 200),
            "max_depth": None,
            "min_samples_leaf": max(1, args.min_samples_leaf),
            "max_features": selected_max_features,
        },
        {
            "n_estimators": max(1100, args.n_estimators + 400),
            "max_depth": None,
            "min_samples_leaf": max(1, args.min_samples_leaf),
            "max_features": "log2" if selected_max_features != "log2" else selected_max_features,
        },
        {
            "n_estimators": max(1200, args.n_estimators + 500),
            "max_depth": None,
            "min_samples_leaf": max(2, args.min_samples_leaf + 1),
            "max_features": None,
        },
    ]

    best_model = None
    best_metrics = None
    best_calibration = None
    best_config = None
    best_selection_score = float("-inf")

    for config_index, config in enumerate(candidate_configs):
        base_model = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            class_weight="balanced_subsample",
            bootstrap=True,
            random_state=args.seed + config_index,
            n_jobs=-1,
        )

        for calibration_method in candidate_calibrations:
            candidate_model = base_model
            if calibration_method is not None:
                candidate_model = CalibratedClassifierCV(base_model, method=calibration_method, cv=3)

            candidate_model.fit(X_train, y_train, sample_weight=train_weights)
            candidate_probs = candidate_model.predict_proba(X_test)[:, 1]
            candidate_metrics = evaluate_predictions(y_true=y_test, y_prob=candidate_probs)
            candidate_score = model_selection_score(candidate_metrics, args.selection_objective)

            print(
                "candidate",
                f"rf_variant={config_index + 1}",
                f"n_estimators={config['n_estimators']}",
                f"max_depth={config['max_depth']}",
                f"min_samples_leaf={config['min_samples_leaf']}",
                f"max_features={config['max_features'] if config['max_features'] is not None else 'all'}",
                f"calibration={calibration_method or 'none'}",
                f"selection_score={candidate_score:.6f}",
                format_metrics(candidate_metrics),
            )

            if candidate_score > best_selection_score:
                best_selection_score = candidate_score
                best_model = candidate_model
                best_metrics = candidate_metrics
                best_calibration = calibration_method
                best_config = dict(config)

    artifact_meta = {
        "chunk_size": float(args.chunk_size),
        "min_chunk_size": float(args.min_chunk_size),
        "stride": float(args.stride),
        "repeats": float(args.repeats),
        "n_estimators": float(args.n_estimators),
        "max_depth": float(args.max_depth),
        "min_samples_leaf": float(args.min_samples_leaf),
        "benchmark_weight": float(args.benchmark_weight),
        "calibration": best_calibration or "none",
        "selection_objective": args.selection_objective,
        "framework": "sklearn-random-forest+calibration",
        "selected_n_estimators": float((best_config or {}).get("n_estimators", args.n_estimators)),
        "selected_max_depth": float((best_config or {}).get("max_depth") or 0.0),
        "selected_min_samples_leaf": float(
            (best_config or {}).get("min_samples_leaf", args.min_samples_leaf)
        ),
        "selected_max_features": str((best_config or {}).get("max_features") or "all"),
        "human_path": str(human_path),
        "bot_path": str(bot_path),
        "benchmark_path": str(benchmark_path) if benchmark_path is not None else "",
        "benchmark_train_rows": float(len(benchmark_train_rows)),
        "benchmark_validation_rows": float(len(benchmark_validation_rows)),
        "raw_rows": float(len(raw_rows)),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_names,
            "metadata": artifact_meta,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency(
        [human_hands[: args.chunk_size], bot_hands[: args.chunk_size]]
    )
    metrics = dict(best_metrics or {})
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return best_model, feature_names, metrics


def main() -> None:
    args = parse_args()
    _, feature_names, metrics = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
