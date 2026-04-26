from __future__ import annotations

import math
from typing import Any

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    average_precision_score = None
    balanced_accuracy_score = None
    brier_score_loss = None
    log_loss = None
    precision_recall_curve = None
    roc_auc_score = None


def _require_sklearn() -> None:
    if any(
        metric is None
        for metric in (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            log_loss,
            precision_recall_curve,
            roc_auc_score,
        )
    ):
        raise RuntimeError("scikit-learn is required to compute evaluation metrics.")


def balanced_accuracy_at_threshold(
    y_true: list[int],
    y_prob: list[float],
    threshold: float = 0.5,
) -> float:
    _require_sklearn()
    predicted = [1 if float(prob) >= threshold else 0 for prob in y_prob]
    return float(balanced_accuracy_score([int(label) for label in y_true], predicted))


def best_balanced_accuracy_threshold(
    y_true: list[int],
    y_prob: list[float],
) -> dict[str, float]:
    _require_sklearn()
    if not y_prob:
        return {"threshold": 0.5, "balanced_accuracy": 0.0}

    candidate_thresholds = sorted({0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95} | {float(prob) for prob in y_prob})
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        score = balanced_accuracy_at_threshold(y_true, y_prob, threshold=threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "balanced_accuracy": best_score}


def false_positive_rate_at_threshold(
    y_true: list[int],
    y_prob: list[float],
    threshold: float = 0.5,
) -> float:
    negative_count = sum(1 for label in y_true if int(label) == 0)
    if negative_count == 0:
        return 0.0
    false_positives = sum(
        1
        for label, prob in zip(y_true, y_prob)
        if int(label) == 0 and float(prob) >= threshold
    )
    return false_positives / negative_count


def false_positive_rate_at_recall(
    y_true: list[int],
    y_prob: list[float],
    target_recall: float = 0.9,
) -> dict[str, float]:
    _require_sklearn()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds_list = list(thresholds)
    recall_list = list(recall[:-1])
    if not thresholds_list:
        return {
            "threshold": 0.5,
            "recall": 0.0,
            "fpr": false_positive_rate_at_threshold(y_true, y_prob),
        }

    best_index = min(
        range(len(thresholds_list)),
        key=lambda index: abs(float(recall_list[index]) - target_recall),
    )
    threshold = float(thresholds_list[best_index])
    predicted_positive = [float(prob) >= threshold for prob in y_prob]

    positive_count = max(sum(1 for label in y_true if int(label) == 1), 1)
    negative_count = max(sum(1 for label in y_true if int(label) == 0), 1)
    true_positives = sum(
        1
        for label, pred in zip(y_true, predicted_positive)
        if int(label) == 1 and pred
    )
    false_positives = sum(
        1
        for label, pred in zip(y_true, predicted_positive)
        if int(label) == 0 and pred
    )
    return {
        "threshold": threshold,
        "recall": true_positives / positive_count,
        "fpr": false_positives / negative_count,
    }


def evaluate_predictions(
    y_true: list[int],
    y_prob: list[float],
    latency_per_chunk_ms: float | None = None,
    recall_target: float = 0.9,
) -> dict[str, float]:
    _require_sklearn()
    clipped_prob = [min(max(float(prob), 1e-6), 1.0 - 1e-6) for prob in y_prob]
    labels = [int(label) for label in y_true]

    recall_stats = false_positive_rate_at_recall(
        y_true=labels,
        y_prob=clipped_prob,
        target_recall=recall_target,
    )
    best_balanced = best_balanced_accuracy_threshold(labels, clipped_prob)
    metrics = {
        "roc_auc": float(roc_auc_score(labels, clipped_prob)),
        "pr_auc": float(average_precision_score(labels, clipped_prob)),
        "log_loss": float(log_loss(labels, clipped_prob)),
        "brier_score": float(brier_score_loss(labels, clipped_prob)),
        "balanced_accuracy_0_5": float(balanced_accuracy_at_threshold(labels, clipped_prob, threshold=0.5)),
        "balanced_accuracy_best": float(best_balanced["balanced_accuracy"]),
        "threshold_best_balanced_accuracy": float(best_balanced["threshold"]),
        "fpr_at_recall": float(recall_stats["fpr"]),
        "recall_target": float(recall_target),
        "threshold_at_recall": float(recall_stats["threshold"]),
        "achieved_recall": float(recall_stats["recall"]),
        "fpr_at_threshold_0_5": float(false_positive_rate_at_threshold(labels, clipped_prob, threshold=0.5)),
    }
    if latency_per_chunk_ms is not None and not math.isnan(latency_per_chunk_ms):
        metrics["latency_per_chunk_ms"] = float(latency_per_chunk_ms)
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    ordered_keys = (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "balanced_accuracy_0_5",
        "balanced_accuracy_best",
        "threshold_best_balanced_accuracy",
        "fpr_at_recall",
        "achieved_recall",
        "threshold_at_recall",
        "fpr_at_threshold_0_5",
        "latency_per_chunk_ms",
    )
    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")
    return " | ".join(parts)
