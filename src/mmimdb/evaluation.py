"""Multilabel evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from mmimdb.constants import GENRE_LABELS


def threshold_to_array(threshold: float | list[float] | np.ndarray, num_labels: int) -> np.ndarray:
    arr = np.asarray(threshold, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(num_labels, float(arr), dtype=np.float32)
    if arr.shape != (num_labels,):
        raise ValueError(f"Expected scalar threshold or shape ({num_labels},), got {arr.shape}.")
    return arr


def threshold_to_serializable(threshold: float | list[float] | np.ndarray) -> float | list[float]:
    arr = np.asarray(threshold, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return [float(v) for v in arr.reshape(-1)]


def threshold_predictions(y_prob: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    threshold_arr = threshold_to_array(threshold, y_prob.shape[1])
    return (y_prob >= threshold_arr.reshape(1, -1)).astype(np.int8)


def predicted_label_indices(
    probabilities: np.ndarray,
    threshold: float | list[float] | np.ndarray,
    ensure_at_least_one: bool = True,
) -> list[int]:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    threshold_arr = threshold_to_array(threshold, probs.shape[0])
    indices = np.flatnonzero(probs >= threshold_arr).astype(int).tolist()
    if ensure_at_least_one and not indices:
        indices = [int(np.argmax(probs))]
    return sorted(indices, key=lambda i: float(probs[i]), reverse=True)


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    labels: list[str] | None = None,
) -> dict:
    labels = labels or GENRE_LABELS
    y_pred = threshold_predictions(y_prob, threshold)
    metrics = {
        "threshold": threshold_to_serializable(threshold),
        "sample_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_label"] = {
        label: {
            "f1": float(per_label_f1[i]),
            "precision": float(per_label_precision[i]),
            "recall": float(per_label_recall[i]),
            "support": int(y_true[:, i].sum()),
        }
        for i, label in enumerate(labels)
    }
    return metrics


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if metric != "f1":
        raise ValueError("Per-label threshold tuning currently supports metric='f1'.")

    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    best_thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for label_i in range(y_true.shape[1]):
        y_label = y_true[:, label_i]
        best_score = f1_score(y_label, y_prob[:, label_i] >= 0.5, zero_division=0)
        for threshold in thresholds:
            score = f1_score(y_label, y_prob[:, label_i] >= float(threshold), zero_division=0)
            if score > best_score:
                best_score = score
                best_thresholds[label_i] = float(threshold)
    return best_thresholds, multilabel_metrics(y_true, y_prob, threshold=best_thresholds)


def tune_global_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "macro_f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_metrics = multilabel_metrics(y_true, y_prob, best_threshold)
    best_score = best_metrics[metric]
    for threshold in thresholds:
        metrics = multilabel_metrics(y_true, y_prob, float(threshold))
        score = metrics[metric]
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
            best_metrics = metrics
    return best_threshold, best_metrics


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "macro_f1",
    strategy: str = "global",
    thresholds: np.ndarray | None = None,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_steps: int = 19,
) -> tuple[float | np.ndarray, dict]:
    if thresholds is None:
        thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    strategy = strategy.lower()
    if strategy == "global":
        return tune_global_threshold(y_true, y_prob, metric=metric, thresholds=thresholds)
    if strategy in {"per_label", "per-label", "label"}:
        return tune_per_label_thresholds(y_true, y_prob, thresholds=thresholds)
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def per_label_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for label, values in metrics["per_label"].items():
        rows.append({"genre": label, **values})
    return pd.DataFrame(rows)
