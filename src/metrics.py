"""Transparent NumPy implementations of common evaluation metrics."""

from __future__ import annotations

import numpy as np


def _paired_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same number of values")
    if true.size == 0:
        raise ValueError("metrics require at least one observation")
    return true, pred


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute difference between targets and predictions."""

    true, pred = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared prediction error."""

    true, pred = _paired_arrays(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE in the same units as the target."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of target variance explained by predictions."""

    true, pred = _paired_arrays(y_true, y_pred)
    total = np.sum((true - np.mean(true)) ** 2)
    if np.isclose(total, 0.0):
        raise ValueError("R-squared is undefined when y_true is constant")
    residual = np.sum((true - pred) ** 2)
    return float(1 - residual / total)


def binary_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: int | str = 1,
) -> tuple[int, int, int, int]:
    """Return true-positive, true-negative, false-positive, false-negative counts."""

    true, pred = _paired_arrays(y_true, y_pred)
    true_positive = int(np.sum((true == positive_label) & (pred == positive_label)))
    true_negative = int(np.sum((true != positive_label) & (pred != positive_label)))
    false_positive = int(np.sum((true != positive_label) & (pred == positive_label)))
    false_negative = int(np.sum((true == positive_label) & (pred != positive_label)))
    return true_positive, true_negative, false_positive, false_negative


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: int | str = 1,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 with safe zero divisions."""

    tp, tn, fp, fn = binary_confusion_counts(
        y_true, y_pred, positive_label=positive_label
    )
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

