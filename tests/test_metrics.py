import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics

from src.metrics import (
    binary_classification_metrics,
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


def test_regression_metrics_match_scikit_learn():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(
        sklearn_metrics.mean_absolute_error(y_true, y_pred)
    )
    assert mean_squared_error(y_true, y_pred) == pytest.approx(
        sklearn_metrics.mean_squared_error(y_true, y_pred)
    )
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        sklearn_metrics.root_mean_squared_error(y_true, y_pred)
    )
    assert r_squared(y_true, y_pred) == pytest.approx(
        sklearn_metrics.r2_score(y_true, y_pred)
    )


def test_binary_metrics_known_example():
    scores = binary_classification_metrics(
        np.array([1, 1, 1, 0, 0, 0]),
        np.array([1, 1, 0, 1, 0, 0]),
    )
    assert scores == pytest.approx(
        {"accuracy": 4 / 6, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    )


def test_metric_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same number"):
        mean_squared_error(np.array([1, 2]), np.array([1]))


def test_r_squared_rejects_constant_target():
    with pytest.raises(ValueError, match="undefined"):
        r_squared(np.ones(3), np.ones(3))

