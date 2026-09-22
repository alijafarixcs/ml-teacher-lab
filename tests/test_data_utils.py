import numpy as np
import pytest

from src.data_utils import (
    load_course_dataset,
    standardize_from_training,
    train_validation_test_split,
)


def test_builtin_dataset_is_loaded_without_network_access():
    bundle = load_course_dataset("iris")
    assert bundle.features.shape == (150, 4)
    assert bundle.target.shape == (150,)
    assert bundle.task == "classification"
    assert not bundle.frame.empty


def test_generated_dataset_is_reproducible():
    first = load_course_dataset("regression", random_state=7)
    second = load_course_dataset("regression", random_state=7)
    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_allclose(first.target, second.target)


def test_three_way_split_preserves_all_rows_and_is_disjoint():
    features = np.arange(200).reshape(100, 2)
    target = np.arange(100)
    splits = train_validation_test_split(features, target)
    x_train, x_validation, x_test, y_train, y_validation, y_test = splits
    assert len(x_train) + len(x_validation) + len(x_test) == 100
    assert set(y_train).isdisjoint(y_validation)
    assert set(y_train).isdisjoint(y_test)
    assert set(y_validation).isdisjoint(y_test)


def test_scaler_is_fit_on_training_data_only():
    train = np.array([[0.0], [2.0]])
    validation = np.array([[100.0]])
    scaler, train_scaled, validation_scaled = standardize_from_training(train, validation)
    np.testing.assert_allclose(train_scaled.mean(axis=0), [0.0], atol=1e-12)
    assert scaler.mean_[0] == pytest.approx(1.0)
    assert validation_scaled[0, 0] == pytest.approx(99.0)


def test_unknown_dataset_has_actionable_error():
    with pytest.raises(ValueError, match="Choose one of"):
        load_course_dataset("not-a-dataset")

