"""Small, deterministic dataset helpers used throughout the course."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetTask = Literal["classification", "regression", "clustering"]


@dataclass(frozen=True)
class DatasetBundle:
    """A dataset plus the metadata needed to explain and reproduce it."""

    frame: pd.DataFrame
    features: np.ndarray
    target: np.ndarray | None
    feature_names: list[str]
    target_names: list[str] | None
    task: DatasetTask
    source: str


def load_course_dataset(name: str, random_state: int = 42) -> DatasetBundle:
    """Load a small built-in or generated course dataset by name.

    Supported names are ``iris``, ``wine``, ``breast_cancer``, ``digits``,
    ``diabetes``, ``regression``, ``classification``, ``blobs``, and ``moons``.
    The function performs no network access.
    """

    key = name.lower().replace("-", "_").strip()
    sklearn_loaders = {
        "iris": (load_iris, "classification"),
        "wine": (load_wine, "classification"),
        "breast_cancer": (load_breast_cancer, "classification"),
        "digits": (load_digits, "classification"),
        "diabetes": (load_diabetes, "regression"),
    }
    if key in sklearn_loaders:
        loader, task = sklearn_loaders[key]
        bunch = loader(as_frame=True)
        frame = bunch.frame.copy()
        features = bunch.data.to_numpy(dtype=float)
        target = bunch.target.to_numpy()
        target_names = (
            [str(value) for value in bunch.target_names]
            if hasattr(bunch, "target_names")
            else None
        )
        return DatasetBundle(
            frame=frame,
            features=features,
            target=target,
            feature_names=[str(value) for value in bunch.feature_names],
            target_names=target_names,
            task=task,
            source=f"scikit-learn built-in {key} dataset",
        )

    if key == "regression":
        features, target = make_regression(
            n_samples=240,
            n_features=3,
            n_informative=3,
            noise=18.0,
            random_state=random_state,
        )
        task: DatasetTask = "regression"
    elif key == "classification":
        features, target = make_classification(
            n_samples=300,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            class_sep=1.2,
            random_state=random_state,
        )
        task = "classification"
    elif key == "blobs":
        features, target = make_blobs(
            n_samples=300,
            centers=3,
            cluster_std=1.0,
            random_state=random_state,
        )
        task = "clustering"
    elif key == "moons":
        features, target = make_moons(
            n_samples=300,
            noise=0.08,
            random_state=random_state,
        )
        task = "clustering"
    else:
        supported = sorted([*sklearn_loaders, "regression", "classification", "blobs", "moons"])
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {', '.join(supported)}")

    feature_names = [f"feature_{index + 1}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=feature_names)
    frame["target"] = target
    return DatasetBundle(
        frame=frame,
        features=features,
        target=target,
        feature_names=feature_names,
        target_names=None,
        task=task,
        source=f"scikit-learn generated {key} dataset (seed={random_state})",
    )


def train_validation_test_split(
    features: np.ndarray,
    target: np.ndarray,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[np.ndarray, ...]:
    """Split arrays into train, validation, and test sets without leakage."""

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    stratify_values = target if stratify else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_val_stratify = y_train_val if stratify else None
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_val_stratify,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def standardize_from_training(
    x_train: np.ndarray,
    *other_splits: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, *tuple[np.ndarray, ...]]:
    """Fit a scaler only on training data and transform every supplied split."""

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(x_train)
    transformed_others = tuple(scaler.transform(split) for split in other_splits)
    return scaler, transformed_train, *transformed_others

