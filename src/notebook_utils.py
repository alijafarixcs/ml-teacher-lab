"""Cross-notebook helpers for reproducibility and project discovery."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is available."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Classical notebooks are intentionally usable without importing PyTorch.
        return


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml``."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find a parent directory containing pyproject.toml")


def add_project_root_to_path(start: str | Path | None = None) -> Path:
    """Add the repository root to ``sys.path`` and return it."""

    import sys

    root = find_project_root(start)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def moving_average(values: Sequence[float], window: int = 25) -> np.ndarray:
    """Return the simple moving average over a positive window."""

    array = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be at least 1")
    if array.size < window:
        raise ValueError("window cannot be larger than the number of values")
    weights = np.ones(window, dtype=float) / window
    return np.convolve(array, weights, mode="valid")


def count_trainable_parameters(model: object) -> int:
    """Count trainable parameters in a PyTorch model."""

    parameters = getattr(model, "parameters", None)
    if parameters is None:
        raise TypeError("model must expose a parameters() method")
    return int(sum(parameter.numel() for parameter in parameters() if parameter.requires_grad))

