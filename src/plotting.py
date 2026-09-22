"""Consistent, accessible Matplotlib helpers for course notebooks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


COURSE_COLORS = {
    "blue": "#2463A9",
    "orange": "#E07A2D",
    "gold": "#C69C24",
    "olive": "#6B7D32",
    "pink": "#B34E77",
    "charcoal": "#2F3437",
    "light_gray": "#D9DEE3",
}


def set_course_style() -> None:
    """Apply the shared visual style used by every notebook."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def plot_decision_boundary(
    predict: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    target: np.ndarray,
    *,
    title: str,
    feature_names: Sequence[str] = ("Feature 1", "Feature 2"),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a classifier or clusterer's regions for exactly two features."""

    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("plot_decision_boundary requires a (n_samples, 2) feature array")
    x_padding = 0.08 * np.ptp(features[:, 0])
    y_padding = 0.08 * np.ptp(features[:, 1])
    x_values = np.linspace(features[:, 0].min() - x_padding, features[:, 0].max() + x_padding, 250)
    y_values = np.linspace(features[:, 1].min() - y_padding, features[:, 1].max() + y_padding, 250)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = np.asarray(predict(grid)).reshape(xx.shape)

    figure, axis = plt.subplots()
    axis.contourf(xx, yy, regions, alpha=0.18, cmap="cividis")
    scatter = axis.scatter(
        features[:, 0],
        features[:, 1],
        c=target,
        cmap="cividis",
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set(title=title, xlabel=feature_names[0], ylabel=feature_names[1])
    axis.legend(*scatter.legend_elements(), title="Class", loc="best")
    figure.tight_layout()
    return figure, axis


def plot_learning_history(
    train_values: Sequence[float],
    validation_values: Sequence[float] | None = None,
    *,
    metric_name: str = "Loss",
    title: str = "Training history",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training and optional validation values over epochs."""

    epochs = np.arange(1, len(train_values) + 1)
    figure, axis = plt.subplots()
    axis.plot(epochs, train_values, label=f"Training {metric_name.lower()}", color=COURSE_COLORS["blue"])
    if validation_values is not None:
        axis.plot(
            epochs,
            validation_values,
            label=f"Validation {metric_name.lower()}",
            color=COURSE_COLORS["orange"],
            linestyle="--",
        )
    axis.set(title=title, xlabel="Epoch", ylabel=metric_name)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_rl_history(
    rewards: Sequence[float],
    exploration_rates: Sequence[float],
    episode_lengths: Sequence[int] | None = None,
    *,
    moving_window: int = 25,
    title: str = "Agent learning progress",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot reward, moving-average reward, exploration, and episode length."""

    from .notebook_utils import moving_average

    panel_count = 3 if episode_lengths is not None else 2
    figure, axes = plt.subplots(panel_count, 1, figsize=(9, 3.2 * panel_count), sharex=True)
    episodes = np.arange(1, len(rewards) + 1)
    axes[0].plot(episodes, rewards, alpha=0.35, color=COURSE_COLORS["blue"], label="Reward")
    averaged = moving_average(rewards, moving_window)
    averaged_episodes = episodes[moving_window - 1 :]
    axes[0].plot(averaged_episodes, averaged, color=COURSE_COLORS["orange"], label=f"{moving_window}-episode average")
    axes[0].set(title=title, ylabel="Reward")
    axes[0].legend()
    axes[1].plot(episodes, exploration_rates, color=COURSE_COLORS["olive"])
    axes[1].set(ylabel="Exploration rate")
    if episode_lengths is not None:
        axes[2].plot(episodes, episode_lengths, color=COURSE_COLORS["pink"])
        axes[2].set(ylabel="Episode length", xlabel="Episode")
    else:
        axes[1].set_xlabel("Episode")
    figure.tight_layout()
    return figure, axes

