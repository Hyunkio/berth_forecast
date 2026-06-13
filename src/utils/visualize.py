"""시각화 유틸리티."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"]       = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    port_names: list,
    dates: pd.DatetimeIndex | None = None,
    title: str = "체선율 예측 vs 실제",
) -> plt.Figure:
    """항만별 예측/실제 비교 플롯. y shape: (N, horizon, n_ports)."""
    n_ports = len(port_names)
    fig, axes = plt.subplots(n_ports, 1, figsize=(14, 3 * n_ports), sharex=True)
    if n_ports == 1:
        axes = [axes]

    for i, (ax, port) in enumerate(zip(axes, port_names)):
        true_flat = y_true[:, 0, i]   # 1-day-ahead 값만 플롯
        pred_flat = y_pred[:, 0, i]
        x = np.arange(len(true_flat)) if dates is None else dates[:len(true_flat)]
        ax.plot(x, true_flat, label="실제", linewidth=0.9)
        ax.plot(x, pred_flat, label="예측", linewidth=0.9, linestyle="--")
        ax.set_title(f"{port}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("체선율")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_loss_curve(train_losses: list, val_losses: list, title: str = "학습 손실") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_model_comparison(results: dict, metric: str = "RMSE") -> plt.Figure:
    """{'LSTM': 0.12, 'Transformer': 0.09, 'Transformer+Event': 0.07} 형식."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    values = [results[n] for n in names]
    bars = ax.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title(f"모델 비교 — {metric}")
    ax.set_ylabel(metric)
    return fig
