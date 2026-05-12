"""训练特征 (W, L, H, Er) 分布散点图。"""

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import load_features

FEATURE_NAMES = ["W", "L", "H", "Er"]
FEATURE_COLORS = ["r", "g", "b", "grey"]


def plot_feature_distributions(save_path: str = "feature_distributions.png") -> str:
    """画 4 个特征的原始(未标准化)分布。"""
    normalized, mean, std = load_features()
    raw = normalized * std + mean
    x = np.arange(1, raw.shape[0] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, ax in enumerate(axes.flat):
        ax.scatter(x, raw[:, idx], marker=".", c=FEATURE_COLORS[idx], s=1)
        ax.set_xlabel(f"Mean = {mean[idx]:.6f}  Std = {std[idx]:.6f}")
        ax.set_title(FEATURE_NAMES[idx])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    path = plot_feature_distributions()
    print(f"Saved: {path}")
