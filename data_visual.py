"""
数据可视化模块。
绘制训练特征（W, L, H, Er）的分布散点图及统计信息。
"""

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import get_training_set


FEATURE_NAMES = ["W", "L", "H", "Er"]
FEATURE_COLORS = ["r", "g", "b", "grey"]


def plot_feature_distributions(dataset_index=0):
    """绘制指定数据集的四个特征分布。"""
    X, _ = get_training_set(dataset_index)
    num_samples = X.shape[0]
    x_axis = np.arange(1, num_samples + 1)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, ax in enumerate(axes.flat):
        ax.scatter(x_axis, X[:, idx], marker=".", c=FEATURE_COLORS[idx], s=1)
        ax.set_xlabel(f"Mean = {means[idx]:.6f}  Std = {stds[idx]:.6f}")
        ax.set_title(FEATURE_NAMES[idx])

    plt.tight_layout()
    plt.savefig("feature_distributions.png")
    plt.show()


if __name__ == "__main__":
    plot_feature_distributions()
