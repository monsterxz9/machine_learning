"""评估:加载训练好的模型,在测试集上推理 + 画对比图。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL_DIR, MODEL_PATH, NORM_STATS_PATH, NUM_TEST_CASES
from get_test_data import load_test_features, load_test_s_params
from model import MicrostripMLP


LABELS = ["S11.real", "S11.imag", "S12.real", "S12.imag"]


def load_model(path: str = MODEL_PATH) -> MicrostripMLP:
    """加载 checkpoint,根据保存的 config 实例化模型。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MicrostripMLP(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_stats(path: str = NORM_STATS_PATH) -> dict:
    npz = np.load(path)
    return {
        "geo_mean": npz["geo_mean"],
        "geo_std": npz["geo_std"],
        "freq_mean": float(npz["freq_mean"]),
        "freq_std": float(npz["freq_std"]),
    }


@torch.no_grad()
def predict_all(
    model: MicrostripMLP,
    stats: dict,
    features: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    对 (N 个测试样本) × (P 个频率点) 全部预测一遍。

    返回 (N, P, 4) 实部虚部预测。
    """
    feat_norm = (features - stats["geo_mean"]) / stats["geo_std"]
    freq_norm = (freqs - stats["freq_mean"]) / stats["freq_std"]

    feat_t = torch.from_numpy(feat_norm).float()
    freq_t = torch.from_numpy(freq_norm).float()
    n, p = feat_t.shape[0], freq_t.shape[0]

    # 笛卡尔积展平再批量推理
    geo_rep = feat_t.unsqueeze(1).expand(n, p, -1).reshape(n * p, -1)
    freq_rep = freq_t.unsqueeze(0).expand(n, p).reshape(n * p)
    pred = model(geo_rep, freq_rep).numpy().reshape(n, p, 4)
    return pred


def plot_case(
    case_idx: int,
    true_vals: np.ndarray,
    pred: np.ndarray,
    save_dir: str = MODEL_DIR,
) -> str:
    """画某个 case 的 true vs pred + 误差曲线,共 4 行(每个 S 分量一行)。"""
    n_freqs = true_vals.shape[1]
    x = np.arange(n_freqs)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    for c in range(4):
        ax_a = axes[c, 0]
        ax_a.plot(x, true_vals[case_idx, :, c], label="true", linewidth=1.5)
        ax_a.plot(x, pred[case_idx, :, c], label="pred", linewidth=1.5, linestyle="--")
        ax_a.set_ylabel(LABELS[c])
        ax_a.legend(loc="best", fontsize=8)
        ax_a.grid(alpha=0.3)

        ax_e = axes[c, 1]
        err = true_vals[case_idx, :, c] - pred[case_idx, :, c]
        ax_e.plot(x, err, color="red", linewidth=1)
        ax_e.set_ylabel(f"{LABELS[c]} err")
        ax_e.axhline(0, color="black", linewidth=0.5)
        ax_e.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Frequency index")
    axes[-1, 1].set_xlabel("Frequency index")
    fig.suptitle(f"Case {case_idx + 1}: prediction vs ground truth")
    plt.tight_layout()

    out = os.path.join(save_dir, f"case_{case_idx + 1}_eval.png")
    plt.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    print("Loading model + stats...")
    model = load_model()
    stats = load_stats()

    print("Loading test data...")
    features = load_test_features()
    true_s, freqs = load_test_s_params()
    print(f"Cases: {features.shape[0]}, Freqs: {freqs.shape[0]}")

    pred = predict_all(model, stats, features, freqs)

    mse = float(((true_s - pred) ** 2).mean())
    mae = float(np.abs(true_s - pred).mean())
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    for i in range(NUM_TEST_CASES):
        print(f"Saved: {plot_case(i, true_s, pred)}")


if __name__ == "__main__":
    main()
