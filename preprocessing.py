"""
数据预处理:
- 读 CSV 特征 + .s2p 文件
- Z-Score 标准化
- 按几何样本随机切分 train/val,构造 PyTorch Dataset
"""

import csv
import os

import numpy as np
import skrf as rf
import torch
from torch.utils.data import Dataset

from config import (
    FEATURE_CSV_PATH,
    NUM_S2P_FILES,
    POINTS_PER_FILE,
    TRAIN_S2P_DIR,
)


def csv_to_numpy(csv_path: str) -> np.ndarray:
    """读 CSV 为 float32 数组,跳过非数字 header。"""
    data: list[list[float]] = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            try:
                data.append([float(v) for v in row])
            except ValueError:
                if row_idx == 0:
                    continue
                raise
    return np.array(data, dtype=np.float32)


def load_features(csv_path: str = FEATURE_CSV_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读特征 CSV(只取前 4 列)并 Z-Score 标准化。返回 (normalized, mean, std)。"""
    data = csv_to_numpy(csv_path)
    if data.shape[1] > 4:
        data = data[:, :4]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized = (data - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def load_all_s2p(s2p_dir: str = TRAIN_S2P_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    读取所有训练 .s2p,提取 S11/S12 实部虚部及频率列表。

    返回:
        s_params: (NUM_S2P_FILES, POINTS_PER_FILE, 4) [S11.real, S11.imag, S12.real, S12.imag]
        freqs:    (POINTS_PER_FILE,)
    """
    s_params = np.empty((NUM_S2P_FILES, POINTS_PER_FILE, 4), dtype=np.float32)
    freqs: np.ndarray | None = None

    for i in range(NUM_S2P_FILES):
        path = os.path.join(s2p_dir, f"{i + 1}.s2p")
        net = rf.Network(path)
        if freqs is None:
            freqs = net.f.astype(np.float32)
        s = net.s  # (P, 2, 2) complex
        s_params[i, :, 0] = s[:, 0, 0].real
        s_params[i, :, 1] = s[:, 0, 0].imag
        s_params[i, :, 2] = s[:, 0, 1].real
        s_params[i, :, 3] = s[:, 0, 1].imag

    assert freqs is not None
    return s_params, freqs


class MicrostripDataset(Dataset):
    """
    展平 (sample, freq) 为一维样本,每条数据 = (geo, freq, S_target)。

    Args:
        geo: (N, 4) 标准化几何参数
        freq_normalized: (P,) 标准化频率
        s_params: (N, P, 4) S 参数标签(实部虚部)
    """

    def __init__(self, geo: np.ndarray, freq_normalized: np.ndarray, s_params: np.ndarray) -> None:
        assert geo.shape[0] == s_params.shape[0]
        assert freq_normalized.shape[0] == s_params.shape[1]
        self.geo = torch.from_numpy(geo).float()
        self.freq = torch.from_numpy(freq_normalized).float()
        self.s = torch.from_numpy(s_params).float()
        self.n_samples, self.n_freqs = s_params.shape[:2]

    def __len__(self) -> int:
        return self.n_samples * self.n_freqs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx, freq_idx = divmod(idx, self.n_freqs)
        return (
            self.geo[sample_idx],
            self.freq[freq_idx],
            self.s[sample_idx, freq_idx],
        )


def build_datasets(
    seed: int = 42,
    val_ratio: float = 0.2,
) -> tuple[MicrostripDataset, MicrostripDataset, dict]:
    """
    构造训练/验证 Dataset。**按几何样本切分**,避免同一几何的
    频率点跨越训练/验证集造成数据泄漏。

    返回 (train_ds, val_ds, stats_dict)
    """
    geo, geo_mean, geo_std = load_features()
    s_params, freqs = load_all_s2p()

    freq_mean = float(freqs.mean())
    freq_std = float(freqs.std())
    freq_norm = ((freqs - freq_mean) / freq_std).astype(np.float32)

    rng = np.random.default_rng(seed)
    n = geo.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = MicrostripDataset(geo[train_idx], freq_norm, s_params[train_idx])
    val_ds = MicrostripDataset(geo[val_idx], freq_norm, s_params[val_idx])

    stats = {
        "geo_mean": geo_mean,
        "geo_std": geo_std,
        "freq_mean": np.array(freq_mean, dtype=np.float32),
        "freq_std": np.array(freq_std, dtype=np.float32),
    }
    return train_ds, val_ds, stats


if __name__ == "__main__":
    train_ds, val_ds, stats = build_datasets()
    print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")
    print(f"Geo mean: {stats['geo_mean']}")
    print(f"Geo std:  {stats['geo_std']}")
    print(f"Freq mean/std: {float(stats['freq_mean']):.2f} / {float(stats['freq_std']):.2f}")
    geo, freq, s = train_ds[0]
    print(f"Sample 0 - geo: {geo.shape}, freq: {freq.shape}, S: {s.shape}")
