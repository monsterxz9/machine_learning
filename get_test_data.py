"""测试数据加载:从测试 CSV 和 .s2p 提取特征与全频段 S 参数。"""

import os

import numpy as np
import pandas as pd
import skrf as rf

from config import NUM_TEST_CASES, TEST_CSV_PATH, TEST_S2P_DIR


def load_test_features(csv_path: str = TEST_CSV_PATH) -> np.ndarray:
    """读测试集特征 CSV(前 4 列:W, L, H, Er)。"""
    df = pd.read_csv(csv_path)
    return df.iloc[:, :4].values.astype(np.float32)


def load_test_s_params(s2p_dir: str = TEST_S2P_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    读测试 .s2p,提取全频段 S 参数。

    返回:
        s_params: (NUM_TEST_CASES, P, 4) [S11.real, S11.imag, S12.real, S12.imag]
        freqs:    (P,)
    """
    files = [os.path.join(s2p_dir, f"case{i + 1}.s2p") for i in range(NUM_TEST_CASES)]
    sample = rf.Network(files[0])
    n_freqs = len(sample.f)
    freqs = sample.f.astype(np.float32)

    s_params = np.empty((NUM_TEST_CASES, n_freqs, 4), dtype=np.float32)
    for i, path in enumerate(files):
        net = rf.Network(path)
        s = net.s
        s_params[i, :, 0] = s[:, 0, 0].real
        s_params[i, :, 1] = s[:, 0, 0].imag
        s_params[i, :, 2] = s[:, 0, 1].real
        s_params[i, :, 3] = s[:, 0, 1].imag

    return s_params, freqs


if __name__ == "__main__":
    feats = load_test_features()
    s, freqs = load_test_s_params()
    print(f"Features: {feats.shape}")
    print(f"S params: {s.shape}")
    print(f"Freqs: {freqs.shape} | range: [{freqs.min():.2e}, {freqs.max():.2e}]")
