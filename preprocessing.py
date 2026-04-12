"""
数据预处理模块。
负责读取 CSV 特征数据和 S2P 散射参数文件，并进行标准化处理。
支持单模型（频率作为输入）和多模型两种数据管线。
"""

import numpy as np
import csv
import os
import skrf as rf

from config import (
    FEATURE_CSV_PATH,
    TRAIN_S2P_DIR,
    NUM_S2P_FILES,
    POINTS_PER_FILE,
    TOTAL_POINTS,
)


def csv_to_numpy(csv_file_path):
    """读取 CSV 文件并转换为 NumPy 浮点数组。"""
    data = []
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file)
        for row_idx, row in enumerate(csv_reader):
            if not row:
                continue
            try:
                data.append([float(value) for value in row])
            except ValueError:
                if row_idx == 0:
                    continue
                raise
    return np.array(data, dtype=np.float32)


def feature_processing(csv_file_path=FEATURE_CSV_PATH):
    """读取特征 CSV 并进行 Z-Score 标准化，返回 (标准化数据, 均值, 标准差)。"""
    data = csv_to_numpy(csv_file_path)
    if data.shape[1] > 4:
        data = data[:, :4]
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    normalized = (data - data_mean) / data_std
    return normalized, data_mean, data_std


def _load_all_s2p_data(s2p_dir=TRAIN_S2P_DIR):
    """
    读取所有 S2P 文件，提取全部频率点的 S11 和 S12 参数。

    返回:
        result_array: shape (TOTAL_POINTS, 4) 的数组
        freqs: shape (POINTS_PER_FILE,) 的频率数组
    """
    s2p_files = [os.path.join(s2p_dir, f"{i}.s2p") for i in range(1, NUM_S2P_FILES + 1)]

    result_array = np.empty((TOTAL_POINTS, 4))
    freqs = None

    for i, s2p_file in enumerate(s2p_files):
        network = rf.Network(s2p_file)
        s_params = network.s
        if freqs is None:
            freqs = network.f
        for j in range(len(network.f)):
            s11 = s_params[j, 0, 0]
            s12 = s_params[j, 0, 1]
            index = i * len(network.f) + j
            result_array[index, 0] = s11.real
            result_array[index, 1] = s11.imag
            result_array[index, 2] = s12.real
            result_array[index, 3] = s12.imag

    return result_array, freqs


def process_s2p_files(n=0, s2p_dir=TRAIN_S2P_DIR):
    """
    多模型模式：提取第 n 个频率点对应的 S 参数。

    参数:
        n: 频率点索引（0 ~ POINTS_PER_FILE-1）

    返回:
        Z: shape (NUM_S2P_FILES, 4) 的数组
    """
    result_array, _ = _load_all_s2p_data(s2p_dir)
    freq_offset = n % POINTS_PER_FILE
    indices = np.arange(freq_offset, TOTAL_POINTS, POINTS_PER_FILE)
    return result_array[indices, :]


def get_training_set(n):
    """
    多模型模式：获取第 n 个模型的训练集。

    返回:
        x: 标准化后的特征数据 shape (NUM_S2P_FILES, 4)
        Z: 对应频率点的 S 参数标签 shape (NUM_S2P_FILES, 4)
    """
    x, _, _ = feature_processing()
    Z = process_s2p_files(n)
    return x, Z


def get_unified_training_set():
    """
    单模型模式：将频率作为第 5 个输入特征，生成完整训练集。

    返回:
        X: shape (TOTAL_POINTS, 5) — [W, L, H, Er, freq]（均已标准化）
        Z: shape (TOTAL_POINTS, 4) — [S11.real, S11.imag, S12.real, S12.imag]
        feat_mean, feat_std: 4 维特征的均值和标准差
        freq_mean, freq_std: 频率的均值和标准差
    """
    features, feat_mean, feat_std = feature_processing()

    # 加载全量 S 参数数据和频率
    Z, freqs = _load_all_s2p_data()
    if freqs is None:
        raise ValueError("S2P 频率数据为空")
    freq_mean = freqs.mean()
    freq_std = freqs.std()
    freq_normalized = (freqs - freq_mean) / freq_std

    # 构建 X：每个样本的 4 特征重复 POINTS_PER_FILE 次，拼接对应频率
    X_feat = np.repeat(features, len(freqs), axis=0)  # (TOTAL_POINTS, 4)
    freq_col = np.tile(freq_normalized, len(features))  # (TOTAL_POINTS,)
    X = np.column_stack([X_feat, freq_col])  # (TOTAL_POINTS, 5)

    return X, Z, feat_mean, feat_std, freq_mean, freq_std


if __name__ == "__main__":
    from config import MODEL_MODE

    if MODEL_MODE == "single":
        X, Z, fm, fs, frm, frs = get_unified_training_set()
        print(f"单模型 - X shape: {X.shape}, Z shape: {Z.shape}")
        print(f"特征均值: {fm}, 特征标准差: {fs}")
        print(f"频率均值: {frm:.2f}, 频率标准差: {frs:.2f}")
    else:
        x, Z = get_training_set(0)
        print(f"多模型 - x shape: {x.shape}, Z shape: {Z.shape}")
