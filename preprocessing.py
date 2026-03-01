"""
数据预处理模块。
负责读取 CSV 特征数据和 S2P 散射参数文件，并进行标准化处理。
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
        for row in csv_reader:
            data.append(row)
    return np.array(data, dtype=np.float32)


def feature_processing(csv_file_path=FEATURE_CSV_PATH):
    """读取特征 CSV 并进行 Z-Score 标准化，返回 (标准化数据, 均值, 标准差)。"""
    data = csv_to_numpy(csv_file_path)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    normalized = (data - data_mean) / data_std
    return normalized, data_mean, data_std


def process_s2p_files(n=0, s2p_dir=TRAIN_S2P_DIR):
    """
    读取 S2P 文件，提取第 n 个频率点（0-indexed）对应的 S11 和 S12 参数。

    参数:
        n: 频率点索引（0 ~ POINTS_PER_FILE-1），每个模型对应不同的频率点
        s2p_dir: S2P 文件所在目录

    返回:
        Z: shape (NUM_S2P_FILES, 4) 的数组，列分别为 S11实部/虚部、S12实部/虚部
    """
    s2p_files = [
        os.path.join(s2p_dir, f"{i}.s2p") for i in range(1, NUM_S2P_FILES + 1)
    ]

    result_array = np.empty((TOTAL_POINTS, 4))

    for i, s2p_file in enumerate(s2p_files):
        network = rf.Network(s2p_file)
        s_params = network.s
        for j in range(len(network.f)):
            s11 = s_params[j, 0, 0]
            s12 = s_params[j, 0, 1]
            index = i * len(network.f) + j
            result_array[index, 0] = s11.real
            result_array[index, 1] = s11.imag
            result_array[index, 2] = s12.real
            result_array[index, 3] = s12.imag

    # 从每个文件中取第 n 个频率点的数据（步长 = POINTS_PER_FILE）
    freq_offset = n % POINTS_PER_FILE
    indices = np.arange(freq_offset, TOTAL_POINTS, POINTS_PER_FILE)
    Z = result_array[indices, :]
    return Z


def get_training_set(n):
    """
    获取第 n 个模型的训练集。

    返回:
        x: 标准化后的特征数据
        Z: 对应频率点的 S 参数标签
    """
    x, _, _ = feature_processing()
    Z = process_s2p_files(n)
    return x, Z


if __name__ == "__main__":
    x, Z = get_training_set(0)
    print("x (特征数据):")
    print(x)
    print(f"x shape: {x.shape}")
    print("Z (S参数标签):")
    print(Z)
    print(f"Z shape: {Z.shape}")
