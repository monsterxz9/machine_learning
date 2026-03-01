"""
测试数据导入模块。
从 CSV 和 S2P 文件中加载测试数据及其真实 S 参数。
"""

import numpy as np
import os
import pandas as pd
import skrf as rf

from config import TEST_CSV_PATH, TEST_S2P_DIR, FEATURE_CSV_PATH, NUM_TEST_CASES
from preprocessing import csv_to_numpy


def import_test_data(csv_file_path=TEST_CSV_PATH):
    """从 CSV 文件导入测试集特征数据（前4列：W, L, H, Er）。"""
    data = pd.read_csv(csv_file_path)
    return data.iloc[:, :4].values


def import_s2p_data(s2p_dir=TEST_S2P_DIR):
    """
    从测试 S2P 文件中提取 S 参数。

    返回:
        result: shape (NUM_TEST_CASES, 4) 的复数数组，
                列分别为 S11, S12, S21, S22（取每个文件的第一个频率点）
    """
    s2p_files = [
        os.path.join(s2p_dir, f"case{i}.s2p") for i in range(1, NUM_TEST_CASES + 1)
    ]
    result = np.empty((NUM_TEST_CASES, 4), dtype=complex)

    for i, s2p_file in enumerate(s2p_files):
        network = rf.Network(s2p_file)
        s_params = network.s
        # 取第一个频率点的 S 参数
        result[i, 0] = s_params[0, 0, 0]  # S11
        result[i, 1] = s_params[0, 0, 1]  # S12
        result[i, 2] = s_params[0, 1, 0]  # S21
        result[i, 3] = s_params[0, 1, 1]  # S22

    return result


def feature_of_data(csv_file_path=FEATURE_CSV_PATH):
    """返回训练特征的均值和标准差，用于测试数据的标准化。"""
    data = csv_to_numpy(csv_file_path)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return data_mean, data_std
