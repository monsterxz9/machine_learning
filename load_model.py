"""
模型评估模块。
加载训练好的模型，对测试数据进行预测并计算误差。
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import MODEL_DIR, NUM_MODELS, NUM_TEST_CASES
from get_test_data import import_test_data, import_s2p_data, feature_of_data


def calculate_s_parameters(Z0_real, Z0_imag, ZL_real, ZL_imag):
    """
    根据模型预测的阻抗值计算 S 参数。

    参数:
        Z0_real, Z0_imag: 特征阻抗的实部和虚部
        ZL_real, ZL_imag: 负载阻抗的实部和虚部

    返回:
        S11, S12, S21, S22 四个 S 参数（复数数组）
    """
    Z0 = Z0_real + 1j * Z0_imag
    ZL = ZL_real + 1j * ZL_imag
    S11 = (ZL / Z0 - 1) / (ZL / Z0 + 1)
    S12 = 2 * ZL / (Z0 * (1 - S11))
    return S11, S12, S12, S11


def use_model(n, test_data, model_dir=MODEL_DIR):
    """加载第 n 个模型并预测 S 参数。"""
    model_path = os.path.join(model_dir, f"model_{n}.h5")
    model = tf.keras.models.load_model(model_path)
    predict = model.predict(test_data)
    S11, S12, S21, S22 = calculate_s_parameters(
        predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3]
    )
    return S11, S12, S21, S22


def test_all():
    """使用所有模型对测试数据进行预测，返回 shape (NUM_MODELS, NUM_TEST_CASES, 4) 的复数数组。"""
    test = import_test_data()
    data_mean, data_std = feature_of_data()
    test = (test - data_mean) / data_std

    predict = np.empty((NUM_MODELS, NUM_TEST_CASES, 4), dtype=complex)
    for i in range(1, NUM_MODELS + 1):
        S11, S12, S21, S22 = use_model(i, test)
        predict[i - 1, :, 0] = S11
        predict[i - 1, :, 1] = S12
        predict[i - 1, :, 2] = S21
        predict[i - 1, :, 3] = S22
    return predict


def calculate_error():
    """计算所有模型的绝对误差和相对误差。"""
    true_values = import_s2p_data()
    predict = test_all()
    absolute_error = true_values - predict
    # 避免除零
    relative_error = np.where(
        np.abs(true_values) > 1e-10,
        absolute_error / true_values,
        0.0,
    )
    return absolute_error, relative_error


def draw_picture(n, absolute_error, relative_error):
    """绘制第 n 个模型的绝对误差和相对误差散点图。"""
    x_axis = np.arange(1, NUM_TEST_CASES + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 绝对误差
    for i in range(2):
        ax1.scatter(x_axis, np.real(absolute_error[n, :, i]), marker=".", c="r", label="real" if i == 0 else "")
        ax1.scatter(x_axis, np.imag(absolute_error[n, :, i]), marker=".", c="b", label="imag" if i == 0 else "")
    ax1.set_title(f"Model {n + 1} - Absolute Error")
    ax1.legend()

    # 相对误差
    for i in range(2):
        ax2.scatter(x_axis, np.real(relative_error[n, :, i]), marker=".", c="r")
        ax2.scatter(x_axis, np.imag(relative_error[n, :, i]), marker=".", c="b")
    ax2.set_title(f"Model {n + 1} - Relative Error")
    ax2.set_xlabel("Test Case")

    plt.tight_layout()
    plt.savefig(f"model_{n + 1}_error.png")
    plt.close(fig)


def draw_all_pictures():
    """计算误差并绘制所有模型的误差图。"""
    absolute_error, relative_error = calculate_error()
    for i in range(NUM_MODELS):
        draw_picture(i, absolute_error, relative_error)
        print(f"已保存 model_{i + 1}_error.png")


if __name__ == "__main__":
    draw_all_pictures()
