"""
模型评估模块。
加载训练好的模型，对测试数据进行预测并计算误差。
支持单模型和多模型两种评估管线。

重要修复：模型预测值直接就是 S 参数（S11.real, S11.imag, S12.real, S12.imag），
不需要再做阻抗→S 参数的转换。
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    MODEL_MODE,
    MODEL_DIR,
    NUM_MODELS,
    NUM_TEST_CASES,
    SINGLE_MODEL_PATH,
    NORMALIZATION_STATS_PATH,
)
from get_test_data import (
    import_test_data,
    import_s2p_data,
    import_s2p_data_all_freqs,
    feature_of_data,
)


# ==================== 预测函数 ====================


def predict_to_s_params(predictions):
    """将模型原始预测 (N, 4) 转换为复数 S 参数。

    模型输出列: [S11.real, S11.imag, S12.real, S12.imag]
    对于互易网络: S21 = S12, S22 = S11
    """
    s11 = predictions[:, 0] + 1j * predictions[:, 1]
    s12 = predictions[:, 2] + 1j * predictions[:, 3]
    return s11, s12, s12, s11


def predict_single_model():
    """
    单模型模式：加载统一模型，对所有测试样本和频率点进行预测。

    返回:
        predict: shape (NUM_TEST_CASES, num_freqs, 4) 的实数数组
                 列为 [S11.real, S11.imag, S12.real, S12.imag]
    """
    model = tf.keras.models.load_model(SINGLE_MODEL_PATH)
    stats = np.load(NORMALIZATION_STATS_PATH)
    feat_mean, feat_std = stats["feat_mean"], stats["feat_std"]
    freq_mean, freq_std = float(stats["freq_mean"]), float(stats["freq_std"])

    # 加载测试特征并标准化
    test_features = import_test_data()
    test_norm = (test_features - feat_mean) / feat_std

    # 获取测试 S2P 的频率列表
    _, freqs = import_s2p_data_all_freqs()
    freq_normalized = (freqs - freq_mean) / freq_std

    predict = np.empty((NUM_TEST_CASES, len(freqs), 4))
    for j, fn in enumerate(freq_normalized):
        X = np.column_stack([test_norm, np.full(len(test_norm), fn)])
        pred = model.predict(X, verbose=0)
        predict[:, j, :] = pred

    return predict


def predict_multi_model():
    """
    多模型模式：加载各频率点模型进行预测。

    返回:
        predict: shape (NUM_MODELS, NUM_TEST_CASES, 4) 的复数数组
                 列为 [S11, S12, S21, S22]
    """
    test = import_test_data()
    data_mean, data_std = feature_of_data()
    test = (test - data_mean) / data_std

    predict = np.empty((NUM_MODELS, NUM_TEST_CASES, 4), dtype=complex)
    for i in range(1, NUM_MODELS + 1):
        model_path = os.path.join(MODEL_DIR, f"model_{i}.h5")
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(test, verbose=0)
        s11, s12, s21, s22 = predict_to_s_params(pred)
        predict[i - 1, :, 0] = s11
        predict[i - 1, :, 1] = s12
        predict[i - 1, :, 2] = s21
        predict[i - 1, :, 3] = s22
    return predict


# ==================== 误差计算 ====================


def calculate_error_single():
    """单模型模式：计算所有频率点的绝对和相对误差。"""
    true_values, _ = import_s2p_data_all_freqs()  # (NUM_TEST_CASES, num_freqs, 4)
    predict = predict_single_model()               # (NUM_TEST_CASES, num_freqs, 4)
    absolute_error = true_values - predict
    relative_error = np.where(
        np.abs(true_values) > 1e-10,
        absolute_error / true_values,
        0.0,
    )
    return absolute_error, relative_error


def calculate_error_multi():
    """多模型模式：计算绝对和相对误差。"""
    true_values = import_s2p_data()  # (NUM_TEST_CASES, 4)
    predict = predict_multi_model()   # (NUM_MODELS, NUM_TEST_CASES, 4)
    absolute_error = true_values - predict
    relative_error = np.where(
        np.abs(true_values) > 1e-10,
        absolute_error / true_values,
        0.0,
    )
    return absolute_error, relative_error


# ==================== 可视化 ====================


def draw_picture_single(case_idx, absolute_error, relative_error):
    """单模型模式：绘制某个测试样本在所有频率上的误差。"""
    num_freqs = absolute_error.shape[1]
    x_axis = np.arange(num_freqs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for col in range(4):
        ax1.plot(x_axis, absolute_error[case_idx, :, col], label=f"col{col}")
    ax1.set_title(f"Case {case_idx + 1} - Absolute Error")
    ax1.set_xlabel("Frequency Index")
    ax1.legend()

    for col in range(4):
        ax2.plot(x_axis, relative_error[case_idx, :, col], label=f"col{col}")
    ax2.set_title(f"Case {case_idx + 1} - Relative Error")
    ax2.set_xlabel("Frequency Index")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"case_{case_idx + 1}_error.png")
    plt.close(fig)


def draw_picture_multi(model_idx, absolute_error, relative_error):
    """多模型模式：绘制某个模型在所有测试样本上的误差。"""
    x_axis = np.arange(1, NUM_TEST_CASES + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i in range(2):
        ax1.scatter(x_axis, np.real(absolute_error[model_idx, :, i]), marker=".", c="r", label="real" if i == 0 else "")
        ax1.scatter(x_axis, np.imag(absolute_error[model_idx, :, i]), marker=".", c="b", label="imag" if i == 0 else "")
    ax1.set_title(f"Model {model_idx + 1} - Absolute Error")
    ax1.legend()

    for i in range(2):
        ax2.scatter(x_axis, np.real(relative_error[model_idx, :, i]), marker=".", c="r")
        ax2.scatter(x_axis, np.imag(relative_error[model_idx, :, i]), marker=".", c="b")
    ax2.set_title(f"Model {model_idx + 1} - Relative Error")
    ax2.set_xlabel("Test Case")

    plt.tight_layout()
    plt.savefig(f"model_{model_idx + 1}_error.png")
    plt.close(fig)


def draw_all_pictures():
    """根据模式绘制所有误差图。"""
    if MODEL_MODE == "single":
        absolute_error, relative_error = calculate_error_single()
        for i in range(NUM_TEST_CASES):
            draw_picture_single(i, absolute_error, relative_error)
            print(f"已保存 case_{i + 1}_error.png")
    else:
        absolute_error, relative_error = calculate_error_multi()
        for i in range(NUM_MODELS):
            draw_picture_multi(i, absolute_error, relative_error)
            print(f"已保存 model_{i + 1}_error.png")


if __name__ == "__main__":
    print(f"评估模式: {MODEL_MODE}")
    draw_all_pictures()
