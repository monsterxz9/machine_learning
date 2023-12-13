import os
import numpy as np
import tensorflow as tf
from get_test_data import import_test_data, import_s2p_data, feature_of_data
import matplotlib.pyplot as plt

def calculate_s_parameters(Z0_real, Z0_imag, ZL_real, ZL_imag):
    # 使用NumPy矢量化操作处理数组
    Z0 = Z0_real + 1j * Z0_imag
    ZL = ZL_real + 1j * ZL_imag
    # 计算S11和S12
    S11 = (ZL / Z0 - 1) / (ZL / Z0 + 1)
    S12 = 2 * ZL / (Z0 * (1 - S11))
    return S11, S12, S12, S11


def use_model(n, test):
    model_path = os.path.join(r'E:/program/machine_learning/model/', f'model_{n}.h5')
    model = tf.keras.models.load_model(model_path)
    predict = model.predict(test)
    # 通过索引访问四个值并传递给calculate_s_parameters函数
    S11, S12, S21, S22 = calculate_s_parameters(predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3])
    return S11, S12, S21, S22

def test_all():
    test = import_test_data()
    data_mean, data_std = feature_of_data()
    predict = np.empty((20, 10, 4), dtype=complex)
    test = (test - data_mean) / data_std
    for i in range(1, 21):
        S11, S12, S21, S22 = use_model(i, test)
        # 存储S参数结果
        predict[i-1, :, 0] = S11
        predict[i-1, :, 1] = S12
        predict[i-1, :, 2] = S21
        predict[i-1, :, 3] = S22
    return predict

def calculate_error():
    true = import_s2p_data()
    predict = test_all()
    absolute_error = true - predict
    relative_error = absolute_error / true
    return absolute_error, relative_error

# absolute_error, relative_error = calculate_error()
# print(absolute_error)
# # print(relative_error)

def draw_picture(n):
    absolute_error, relative_error = calculate_error()
    real_part = np.real(absolute_error)
    imag_part = np.imag(absolute_error)
    a = np.array([i for i in range(1, 11, 1)])
    plt.subplot(2, 1, 1)  # 第一个子图
    for i in range(2):
        plt.scatter(a, real_part[n, :, i], marker='.', c='r')
        plt.scatter(a, imag_part[n, :, i], marker='.', c='r')
    plt.title("absolute error")
    plt.subplot(2, 1, 2)  # 第二个子图
    real_part = np.real(relative_error)
    imag_part = np.imag(relative_error)
    for i in range(2):
        plt.scatter(a, real_part[n, :, i], marker='.', c='r')
        plt.scatter(a, imag_part[n, :, i], marker='.', c='r')
    plt.xlabel(f"time")
    plt.title("relative error")
    plt.savefig(f'model_{n + 1}.png')
    plt.show()

def draw_all_picture():
    for i in range(20):
        draw_picture(i)

draw_all_picture()

