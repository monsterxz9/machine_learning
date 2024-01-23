import numpy as np
import csv
import os
import skrf as rf


def csv_to_numpy(csv_file_path):
    # 读取CSV文件
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    # 转换数据为NumPy数组，并将数据类型设为浮点数
    data_array = np.array(data, dtype=np.float32)
    return data_array


def feature_processing():
    csv_file_path = r'E:\mline_size.csv'
    data = csv_to_numpy(csv_file_path)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    return data


def process_s2p_files(n=0):
    # 获取 S2P 文件列表
    s2p_files = [os.path.join(r'E:\微带线赛题数据\专题赛数据\s2p', f'{i}.s2p') for i in range(1, 5001)]
    # 初始化一个空的结果数组
    result_array = np.empty((500000, 4))
    # 读取每个S2P文件并计算特征阻抗 Z0 和负载阻抗 ZL
    for i, s2p_file in enumerate(s2p_files):
        network = rf.Network(s2p_file)
        s_params = network.s
        frequencies = network.f
        for j, frequency in enumerate(frequencies):
            s11 = s_params[j, 0, 0]
            s12 = s_params[j, 0, 1]
            index = i * len(frequencies) + j
            result_array[index, 0] = s11.real
            result_array[index, 1] = s11.imag
            result_array[index, 2] = s12.real
            result_array[index, 3] = s12.imag
    # 初始化一个新的数组用于存储平均值
    # new_array = np.empty((100000, 4))
    #
    # # 每5行取平均值
    # for i in range(0, 500000, 5):
    #     new_array[i // 5, :] = np.mean(result_array[i:i+5, :], axis=0)
    Z = np.empty((5000, 4))
    for i in range(5 * (n + 1) - 1, 500000, 100):
        Z[i // 100, 0] = result_array[i, 0]
        Z[i // 100, 1] = result_array[i, 1]
        Z[i // 100, 2] = result_array[i, 2]
        Z[i // 100, 3] = result_array[i, 3]
    return Z


def get_training_set(n):
    # 从CSV文件获取x
    x = feature_processing()
    Z = process_s2p_files(n)
    return x, Z


# 示例用法
if __name__ == "__main__":
    # 调用get_training_set方法获取x
    x, Z = get_training_set(0)
    # 打印x, t的内容
    print("x (CSV数据):")
    print(x)
    print("R (s2p数据):")
    print(Z)
