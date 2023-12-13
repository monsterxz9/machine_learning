import numpy as np
import os
import pandas as pd
import skrf as rf
from data_processing.preprocessing import csv_to_numpy
def import_test_data(csv_file_path=r'E:\微带线赛题数据\专题赛数据\test.csv'):
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    extracted_data = data.iloc[:, :4]
    extracted_data = extracted_data.values
    return extracted_data

def import_s2p_data(n=0):
    # 获取 S2P 文件列表
    s2p_files = [os.path.join(r'E:\微带线赛题数据\专题赛数据\CaseData', f'case{i}.s2p') for i in range(1, 11)]
    # 初始化一个空的结果数组
    result_array = np.empty((500000, 4))
    for i, s2p_file in enumerate(s2p_files):
        network = rf.Network(s2p_file)
        s_params = network.s
        frequencies = network.f
        for j, frequency in enumerate(frequencies):
            s11 = s_params[j, 0, 0]
            s22 = s_params[j, 1, 1]
            s12 = s_params[j, 0, 1]
            s21 = s_params[j, 1, 0]
    return s11, s12, s21, s22

def feature_of_data():
    csv_file_path = r'E:\mline_size.csv'
    data = csv_to_numpy(csv_file_path)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return data_mean, data_std