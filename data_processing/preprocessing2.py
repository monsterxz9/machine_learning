import numpy as np
import csv
import os
import skrf as rf
import pandas as pd

def csv_to_numpy(csv_file_path=r'/E:/微带线赛题数据/专题赛数据/mline_size.csv'):
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    extracted_data = data.iloc[:, :4]
    extracted_data = extracted_data.values
    return extracted_data

def import_from_s2p(n, start=1, end=5000, directory_path=r'/E:/微带线赛题数据/专题赛数据/s2p/'):
    # Initialize an empty list to store the extracted rows
    extracted_rows = []
    # Loop over the specified range of file numbers
    for i in range(start, end + 1):
        filename = f'{i}.s2p'
        filepath = os.path.join(directory_path, filename)
        # Read the S2P file using skrf
        ntwk = rf.Network(filepath)
        # Extract the complex-valued S-parameters
        s_params_complex = ntwk.s
        # Check if there are enough rows to extract the desired range
        if n + 5 <= s_params_complex.shape[0]:
            # Extract the desired rows and calculate their average
            extracted_row = s_params_complex[n:n + 5].mean(axis=0)
            # Append the average row to the list
            extracted_rows.append(extracted_row)
    # Check if any rows were extracted
    if extracted_rows:
        # Concatenate the extracted rows into a single NumPy array
        extracted_rows_array = np.array(extracted_rows)
    else:
        # Return an empty NumPy array if no rows were extracted
        extracted_rows_array = np.array([])
    return extracted_rows_array

def get_training_set(n):
    # 从CSV文件获取x
    x = csv_to_numpy()
    # 处理.s2p文件并获取平均值列表
    n = 5 * n
    T = import_from_s2p(n)
    return x, T

# 示例用法
if __name__ == "__main__":
    x, selected_training_set = get_training_set(0)
    print("x (CSV数据):")
    print(x)
    print(x.shape)
    print("T:")
    print(selected_training_set)
    print(selected_training_set.shape)
