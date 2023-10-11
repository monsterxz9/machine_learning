import numpy as np
import csv
import os
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

def extract_data_from_generated_txt(n, output_folder_path=r'E:/微带线赛题数据/专题赛数据', filename='output.txt'):
   # 初始化一个空列表，用于存储从大的txt文件中读取的数据
    data_list = []

    # 构建大的txt文件的完整路径
    big_txt_file_path = os.path.join(output_folder_path, filename)

    with open(big_txt_file_path, 'r') as big_txt_file:
        lines = big_txt_file.readlines()

        # 从第n行开始，然后每次跳过20行，直到文档结束
        start_line = n
        skip = 20
        while start_line < len(lines):
            line = lines[start_line]
            values = line.strip().split()[1:5]
            # 添加数据有效性检查，确保只有有效的数字被转换为浮点数
            valid_numbers = []
            for val in values:
                try:
                    number = np.float32(val)
                    valid_numbers.append(number)
                except ValueError:
                    pass
            if len(valid_numbers) == 4:
                data_list.append(valid_numbers)
            start_line += skip

    # Convert the data list into a NumPy array
    data_array = np.array(data_list)

    # 返回提取的数据数组
    return data_array

# 使用示例，提取数据数组




def get_training_set(n):
    # 从CSV文件获取x
    csv_file_path = r'E:\mline_size.csv'
    x = csv_to_numpy(csv_file_path)


    resulting_data_array = extract_data_from_generated_txt(n)
    return x, resulting_data_array

# 示例用法
if __name__ == "__main__":
    # 调用get_training_set方法获取x
    x, t = get_training_set(19)

    # 打印x, t的内容
    print("x (CSV数据):")
    print(x)
    print("T (s2p数据):")
    print(t)
