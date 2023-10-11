import os
import numpy as np

def process_s2p_files():
    # 输入文件夹路径
    input_folder_path = r'E:\微带线赛题数据\专题赛数据\s2p'

    # 输出文件夹路径
    output_folder_path = r'E:/微带线赛题数据/专题赛数据/s2p_to_npy'

    # 获取所有.s2p文件的文件名
    s2p_files = [file for file in os.listdir(input_folder_path) if file.endswith('.s2p')]

    # 初始化一个空列表，用于存储大的NumPy数组
    all_data = []

    # 初始化一个空字符串，用于存储大的txt文件内容
    big_txt_file_content = ""

    # 遍历每个.s2p文件
    for s2p_file in s2p_files:
        # 构建输入文件的完整路径
        input_file_path = os.path.join(input_folder_path, s2p_file)

        # 读取输入文件内容
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        # 初始化存储平均值的列表
        averages = []

        # 遍历输入文件的内容，从第二行开始每五行计算平均值并存储
        for i in range(1, len(lines), 5):
            data = lines[i:i + 5]
            data = [line.split() for line in data]
            data = np.array(data, dtype=float)
            average_data = np.mean(data, axis=0)
            averages.append(average_data)

        # 将平均值写入输出文件（txt文件）
        output_file_path = os.path.join(output_folder_path, s2p_file.replace('.s2p', '.txt'))
        with open(output_file_path, 'w') as output_file:
            for average_data in averages:
                output_file.write('\t'.join(map(str, average_data)) + '\n')

        print(f"平均值已保存到输出文件: {output_file_path}")
if __name__ == "__main__":
    # You can add code here that should only be executed when the script is run directly
    process_s2p_files()
