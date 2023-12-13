import os
import numpy as np

def extract_data_from_generated_txt(n, output_folder_path=r'E:/微带线赛题数据/专题赛数据', filename='output.txt'):
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
    return data_array

if __name__ == "__main__":
    n = 0
    data_array = extract_data_from_generated_txt(n)
    print(data_array)
