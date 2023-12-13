import os
import numpy as np
import tensorflow as tf
from data_processing.preprocessing import feature_processing
import time
start_time = time.time()
# 获取当前工作目录
current_directory = "E:/program/machine_learning/model/"

# 指定模型文件的文件名
model_filename = "model_20.h5"

# 将目录和文件名连接起来，并将其转换为字节字符串
model_path = os.path.join(current_directory, model_filename)

# 加载模型
model = tf.keras.models.load_model(model_path)

# 现在你可以像以前一样使用加载的模型
test_array = feature_processing()
predict = model.predict(test_array)

def calculate_s_parameters(Z0_real, Z0_imag, ZL_real, ZL_imag):
    # 将复数表示为实部和虚部
    Z0 = complex(Z0_real, Z0_imag)
    ZL = complex(ZL_real, ZL_imag)

    # 计算S11和S12
    S11 = (ZL / Z0 - 1) / (ZL / Z0 + 1)
    S12 = 2 * ZL / (Z0 * (1 - S11))
    return S11, S12

# 计算S参数
for i in range(len(predict)):
    Z0_real, Z0_imag, ZL_real, ZL_imag = predict[i]
    S11, S12 = calculate_s_parameters(Z0_real, Z0_imag, ZL_real, ZL_imag)

    print(np.array([S11.real, S11.imag, S12.real, S12.imag]))

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

print(extract_data_from_generated_txt(19))
end_time = time.time()
print(f" Duration: {end_time - start_time:.2f} seconds")
