import os

# 输入文件夹路径和输出文件路径
input_folder = r'E:\微带线赛题数据\专题赛数据\s2p_to_npy'
output_file = r'E:\微带线赛题数据\专题赛数据\output.txt'

# 打开输出文件以写入模式
with open(output_file, 'w') as output:
    # 按照1到5000的顺序合并文件
    for i in range(1, 5001):
        file_name = f"{i}.txt"
        file_path = os.path.join(input_folder, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    output.write(line)

print("合并完成！")