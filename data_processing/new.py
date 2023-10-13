import time
import skrf as rf
import numpy as np
import scipy.constants
start_time = time.time()
# 读取S2P文件

s2p_file = r'E:\微带线赛题数据\专题赛数据\s2p\1.s2p'  # 请替换为实际的S2P文件路径
network = rf.Network(s2p_file)

# 获取S参数数据
s_params = network.s  # 这是一个复数数组，包含S参数数据

# 获取频率
frequencies = network.f

# 计算特征阻抗Z0和归一化波长λg for 每个频率点
c = scipy.constants.speed_of_light  # 光速

for i, frequency in enumerate(frequencies):
    s11 = s_params[i, 0, 0]  # S11参数
    s22 = s_params[i, 1, 1]  # S22参数
    s12 = s_params[i, 0, 1]  # S12参数
    s21 = s_params[i, 1, 0]  # S21参数

    Z0 = np.sqrt((1 + s11) * (1 - s22) / (1 - s12 * s21))

    lambda0 = c / frequency
    lambda_g = lambda0 / Z0

    print(f'在 {frequency / 1e9} GHz 频率下：')
    print(f'特征阻抗 (Z0): {float(Z0.real):.2f} ohms')
    print(f'归一化波长 (λg): {lambda_g:.2f} units')
end_time = time.time()
print(f"Training Time: {end_time-start_time:.2f} seconds")
