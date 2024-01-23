from preprocessing2 import get_training_set
import matplotlib.pyplot as plt  
import numpy as np  
X, y = get_training_set(0)
a = np.array([j for j in range(1, 5001, 1)])
# 计算X的每一列的平均值  
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
# 创建一个 2x2 的画布，并在每个部分画一个散点图  
plt.subplot(2, 2, 1)  # 第一个子图  
plt.scatter(a, X[:, 0], marker='.', c='r')  
plt.xlabel(f"Mean = {means[0]:.6f}  Std = {stds[0]:.6f}")
plt.title("W")
plt.subplot(2, 2, 2)  # 第二个子图  
plt.scatter(a, X[:, 1], marker='.', c='g')  
plt.xlabel(f"Mean = {means[1]:.6f}  Std = {stds[1]:.6f}")
plt.title("L")
plt.subplot(2, 2, 3)  # 第三个子图  
plt.scatter(a, X[:, 2], marker='.', c='b')  
plt.xlabel(f"Mean = {means[2]:.6f}  Std = {stds[2]:.6f}")
plt.title("H")
plt.subplot(2, 2, 4)  # 第四个子图  
plt.scatter(a, X[:, 3], marker='.', c='grey')  
plt.xlabel(f"Mean = {means[3]:.6f}  Std = {stds[3]:.6f}")
plt.title("Er")
# 调整子图之间的间距  
plt.tight_layout()  
# 显示图形  
plt.show()
