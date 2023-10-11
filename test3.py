from data_processing.preprocessing import get_training_set
import numpy as np
from sklearn.linear_model import LinearRegression


def mono_linear_regression(x, t, example):
    # 创建线性回归模型
    reg = LinearRegression()
    # 使用数据拟合模型
    reg.fit(x, t[:, 0])  # 切片操作选择每一行的第一个元素
    # 进行预测
    prediction = reg.predict(example)

    return prediction
# 示例数据
x,t= get_training_set(19)


print(x,t)
example = np.array([[2.20E-05,	0.000346,	0.000175,	13.5]])  # 示例输入

prediction = mono_linear_regression(x, t, example)
print("Prediction:", prediction)
