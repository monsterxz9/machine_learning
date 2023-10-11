import torch
import torch.nn as nn
import torch.optim as optim
from data_processing.preprocessing import get_training_set

# 创建一个示例数据集（二维输入，二分类任务）
x_train, y_train = get_training_set(0)

# 将数据转换为PyTorch张量，并指定数据类型为float64
x_train = torch.tensor(x_train).double()
y_train = torch.tensor(y_train).double()

# 定义一个多层感知器模型
class MLP(nn.Module):
    # 修改模型的输入层大小
    input_size = 4  # 输入特征维度，与每次训练使用的输入数据维度相匹配

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x


# 初始化模型
input_size = 2  # 输入特征维度
hidden_size = 2  # 隐层维度
output_size = 1  # 输出维度
model = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

model = MLP(input_size, hidden_size, output_size)
model.double()  # 将模型的权重和偏差转换为float64

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        # 从训练集中取一行作为输入
        input_data = x_train[i]

        # 前向传播
        output = model(input_data)

        # 获取对应的标签
        label = y_train[i]

        # 计算损失
        loss = criterion(output, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
    predictions = model(test_data)
    predicted_labels = (predictions > 0.5).double()
    print("Predictions:")
    print(predicted_labels)
