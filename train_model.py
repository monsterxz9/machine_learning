import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocessing2 import get_training_set
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os

start_time = time.time()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 定义创建模型的函数
l2_lambda = 0.01  # 你可以根据需要调整lambda的值
regularizer = l2(l2_lambda)
def create_model():
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(4,),  kernel_regularizer=regularizer),
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dense(256, activation='relu', kernel_regularizer=regularizer),
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dense(64, activation='relu', kernel_regularizer=regularizer),
        Dense(32, activation='relu', kernel_regularizer=regularizer),
        Dense(4)
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

# 准备多个不同的数据集
num_models = 20
datasets = [get_training_set(i) for i in range(num_models)]
# 使用单个GPU策略
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
# 创建多个模型，每个模型在策略下运行
with strategy.scope():
    models = [create_model() for _ in range(num_models)]
# 创建保存模型的目录
model_save_dir = r'E:\program\machine_learning\new_model'
os.makedirs(model_save_dir, exist_ok=True)
# 训练所有模型并行
for i in range(num_models):
    x_train, Z = datasets[i]
    model = models[i]
    batch_size = 4000
    early_stopping = EarlyStopping(monitor='val_loss', patience=1000)  # 根据需要调整
    model.fit(x_train, Z, batch_size=batch_size, epochs=1000000, callbacks=[early_stopping], validation_split=0.2)
    # 保存模型
    model_save_path = os.path.join(model_save_dir, f"model_{i + 1}.h5")
    model.save(model_save_path)

    ZR_pred = model.predict(x_train)
    # 打印预测值
    print(f"Model {i + 1} - True values:")
    print(Z)
    print(f"Model {i + 1} - Predicted values:")
    print(ZR_pred)

end_time = time.time()
print(f"Total Training Time: {end_time - start_time:.2f} seconds")
