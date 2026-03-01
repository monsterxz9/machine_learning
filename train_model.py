"""
模型训练模块。
创建并训练多个神经网络模型，用于预测微带线 S 参数。
"""

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from config import (
    MODEL_DIR,
    NUM_MODELS,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    L2_LAMBDA,
    NUM_TEST_FEATURES,
)
from preprocessing import get_training_set


def setup_gpu():
    """配置 GPU 内存按需增长，避免一次性占满显存。"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def create_model():
    """创建一个用于 S 参数预测的全连接神经网络模型。"""
    regularizer = l2(L2_LAMBDA)
    model = Sequential([
        Dense(1024, activation="relu", input_shape=(NUM_TEST_FEATURES,), kernel_regularizer=regularizer),
        Dense(512, activation="relu", kernel_regularizer=regularizer),
        Dense(256, activation="relu", kernel_regularizer=regularizer),
        Dense(128, activation="relu", kernel_regularizer=regularizer),
        Dense(64, activation="relu", kernel_regularizer=regularizer),
        Dense(32, activation="relu", kernel_regularizer=regularizer),
        Dense(4),
    ])
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    )
    return model


def train_all_models(save_dir=MODEL_DIR):
    """训练所有模型并保存到指定目录。"""
    os.makedirs(save_dir, exist_ok=True)

    datasets = [get_training_set(i) for i in range(NUM_MODELS)]
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE
    )

    for i in range(NUM_MODELS):
        x_train, Z = datasets[i]
        model = create_model()

        print(f"\n===== 训练模型 {i + 1}/{NUM_MODELS} =====")
        model.fit(
            x_train,
            Z,
            batch_size=BATCH_SIZE,
            epochs=MAX_EPOCHS,
            callbacks=[early_stopping],
            validation_split=VALIDATION_SPLIT,
        )

        model_path = os.path.join(save_dir, f"model_{i + 1}.h5")
        model.save(model_path)
        print(f"模型已保存: {model_path}")

        predictions = model.predict(x_train)
        print(f"模型 {i + 1} - 真实值 (前5行):\n{Z[:5]}")
        print(f"模型 {i + 1} - 预测值 (前5行):\n{predictions[:5]}")


if __name__ == "__main__":
    setup_gpu()
    start_time = time.time()
    train_all_models()
    elapsed = time.time() - start_time
    print(f"\n总训练时间: {elapsed:.2f} 秒")
