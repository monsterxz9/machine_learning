"""
模型训练模块。
支持单模型（频率作为输入）和多模型两种训练模式。
通过 MODEL_MODE 配置切换。
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
    MODEL_MODE,
    MODEL_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    # 单模型
    SINGLE_MODEL_PATH,
    SINGLE_MODEL_LAYERS,
    SINGLE_INPUT_DIM,
    SINGLE_L2_LAMBDA,
    NORMALIZATION_STATS_PATH,
    # 多模型
    NUM_MODELS,
    MULTI_MODEL_LAYERS,
    MULTI_INPUT_DIM,
    MULTI_L2_LAMBDA,
)
from preprocessing import get_training_set, get_unified_training_set


def setup_gpu():
    """配置 GPU 内存按需增长，避免一次性占满显存。"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def create_model(input_dim, hidden_layers, l2_lambda):
    """根据输入维度和隐藏层配置创建全连接网络。"""
    regularizer = l2(l2_lambda)
    layers = [
        Dense(
            hidden_layers[0],
            activation="relu",
            input_shape=(input_dim,),
            kernel_regularizer=regularizer,
        )
    ]
    for units in hidden_layers[1:]:
        layers.append(Dense(units, activation="relu", kernel_regularizer=regularizer))
    layers.append(Dense(4))

    model = Sequential(layers)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    )
    return model


def train_single_model(save_dir=MODEL_DIR):
    """单模型模式：一个模型学习所有频率点。"""
    os.makedirs(save_dir, exist_ok=True)

    print("加载统一训练数据（频率作为第 5 输入）...")
    X, Z, feat_mean, feat_std, freq_mean, freq_std = get_unified_training_set()
    print(f"训练数据 — X: {X.shape}, Z: {Z.shape}")

    model = create_model(SINGLE_INPUT_DIM, SINGLE_MODEL_LAYERS, SINGLE_L2_LAMBDA)
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X,
        Z,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        validation_split=VALIDATION_SPLIT,
    )

    model.save(SINGLE_MODEL_PATH)
    print(f"模型已保存: {SINGLE_MODEL_PATH}")

    # 保存标准化参数，评估时需要
    np.savez(
        NORMALIZATION_STATS_PATH,
        feat_mean=feat_mean,
        feat_std=feat_std,
        freq_mean=np.array(freq_mean),
        freq_std=np.array(freq_std),
    )
    print(f"标准化参数已保存: {NORMALIZATION_STATS_PATH}")

    # 抽样预测对比
    sample_idx = np.random.choice(len(X), 5, replace=False)
    predictions = model.predict(X[sample_idx])
    print(f"真实值 (抽样):\n{Z[sample_idx]}")
    print(f"预测值 (抽样):\n{predictions}")


def train_multi_models(save_dir=MODEL_DIR):
    """多模型模式：训练 NUM_MODELS 个独立模型。"""
    os.makedirs(save_dir, exist_ok=True)

    datasets = [get_training_set(i) for i in range(NUM_MODELS)]
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    for i in range(NUM_MODELS):
        x_train, Z = datasets[i]
        model = create_model(MULTI_INPUT_DIM, MULTI_MODEL_LAYERS, MULTI_L2_LAMBDA)

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

    if MODEL_MODE == "single":
        print("训练模式: 单模型（频率作为输入）")
        train_single_model()
    else:
        print("训练模式: 多模型（20 个独立模型）")
        train_multi_models()

    elapsed = time.time() - start_time
    print(f"\n总训练时间: {elapsed:.2f} 秒")
