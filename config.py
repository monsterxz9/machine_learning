"""
集中配置管理模块。
所有路径和超参数统一在此处管理，便于跨环境迁移。
"""

import os

# 项目根目录（自动检测）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== 模式切换 ==========
# "single" = 单模型 + 频率输入，"multi" = 20 个独立模型（兼容旧模型）
MODEL_MODE = os.environ.get("MODEL_MODE", "single")

# ========== 数据路径配置 ==========
FEATURE_CSV_PATH = os.environ.get(
    "FEATURE_CSV_PATH",
    os.path.join(PROJECT_ROOT, "data", "mline_size.csv"),
)

TRAIN_S2P_DIR = os.environ.get(
    "TRAIN_S2P_DIR",
    os.path.join(PROJECT_ROOT, "data", "s2p"),
)

TEST_CSV_PATH = os.environ.get(
    "TEST_CSV_PATH",
    os.path.join(PROJECT_ROOT, "data", "test.csv"),
)

TEST_S2P_DIR = os.environ.get(
    "TEST_S2P_DIR",
    os.path.join(PROJECT_ROOT, "data", "CaseData"),
)

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(PROJECT_ROOT, "model"),
)

# ========== 通用超参数 ==========
NUM_S2P_FILES = 5000          # 训练用 S2P 文件数量
POINTS_PER_FILE = 100         # 每个 S2P 文件的频率点数
TOTAL_POINTS = NUM_S2P_FILES * POINTS_PER_FILE  # 500000

BATCH_SIZE = 4000
MAX_EPOCHS = 1_000_000
EARLY_STOPPING_PATIENCE = 1000
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

NUM_TEST_CASES = 10           # 测试 S2P 文件数量

# ========== 单模型配置 ==========
SINGLE_MODEL_PATH = os.path.join(MODEL_DIR, "model_unified.h5")
SINGLE_MODEL_LAYERS = [128, 64, 32]  # 隐藏层维度
SINGLE_INPUT_DIM = 5                  # W, L, H, Er, freq
SINGLE_L2_LAMBDA = 1e-4
NORMALIZATION_STATS_PATH = os.path.join(MODEL_DIR, "normalization_stats.npz")

# ========== 多模型配置（兼容旧模型） ==========
NUM_MODELS = 20
MULTI_MODEL_LAYERS = [1024, 512, 256, 128, 64, 32]
MULTI_INPUT_DIM = 4                   # W, L, H, Er
MULTI_L2_LAMBDA = 0.01
