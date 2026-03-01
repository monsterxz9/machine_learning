"""
集中配置管理模块。
所有路径和超参数统一在此处管理，便于跨环境迁移。
"""

import os

# 项目根目录（自动检测）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== 数据路径配置 ==========
# 训练特征 CSV 文件路径
FEATURE_CSV_PATH = os.environ.get(
    "FEATURE_CSV_PATH",
    os.path.join(PROJECT_ROOT, "data", "mline_size.csv"),
)

# 训练用 S2P 文件目录
TRAIN_S2P_DIR = os.environ.get(
    "TRAIN_S2P_DIR",
    os.path.join(PROJECT_ROOT, "data", "s2p"),
)

# 测试用 CSV 文件路径
TEST_CSV_PATH = os.environ.get(
    "TEST_CSV_PATH",
    os.path.join(PROJECT_ROOT, "data", "test.csv"),
)

# 测试用 S2P 文件目录
TEST_S2P_DIR = os.environ.get(
    "TEST_S2P_DIR",
    os.path.join(PROJECT_ROOT, "data", "CaseData"),
)

# 模型保存目录
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(PROJECT_ROOT, "model"),
)

# ========== 训练超参数 ==========
NUM_MODELS = 20               # 训练模型数量
NUM_S2P_FILES = 5000          # 训练用 S2P 文件数量
POINTS_PER_FILE = 100         # 每个 S2P 文件的频率点数
TOTAL_POINTS = NUM_S2P_FILES * POINTS_PER_FILE  # 500000

BATCH_SIZE = 4000
MAX_EPOCHS = 1_000_000
EARLY_STOPPING_PATIENCE = 1000
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
L2_LAMBDA = 0.01

# 测试集相关
NUM_TEST_CASES = 10           # 测试 S2P 文件数量
NUM_TEST_FEATURES = 4         # 输入特征数（W, L, H, Er）
