"""集中配置:路径与超参,均可通过环境变量覆盖。"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ===== 数据路径 =====
FEATURE_CSV_PATH = os.environ.get(
    "FEATURE_CSV_PATH", os.path.join(PROJECT_ROOT, "data", "mline_size.csv")
)
TRAIN_S2P_DIR = os.environ.get("TRAIN_S2P_DIR", os.path.join(PROJECT_ROOT, "data", "s2p"))
TEST_CSV_PATH = os.environ.get("TEST_CSV_PATH", os.path.join(PROJECT_ROOT, "data", "test.csv"))
TEST_S2P_DIR = os.environ.get("TEST_S2P_DIR", os.path.join(PROJECT_ROOT, "data", "CaseData"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "model"))


# ===== 数据规模 =====
NUM_S2P_FILES = 5000
POINTS_PER_FILE = 100
TOTAL_POINTS = NUM_S2P_FILES * POINTS_PER_FILE
NUM_TEST_CASES = 10


# ===== 模型架构 =====
GEO_DIM = 4
NUM_FOURIER = 16  # 实际输入维度 = 4 + 2*16 = 36
HIDDEN_LAYERS = [256, 256, 128, 128, 64]


# ===== 训练超参 =====
BATCH_SIZE = 4096
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30
VAL_RATIO = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


# ===== 产物路径 =====
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.npz")
