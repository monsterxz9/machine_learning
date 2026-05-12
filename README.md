# machine_learning

> 信通院(CAICT)先进计算大赛参赛项目:微带线散射参数的神经网络代理模型

用 **PyTorch** 训练一个 MLP,从微带线的几何/材料参数预测散射参数 S11/S12(复数,拆成实部/虚部),替代传统的电磁仿真求解过程。

## 问题

**输入**:微带线 4 个物理参数

| 符号 | 含义 |
|------|------|
| `W`  | 线宽 |
| `L`  | 线长 |
| `H`  | 介质厚度 |
| `Er` | 相对介电常数 |

**输出**:某频率点下的散射参数(实部 + 虚部)

| 维度 | 含义 |
|------|------|
| 0    | S11.real |
| 1    | S11.imag |
| 2    | S12.real |
| 3    | S12.imag |

**数据**:5000 个 `.s2p` 文件 × 100 个频率点 = 500K 训练样本;10 个测试 case。S 参数文件由 [`scikit-rf`](https://scikit-rf.readthedocs.io/) 解析。

## 模型

单一统一模型,频率作为输入。三个关键设计:

### 1. Fourier 频率编码

把标量频率编码成 `2*K` 维 sin/cos 多尺度特征(几何递增 scale `1, 2, 4, ..., 2^(K-1)`),
缓解 ReLU/GELU 网络对高频信号的 spectral bias(优先学低频)。这对预测震荡的 S 参数曲线
比直接拿标量频率作输入要强很多。

### 2. 更深的 MLP

```
36 → 256 → 256 → 128 → 128 → 64 → 4   (GELU 激活)
```

参数约 130K,比旧版(`128→64→32`,~10K 参数)大一个数量级,有足够容量逼近多频段振荡。

### 3. 无泄漏的 train/val 切分

**按几何样本** random shuffle 后切 80/20,同一组 (W, L, H, Er) 的所有频率点要么全在
训练,要么全在验证。避免旧版 `validation_split=0.2` 取末尾 20% 导致的偏倚 + 同
几何跨集泄漏。

## 代码结构

```
config.py          统一管理路径/超参,所有配置走环境变量
preprocessing.py   读 CSV 特征 + 解析 .s2p + Z-Score 标准化 + Dataset 构造
model.py           FourierFeatures + MicrostripMLP (nn.Module)
train_model.py     训练入口,Adam + EarlyStopping + 自动保存训练曲线
load_model.py      加载已训练模型,跑测试集 + 画 true/pred/误差对比图
get_test_data.py   从测试 .s2p 提取特征 + 全频段 S 参数
data_visual.py     训练特征 (W, L, H, Er) 分布散点图
tests/test_smoke.py  smoke test:模型 forward/backward/save/load (不依赖真实数据)
```

## 跑起来

```bash
# 1. 装依赖 (uv 自动管理 Python 3.13 + torch + scikit-rf 等)
uv sync

# 2. Smoke test (不需要数据)
uv run pytest tests/ -v

# 3. 准备数据(本仓库未包含,需赛题方提供)
#    data/mline_size.csv         5000 行 [W, L, H, Er]
#    data/s2p/{1..5000}.s2p      S 参数训练文件
#    data/CaseData/              测试集 S 参数 (case1.s2p .. case10.s2p)
#    data/test.csv               测试集特征

# 4. 训练
uv run python train_model.py

# 5. 评估(画对比图)
uv run python load_model.py

# 6. 可视化特征分布
uv run python data_visual.py
```

## 设备

自动检测,优先级:CUDA > MPS (Apple Silicon) > CPU。

## 一些工程细节

- 标准化参数(均值/方差)随模型一起保存到 `norm_stats.npz`,推理时复用,保证训练/推理统计量一致
- Checkpoint 保存模型 state + 架构 config,`load_model()` 自动按 config 重建网络,无需手动同步超参
- Early stopping `patience=30` + best weights 还原,代替原本 `patience=1000` 的"放养"训练
- macOS Apple Silicon 通过 MPS 后端加速,无需 CUDA

## v0.2 重构说明

相比 v0.1 (TensorFlow + 20 个独立模型):

| 维度 | v0.1 | v0.2 (当前) |
|------|------|-------------|
| 框架 | TensorFlow Keras | **PyTorch** |
| 模型保存 | `.h5` | `.pt` |
| 架构 | 1 个单模型 + 20 个独立模型双模式 | **只保留单模型** + Fourier features |
| 频率处理 | 直接作标量输入 | **多尺度 sin/cos 编码** |
| 验证集 | `validation_split=0.2`(末尾偏倚) | **按几何样本随机切**(无泄漏) |
| 包管理 | `requirements.txt` | `pyproject.toml` + `uv` |
| 实验 | 无 | smoke test (pytest) |

## License

学习/比赛用途,代码部分供参考。数据归赛题方所有。
