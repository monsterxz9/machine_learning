# machine_learning

> 信通院(CAICT)先进计算大赛参赛项目:微带线散射参数的神经网络代理模型

用 TensorFlow 训练一个全连接网络,直接从微带线的几何/材料参数预测散射参数 S11/S12(复数,拆成实部/虚部),替代传统的电磁仿真求解过程。

## 问题

**输入**:微带线 4 个物理参数

| 符号 | 含义 |
|------|------|
| `W`  | 线宽 |
| `L`  | 线长 |
| `H`  | 介质厚度 |
| `Er` | 相对介电常数 |

**输出**:某频率点下的散射参数

| 维度 | 含义 |
|------|------|
| 0    | S11.real |
| 1    | S11.imag |
| 2    | S12.real |
| 3    | S12.imag |

**数据**:5000 个 `.s2p` 文件 × 100 个频率点 = 500K 训练样本;10 个测试 case。S 参数文件由 [`scikit-rf`](https://scikit-rf.readthedocs.io/) 解析。

## 两种模式

通过环境变量 `MODEL_MODE` 切换:

### `single`(推荐)— 频率作输入的统一模型

- 一个模型学全频段
- 输入 5 维 `(W, L, H, Er, freq)`
- 网络:`Dense(128) → Dense(64) → Dense(32) → Dense(4)`,L2 正则 `1e-4`
- 产物:`model_unified.h5` + `normalization_stats.npz`

### `multi`— 每频段一个模型(兼容旧版本)

- 20 个独立模型,每个负责一组频率点
- 输入 4 维(不含频率)
- 网络更宽:`1024 → 512 → 256 → 128 → 64 → 32 → 4`
- 产物:`model_1.h5` … `model_20.h5`

## 代码结构

```
config.py          统一管理路径/超参,所有配置走环境变量
preprocessing.py   读 CSV 特征 + 解析 .s2p (scikit-rf) + Z-Score 标准化
train_model.py     训练入口,GPU 内存按需增长 + EarlyStopping
load_model.py      加载已训练模型,跑测试集评估
get_test_data.py   从测试 .s2p 提取标签
data_visual.py     S 参数可视化
```

## 跑起来

```bash
# 1. 装依赖
pip install -r requirements.txt
# tensorflow>=2.10, scikit-rf>=0.25, numpy, pandas, matplotlib

# 2. 准备数据(本仓库未包含,需赛题方提供)
#    data/mline_size.csv         5000 行 [W, L, H, Er]
#    data/s2p/{1..5000}.s2p      S 参数训练文件
#    data/CaseData/              测试集 S 参数
#    data/test.csv               测试集特征

# 3. 训练(默认 single 模式)
python train_model.py

# 切到 multi 模式
MODEL_MODE=multi python train_model.py

# 4. 评估
python load_model.py
```

## 一些工程细节

- GPU 内存按需增长(`tf.config.experimental.set_memory_growth`),避免一次性占满显存。
- 单模型版本把频率塞进特征向量(`np.repeat` + `np.tile`),避免训 100 个模型的冗余。
- 标准化参数随模型一起保存(`normalization_stats.npz`),推理时复用,保证训练/推理统计量一致。
- Early stopping `patience=1000` + `restore_best_weights`,数据量充足时给训练留足收敛空间。

## License

学习/比赛用途,代码部分供参考。数据归赛题方所有。
