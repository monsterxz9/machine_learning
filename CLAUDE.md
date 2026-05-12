# CLAUDE.md

> 给 AI assistant 的项目上下文与约定。修改代码前先读完。

## 项目本质

微带线 S 参数的神经网络代理模型(信通院 CAICT 先进计算大赛参赛项目)。
输入 4 个几何参数 `(W, L, H, Er)` + 频率 → 输出 `[S11.real, S11.imag, S12.real, S12.imag]`,
替代传统电磁仿真求解,实现"几何参数 → 散射参数"的快速预测。

## 架构 invariants — 别 undo

- **PyTorch only。** 2026-05 重构从 TF Keras → PyTorch。不要加回 TF/Keras 代码或 `.h5` 模型格式。
- **Single model only。** v0.1 有过 multi 模式(20 个独立模型,每频段一个),已经删除。
  那是当年单模型欠拟合时的 workaround,**不是正解**。新版用 Fourier features 让
  单模型有能力学振荡曲线,效果远好于 multi。**不要加回 multi**。
- **Fourier features 是核心。** `model.py` 里的 `FourierFeatures` 把标量频率编码成
  多尺度 sin/cos。这是让 ReLU/GELU MLP 能拟合 S 参数振荡曲线的关键。不要换成裸标量。
- **按几何样本切 train/val。** `preprocessing.py` 的 `build_datasets` 用 numpy
  random shuffle 在**几何样本维度**上切。绝对不要按 `(sample, freq)` 展平后再切,
  会让同一组 (W,L,H,Er) 的不同频率点跨集泄漏,val loss 失效。

## Known gotchas

### MPS NaN — 永远别加回 `non_blocking=True`

PyTorch MPS 后端有 race condition:`.to(device, non_blocking=True)` 会在数据传输
完成前返回 tensor。后续 forward 看到 garbage memory → loss = NaN。

```python
# BAD (MPS race condition, loss = NaN):
geo = geo.to(device, non_blocking=True)

# GOOD:
geo = geo.to(device)
```

CPU 单元测试通过、smoke test 通过都**不会**暴露这个 bug — 必须真训练才显形。
已在 commit `6345d79` 修复。如果哪天 PyTorch 修了 MPS 这个 bug 可以再考虑启用,
但 PR 之前必须**真跑一次完整训练**确认 loss 不爆。

### CSV 是 7 列不是 4 列

`mline_size.csv` 列是 `W,L,H,Er,Cond,T,TanD`。代码只用前 4 列,`load_features`
里有 `if data.shape[1] > 4: data = data[:, :4]` 处理这个。如果想用 `Cond / T / TanD`
扩展输入,这条得改,同时模型 `geo_dim` 也要跟着改。

## 数据约定 — 仓库不含数据

代码期望 `data/` 目录在项目根。本地需要软链(或用环境变量覆盖路径):

```bash
# 专题赛数据 = 5000 样本,跟默认 config (NUM_S2P_FILES=5000) 完美对齐
ln -s 微带线赛题数据/专题赛数据 data

# 决赛数据 = 100K 样本,需要先把 config.NUM_S2P_FILES 改成 100000
ln -s 微带线赛题数据/决赛数据 data
```

测试集 (`case1.s2p` ... `case10.s2p` + `test.csv`)**当前没有本地数据**,
`load_model.py` 跑评估会失败。要做评估需要:
- 找赛题方原始测试集,或者
- 从训练数据切一部分当 test(改 `get_test_data.py`)

## 常用命令

```bash
uv sync                                              # 装/更新依赖
uv run pytest tests/                                 # smoke test (CPU,不需数据)
uv run ruff check . --fix && uv run ruff format .    # lint + format
uv run python train_model.py                         # 训练 (需要 data/)
uv run python load_model.py                          # 评估 (需要 data/CaseData/)
```

## 默认超参 (`config.py`)

| 项 | 值 |
|----|----|
| `BATCH_SIZE` | 4096 |
| `MAX_EPOCHS` | 500 |
| `EARLY_STOPPING_PATIENCE` | 30 |
| `LEARNING_RATE` | 1e-3 |
| `WEIGHT_DECAY` | 1e-4 |
| `HIDDEN_LAYERS` | `[256, 256, 128, 128, 64]` + GELU |
| `NUM_FOURIER` | 16 (输入维度 = 4 + 2×16 = 36) |
| `VAL_RATIO` | 0.2 |
| `SEED` | 42 |

Apple Silicon MPS 上 **~2.4 秒/epoch** (专题赛 5K 数据 = 400K 训练样本)。
500 epoch 估计在 100-200 epoch 触发 early stop,**10-20 分钟内出模型**。

## Device

`get_device()` 自动检测:`CUDA > MPS > CPU`。不要硬编码 device。

## 重构历史(为何如此设计)

v0.1 → v0.2(2026-05 重构):

| 维度 | v0.1 | v0.2 |
|------|------|------|
| 框架 | TensorFlow Keras (3 层 MLP) | PyTorch (5+1 层 MLP) |
| 模式 | single + multi 双模式 | 只保留 single |
| 频率 | 标量直接塞 | Fourier sin/cos 16 scale |
| Val 切分 | `validation_split=0.2`(末尾偏倚 + 同几何泄漏) | 按几何样本 random shuffle |
| 参数量 | ~10K | ~130K |
| 包管理 | `requirements.txt` | `pyproject.toml` + uv |

之前 single 模式效果不好就是因为:网络太小 + 频率裸塞 + val 切分有偏。v0.2 三个一起改,
single 才真正发挥威力。**别因为"single 看起来比 multi 复杂"就回退到 multi。**
