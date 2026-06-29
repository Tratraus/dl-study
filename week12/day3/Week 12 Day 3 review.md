# Week 12 Day 3 Review：位置编码

## 代码结构分析

```python
class SinusoidalPositionalEncoding(nn.Module):
    __init__(d_model, max_len, dropout)   # 构建 PE 矩阵，注册为 buffer
    forward(x) → x + PE[:, :L, :] + dropout
```

结构极简——`__init__` 预计算正弦编码矩阵，`forward` 只做加法和 dropout。

## 数据流 / Shape 变化追踪

```
__init__:
    pos:      (max_len, 1)          = (100, 1)
    i:        (d_model/2,)          = (32,)
    div_term: (32,)                 # exp(-2i * log(10000) / d_model)
    PE:       (100, 64)             # sin/cos 填充
    PE:       (1, 100, 64)          # unsqueeze(0)，注册为 buffer

forward:
    x:        (B, L, d_model)       = (2, 50, 64)
    PE[:, :L, :]: (1, 50, 64)       # 取前 L 个位置，广播到 batch
    output:   (2, 50, 64)           # x + PE，shape 不变
```

**关键点**：PE 的 batch 维度是 1，通过广播自动扩展到任意 batch size。

## 关键知识点

### 1. register_buffer vs self.PE

| | `self.PE = tensor` | `register_buffer('PE', tensor)` |
|---|---|---|
| 参与梯度更新 | ❌（不是 nn.Parameter） | ❌ |
| `model.to(device)` 自动迁移 | ❌ | ✅ |
| `model.state_dict()` 包含 | ❌ | ✅ |
| `model.eval()` 行为 | 无区别 | 无区别 |

`register_buffer` 的核心价值：**随模型保存/加载、随设备迁移**。位置编码虽然是固定的，但必须和模型一起保存、一起搬到 GPU，所以用 buffer。

### 2. div_term 与频率的关系

```
div_term[i] = exp(-2i * log(10000) / d_model)
            = 1 / 10000^(2i/d_model)
```

- i 小（低维度）→ div_term 大 → `pos * div_term` 变化快 → sin 波频率高 → 热图中条纹密
- i 大（高维度）→ div_term 小 → `pos * div_term` 变化慢 → sin 波频率低 → 热图中条纹宽

**直觉**：低维度像秒针（快速变化，区分相邻位置），高维度像时针（缓慢变化，区分远距离位置）。不同频率的组合让每个位置都有唯一的"指纹"。

### 3. 正弦编码的外推能力

正弦编码是公式生成的，理论上可以计算任意位置的编码。ESM-2 的可学习编码最大 1026 位，超过就报错。但正弦编码的外推也有局限——超过训练时见过的长度后，编码虽然能算，但模型从未在这些位置上训练过，效果不保证。

## 踩坑与易错点

### 1. div_term 的两种写法

```python
# Week 5（math 模块）
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

# Week 12（torch 模块）
div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0)) / d_model))
```

两种写法结果相同。但 `torch.log(torch.tensor(10000.0))` 创建了一个临时 tensor，如果在 GPU 上会有微小的设备迁移开销。`math.log(10000.0)` 是纯 Python float，更轻量。实际差异可忽略。

### 2. 验证方式的改进

| | Week 5 | Week 12 |
|---|--------|---------|
| 位置唯一性 | `assert not allclose` | `print(allclose)` |
| 打乱顺序测试 | ❌ 没有 | ✅ 有（验证 PE 打破置换不变性） |
| 热图 | 有 | 有（更完整的坐标轴标注） |

Week 12 新增的**打乱顺序测试**是关键验证——它证明了 PE 确实解决了"置换不变性"问题：相同的嵌入向量，加了 PE 后，打乱顺序就得到不同的输出。

### 3. 热图保存路径

Task 中写的是 `pe_heatmap.png`（当前目录），实际保存为 `week12/day3/pe_heatmap.png`（项目根目录）。这是合理的调整，保持了仓库结构整洁。

## 输出问题回答评估

**Q1**：✅ 正确。核心点抓住了——buffer 不参与梯度更新，但随模型保存/加载。可以补充一点：`model.to(device)` 时 buffer 也会自动搬到 GPU。

**Q2**：✅ 正确。低维度频率高（条纹密）、高维度频率低（条纹宽），对应 div_term 的变化。回答中的"低维度更平滑，高维度更敏感"表述需要修正——应该是**低维度变化快（敏感），高维度变化慢（平滑）**。但从最终答案来看，理解是对的。

**Q3**：✅ 正确。ESM-2 超过 1026 会报错/截断；正弦编码可以动态生成任意长度。

## 与 Week 5 对比

| 维度 | Week 5 Day 4 | Week 12 Day 3 |
|------|-------------|--------------|
| 实现方式 | 填骨架（有提示） | 独立完成 |
| div_term 写法 | math.log | torch.log |
| 验证项 | shape + 位置唯一性 | shape + 位置唯一性 + **打乱顺序** + 热图 |
| 理解深度 | 跑通即可 | 能解释频率/外推/buffer 机制 |

**Day 3 达成目标**。正弦位置编码实现正确，新增的打乱顺序测试证明了 PE 的核心作用——打破 Self-Attention 的置换不变性。
