# Week 12 · Day 3：位置编码

## 理论：为什么 Attention 不知道顺序

### Self-Attention 是"置换不变"的

回忆 Day 1 的公式：

$$\text{output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

把输入序列的行顺序打乱（比如把氨基酸序列随机重排），Q、K、V 的行也跟着重排，最终 output 的行也只是重排——**每个位置的输出值完全不变**。

这意味着对 Self-Attention 来说：
```
MKTAY  和  YTAKM  和  AYMKT
```
是**完全等价**的输入，模型无法区分。

但蛋白质序列的顺序至关重要——N 端信号肽、活性位点的相对位置都依赖序列顺序。

---

### 解决方案：给每个位置加一个"位置指纹"

在输入进入 Attention 之前，给每个位置的向量**加上一个只与位置有关的向量**：

$$\text{input\_with\_pos}[i] = \text{embedding}[i] + \text{PE}[i]$$

这样位置 3 和位置 7 的向量即使氨基酸相同，也会因为加了不同的 PE 而产生不同的表示。

---

### 正弦位置编码的公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $$pos$$：位置索引（0, 1, 2, ...）
- $$i$$：维度索引（0, 1, ..., $$d_{model}/2 - 1$$）
- 偶数维用 sin，奇数维用 cos

**直觉**：不同维度用不同频率的正弦波，低维度变化慢（捕捉长程位置关系），高维度变化快（捕捉局部位置关系）。就像时钟的时针、分针、秒针——不同频率组合唯一标识每个时刻。

---

### 正弦编码 vs 可学习编码

| | 正弦编码 | 可学习编码（ESM-2 用的） |
|--|---------|----------------------|
| 参数量 | 0（固定公式） | $$max\_len \times d_{model}$$ |
| 泛化到更长序列 | ✅ 可以外推 | ❌ 超出训练长度则未定义 |
| 训练数据少时 | ✅ 稳定 | ⚠️ 可能过拟合 |
| 性能 | 略低 | 略高（数据充足时） |

ESM-2 用可学习编码，但限制了最大长度 1024——这正是 Day 7 复杂度分析的背景。

---

## 代码任务

新建 `week12/day3/positional_encoding.py`：

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构建位置编码矩阵 PE: (max_len, d_model)
        PE = torch.zeros(max_len, d_model)

        # pos: (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term: (d_model/2,)
        # 公式：10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        i = torch.arange(0, d_model, 2).float()           # 偶数索引：0,2,4,...
        div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数维用 sin，奇数维用 cos
        PE[:, 0::2] = ___   # sin(pos * div_term)
        PE[:, 1::2] = ___   # cos(pos * div_term)

        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        # 增加 batch 维：(1, max_len, d_model)
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
          x: (B, L, d_model)  ← 输入嵌入
        返回：
          x + PE[:, :L, :]    ← 加上位置编码后的嵌入，shape 不变
        """
        L = x.shape[1]
        x = x + ___           # 取 PE 的前 L 个位置
        return self.dropout(x)


# ── 验证 ──────────────────────────────────────────────────
if __name__ == "__main__":
    d_model = 64
    max_len = 100

    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # 1. shape 验证
    x = torch.zeros(2, 50, d_model)
    out = pe(x)
    print(f"输入  shape: {x.shape}")
    print(f"输出  shape: {out.shape}")   # 期望: (2, 50, 64)

    # 2. 验证：不同位置的 PE 向量不同
    pe_matrix = pe.PE.squeeze(0)         # (max_len, d_model)
    print(f"\n位置 0 和位置 1 的 PE 是否相同: {torch.allclose(pe_matrix[0], pe_matrix[1])}")
    print(f"位置 0 和位置 0 的 PE 是否相同: {torch.allclose(pe_matrix[0], pe_matrix[0])}")

    # 3. 验证：打乱顺序后输出不同
    torch.manual_seed(0)
    x_ordered   = torch.randn(1, 5, d_model)
    x_shuffled  = x_ordered[:, [2, 0, 4, 1, 3], :]   # 打乱顺序

    out_ordered  = pe(x_ordered)
    out_shuffled = pe(x_shuffled)
    print(f"\n加位置编码后，打乱顺序的输出 ≠ 原顺序: {not torch.allclose(out_ordered, out_shuffled)}")

    # 4. 热图可视化
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        pe_matrix.numpy(),
        aspect='auto',
        cmap='RdBu',
        vmin=-1, vmax=1
    )
    ax.set_xlabel("Dimension index (d_model)")
    ax.set_ylabel("Position (pos)")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("pe_heatmap.png", dpi=150)
    plt.show()
    print("\n热图已保存为 pe_heatmap.png")
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| `out.shape` | `(2, 50, 64)` |
| 位置 0 和位置 1 的 PE 不同 | `False`（不相同） |
| 打乱顺序后输出不同 | `True` |
| 热图能看到周期性条纹 | 低维度条纹宽，高维度条纹密 |

---

## 输出问题

**Q1**：`register_buffer` 和直接把 PE 存成 `self.PE` 有什么区别？为什么位置编码要用 `register_buffer`？

**Q2**：热图中，左侧（低维度）的条纹比右侧（高维度）宽，这对应公式里的哪个部分？直觉上代表什么？

**Q3**：ESM-2 的可学习位置编码最大长度是 1026（1024 + 2个特殊 token）。如果你要处理一条长度 2000 的蛋白质序列，用 ESM-2 会遇到什么问题？用正弦编码会遇到同样的问题吗？