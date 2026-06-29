# Week 12 · Day 1：Self-Attention 从零推导与实现

## 理论：Q、K、V 从哪里来，在做什么

### 直觉先行

想象你在查字典：
- **Query（Q）**：你想查的词（当前位置"想问什么"）
- **Key（K）**：字典里每个词条的索引（每个位置"能回答什么"）
- **Value（V）**：字典里每个词条的实际内容（每个位置"真正携带的信息"）

Self-Attention 做的事：**用 Q 去和所有 K 比对相似度，得到权重，再用权重加权求和所有 V**。

---

### 数学流程（逐步拆解）

输入：序列 $$X \in \mathbb{R}^{L \times d_{model}}$$，每行是一个位置的向量。

**第一步：线性投影，生成 Q、K、V**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$$，三个独立的可学习矩阵。

**第二步：计算注意力分数**

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{L \times L}$$

`scores[i, j]` = 位置 `i` 对位置 `j` 的原始关注度。

**第三步：Softmax 归一化**

$$\text{weights} = \text{softmax}(\text{scores}) \in \mathbb{R}^{L \times L}$$

每一行和为 1，变成概率分布。

**第四步：加权求和 V**

$$\text{output} = \text{weights} \cdot V \in \mathbb{R}^{L \times d_k}$$

---

### 为什么要除以 $$\sqrt{d_k}$$

$$d_k$$ 越大，$$QK^T$$ 的点积值越大（量级约为 $$d_k$$）。
值过大 → softmax 进入饱和区（梯度接近 0）→ 训练困难。
除以 $$\sqrt{d_k}$$ 把方差压回 1，保持 softmax 的梯度健康。

---

## 代码任务

新建 `week12/day1/self_attention.py`：

```python
import torch
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    参数：
      Q : (B, L, d_k)
      K : (B, L, d_k)
      V : (B, L, d_v)
      mask : (B, L, L) 或 None，mask=True 的位置设为 -inf

    返回：
      output      : (B, L, d_v)
      attn_weights: (B, L, L)  ← 每行和为 1

    步骤：
      1. 计算 scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
      2. 如果有 mask，把 mask=True 的位置填 -inf
      3. softmax(scores, dim=-1) 得到 attn_weights
      4. output = attn_weights @ V
    """
    d_k = Q.shape[-1]

    # 步骤 1：计算原始分数
    scores = ___  # shape: (B, L, L)

    # 步骤 2：应用 mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 步骤 3：softmax 归一化
    attn_weights = ___  # shape: (B, L, L)，每行和为 1

    # 步骤 4：加权求和 V
    output = ___  # shape: (B, L, d_v)

    return output, attn_weights


# ── 验证 ──────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d_k = 2, 10, 64

    Q = torch.randn(B, L, d_k)
    K = torch.randn(B, L, d_k)
    V = torch.randn(B, L, d_k)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print(f"output shape     : {output.shape}")       # 期望: (2, 10, 64)
    print(f"attn_weights shape: {attn_weights.shape}") # 期望: (2, 10, 64)

    # 验证：每行和为 1
    row_sums = attn_weights.sum(dim=-1)
    print(f"attn_weights 行和（应全为 1.0）：")
    print(row_sums)

    # 对比 Week 11 的真实注意力权重 shape
    print(f"\nWeek 11 ESM-2 的注意力 shape：(1, num_heads, L, L)")
    print(f"本次实现的 attn_weights shape：{attn_weights.shape}")
    print("差异：本次是单头，ESM-2 有多个头（多头将在 Day 2 实现）")
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| `output.shape` | `(2, 10, 64)` |
| `attn_weights.shape` | `(2, 10, 10)` |
| `attn_weights.sum(dim=-1)` | 全为 `1.0` |
| 能填写下方三个问题 | — |

---

## 输出问题

**Q1**：Q、K、V 三个矩阵都来自同一个输入 X，为什么要分成三个独立的投影矩阵？如果直接用 $$XX^T$$ 做注意力分数会有什么问题？

**Q2**：如果 $$d_k = 512$$，不除以 $$\sqrt{d_k}$$ 的情况下，点积的数量级大约是多少？softmax 会发生什么？

**Q3**：`attn_weights[b, i, j]` 这个值的含义是什么？如果 `attn_weights[0, 3, 7] = 0.8`，用一句话解释它的生物学直觉（在蛋白质序列的语境下）。

---

完成后把代码输出和三个问题的回答发给我。