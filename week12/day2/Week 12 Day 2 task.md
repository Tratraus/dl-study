# Week 12 · Day 2：多头注意力实现

## 理论：为什么要多个头

### 单头的局限

Day 1 实现的单头注意力，每个位置只能用**一种方式**去关注其他位置。

但蛋白质序列中，一个氨基酸可能同时需要：
- 关注**相邻位置**（局部序列模式）
- 关注**远端的活性位点**（长程依赖）
- 关注**同类氨基酸**（化学性质相似的残基）

单头只能学到这三种关系的某种混合，**多头让每个头专注学一种**。

---

### 多头的做法

不是把输入复制多份分别做注意力（那样参数量会爆炸），而是：

**把 $$d_{model}$$ 维的空间切成 $$h$$ 份，每个头只在 $$d_k = d_{model}/h$$ 维的子空间里做注意力。**

$$\text{head}_i = \text{Attention}(XW_i^Q,\ XW_i^K,\ XW_i^V)$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

图示：

```
输入 X: (B, L, d_model=256)
         ↓ 切成 h=4 个头，每头 d_k=64
head_0: (B, L, 64) → Attention → (B, L, 64)
head_1: (B, L, 64) → Attention → (B, L, 64)
head_2: (B, L, 64) → Attention → (B, L, 64)
head_3: (B, L, 64) → Attention → (B, L, 64)
         ↓ Concat
        (B, L, 256)
         ↓ W_O 线性投影
输出:   (B, L, 256)   ← shape 和输入一样
```

---

### 参数量分析

| 矩阵 | shape | 参数量 |
|------|-------|--------|
| $$W_Q$$ | $$(d_{model}, d_{model})$$ | $$d_{model}^2$$ |
| $$W_K$$ | $$(d_{model}, d_{model})$$ | $$d_{model}^2$$ |
| $$W_V$$ | $$(d_{model}, d_{model})$$ | $$d_{model}^2$$ |
| $$W_O$$ | $$(d_{model}, d_{model})$$ | $$d_{model}^2$$ |
| **合计** | | $$4 \cdot d_{model}^2$$ |

> 注意：**多头和单头的参数量完全相同**。多头只是把同样的参数量拆成了多个子空间，用 reshape 实现，不是真的复制多份权重。

---

## 代码任务

新建 `week12/day2/multihead_attention.py`，**复用 Day 1 的函数**：

```python
import torch
import torch.nn as nn
import sys
sys.path.append("../day1")
from self_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 四个线性投影层（不带 bias 也可以，这里保持默认带 bias）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 (B, L, d_model) reshape 成 (B, num_heads, L, d_k)
        步骤：
          1. x.view(B, L, num_heads, d_k)   → (B, L, num_heads, d_k)
          2. .transpose(1, 2)               → (B, num_heads, L, d_k)
        """
        B, L, _ = x.shape
        x = x.view(___, ___, ___, ___)   # 填入正确的 shape
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
          x    : (B, L, d_model)
          mask : (B, 1, 1, L) 或 None  ← 广播到每个头

        返回：
          output      : (B, L, d_model)
          attn_weights: (B, num_heads, L, L)
        """
        B, L, _ = x.shape

        # 步骤 1：线性投影
        Q = self.W_q(x)  # (B, L, d_model)
        K = ___
        V = ___

        # 步骤 2：拆分多头
        Q = self.split_heads(Q)  # (B, num_heads, L, d_k)
        K = ___
        V = ___

        # 步骤 3：每个头做 scaled dot-product attention
        # 注意：scaled_dot_product_attention 接受 (B, L, d_k)
        # 现在 Q/K/V 是 (B, num_heads, L, d_k)
        # → 把 B 和 num_heads 合并：reshape 成 (B*num_heads, L, d_k)
        Q = Q.reshape(B * self.num_heads, L, self.d_k)
        K = ___
        V = ___

        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out    : (B*num_heads, L, d_k)
        # attn_weights: (B*num_heads, L, L)

        # 步骤 4：把多头结果拼回来
        # (B*num_heads, L, d_k) → (B, num_heads, L, d_k) → (B, L, d_model)
        attn_out = attn_out.reshape(B, self.num_heads, L, self.d_k)
        attn_out = attn_out.transpose(1, 2)          # (B, L, num_heads, d_k)
        attn_out = attn_out.reshape(B, L, self.d_model)  # (B, L, d_model)

        # 步骤 5：最终线性投影 W_O
        output = ___

        # reshape attn_weights 方便返回
        attn_weights = attn_weights.reshape(B, self.num_heads, L, L)

        return output, attn_weights


# ── 验证 ──────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d_model, num_heads = 2, 10, 64, 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(B, L, d_model)

    output, attn_weights = mha(x)

    print(f"输入  shape: {x.shape}")
    print(f"输出  shape: {output.shape}")        # 期望: (2, 10, 64)
    print(f"注意力 shape: {attn_weights.shape}") # 期望: (2, 4, 10, 10)

    # 验证每个头的注意力权重行和为 1
    row_sums = attn_weights.sum(dim=-1)
    print(f"\n各头注意力行和（应全为 1.0）：")
    print(row_sums)

    # 验证参数量
    total_params = sum(p.numel() for p in mha.parameters())
    expected = 4 * d_model * d_model + 4 * d_model  # 权重 + bias
    print(f"\n模型参数量: {total_params}（含 bias）")
    print(f"理论参数量: {expected}")
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| `output.shape` | `(2, 10, 64)` |
| `attn_weights.shape` | `(2, 4, 10, 10)` |
| 每个头的行和 | 全为 `1.0` |
| 参数量计算正确 | `4 * 64 * 64 + 4 * 64 = 16640` |

---

## 输出问题

**Q1**：`split_heads` 里为什么要先 `view` 再 `transpose`？能不能直接 `view(B, num_heads, L, d_k)`？

**Q2**：多头注意力的参数量和单头相同，但效果更好。从"子空间"的角度解释：多头到底多了什么？

**Q3**：`attn_weights` 的 shape 是 `(B, num_heads, L, L)`，和 Week 11 ESM-2 的 `outputs.attentions` 的 shape 对比一下——它们一样吗？