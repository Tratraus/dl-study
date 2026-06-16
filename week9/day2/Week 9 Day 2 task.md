# Week 9 Day 2：Cross-Attention 原理与实现

---

## 任务目标

实现一个 `CrossAttention` 模块，理解 Q/K/V 的来源和 Self-Attention 的本质区别，并用 shape 验证输出正确。

---

## 最小必要理论

### 1. Self-Attention vs Cross-Attention

|  | Q 来自 | K 来自 | V 来自 | 输出长度 |
|--|--------|--------|--------|---------|
| Self-Attention | 自己 | 自己 | 自己 | = 输入长度 |
| Cross-Attention | Decoder 当前状态 | Encoder 输出（memory） | Encoder 输出（memory） | = Q 的长度 |

Self-Attention 是"自己问自己"，Cross-Attention 是"Decoder 拿着问题去问 Encoder"。

---

### 2. Cross-Attention 的计算过程

```text
query   shape: (batch, tgt_len, d_model)   ← 来自 Decoder
key     shape: (batch, src_len, d_model)   ← 来自 Encoder memory
value   shape: (batch, src_len, d_model)   ← 来自 Encoder memory

第一步：Q 和 K 做点积
  scores = Q @ K^T
  shape:  (batch, tgt_len, src_len)

第二步：缩放 + softmax
  weights = softmax(scores / sqrt(d_k))
  shape:   (batch, tgt_len, src_len)

第三步：加权求和 V
  output = weights @ V
  shape:  (batch, tgt_len, d_model)   ← 输出长度等于 tgt_len，不是 src_len
```

关键结论：**输出的序列长度由 Q 决定，和 K/V 的长度无关。**

---

### 3. 投影矩阵

实际实现中，Q/K/V 在做点积之前都要先过一个线性投影：

$$Q' = W_Q \cdot \text{query}, \quad K' = W_K \cdot \text{key}, \quad V' = W_V \cdot \text{value}$$

三个投影矩阵的形状都是 `(d_model, d_model)`（单头情况下）。

---

## 代码任务

新建文件：`week9/day2/cross_attention.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    单头 Cross-Attention。

    query 来自 Decoder，key/value 来自 Encoder memory。

    输入：
      query:   (batch, tgt_len, d_model)
      context: (batch, src_len, d_model)   ← 这就是 memory
    输出：
      output:  (batch, tgt_len, d_model)
    """
    def __init__(self, d_model):
        super().__init__()
        # TODO 1：定义 Q / K / V 三个投影层
        # 每个都是 Linear(d_model, d_model)，无 bias 也可以
        ...

        # TODO 2：定义输出投影层
        # Linear(d_model, d_model)
        ...

        self.scale = math.sqrt(d_model)

    def forward(self, query, context):
        # TODO 3：计算 Q / K / V
        # Q shape: (batch, tgt_len, d_model)
        # K shape: (batch, src_len, d_model)
        # V shape: (batch, src_len, d_model)
        ...

        # TODO 4：计算注意力分数
        # scores shape: (batch, tgt_len, src_len)
        ...

        # TODO 5：softmax + 加权求和
        # weights shape: (batch, tgt_len, src_len)
        # output  shape: (batch, tgt_len, d_model)
        ...

        # TODO 6：过输出投影
        # output shape: (batch, tgt_len, d_model)
        ...

        return output


# ── 验证 ────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, tgt_len, src_len, d_model = 2, 6, 10, 64

    query   = torch.randn(batch, tgt_len, d_model)
    context = torch.randn(batch, src_len, d_model)

    attn = CrossAttention(d_model)
    out  = attn(query, context)

    # TODO 7：打印 out 的形状，验证是否为 (2, 6, 64)
    print(f"output shape: {out.shape}")
    assert out.shape == (batch, tgt_len, d_model), "shape 错误！"
    print("验证通过 ✅")
```

---

## 完成标准

1. 六个 TODO 全部填完，无报错
2. 打印出：
```text
output shape: torch.Size([2, 6, 64])
验证通过 ✅
```
3. 能回答下面三个问题

---

## 输出问题

**Q1**：`scores = Q @ K^T` 这一步，`K^T` 在 PyTorch 里怎么写？（提示：`@` 是矩阵乘法，K 的形状是 `(batch, src_len, d_model)`，你需要转置哪两个维度？）

**Q2**：为什么要除以 `sqrt(d_model)`（缩放）？如果不缩放会发生什么？

**Q3**：Cross-Attention 的输出形状是 `(batch, tgt_len, d_model)`，和 `src_len` 无关。用一句话解释为什么。

---

准备好后提交代码和三个问题的回答。