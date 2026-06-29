# Week 12 · Day 4：完整 Transformer Encoder Block

## 理论：还缺哪两个零件

前三天的零件：
```
Day 1: scaled_dot_product_attention()   ✅
Day 2: MultiHeadAttention               ✅
Day 3: SinusoidalPositionalEncoding     ✅
```

今天补齐最后两个，然后组装：

---

### 零件 4：Layer Normalization

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

其中 $$\mu, \sigma$$ 是**同一个样本、同一个位置**在 $$d_{model}$$ 维上的均值和标准差。

**和 BatchNorm 的区别：**

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| 归一化方向 | 跨样本（batch 维） | 跨特征（d_model 维） |
| 序列长度变化时 | ⚠️ 统计量不稳定 | ✅ 每个位置独立计算 |
| 小 batch 时 | ⚠️ 统计量噪声大 | ✅ 不受 batch size 影响 |

Transformer 用 LayerNorm 而不是 BatchNorm，正是因为序列长度可变、batch size 通常较小。

---

### 零件 5：Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- 两层全连接，中间维度 $$d_{ff}$$ 通常是 $$4 \times d_{model}$$
- **逐位置独立**作用（position-wise）：每个位置用同一套参数，互不影响
- 直觉：Attention 负责"收集信息"，FFN 负责"处理信息"

---

### 组装：Pre-LN vs Post-LN

原始论文（Attention is All You Need）用 **Post-LN**：

$$x = \text{LayerNorm}(x + \text{SubLayer}(x))$$

现代实现（包括 ESM-2）更多用 **Pre-LN**（训练更稳定）：

$$x = x + \text{SubLayer}(\text{LayerNorm}(x))$$

今天实现 **Pre-LN** 版本：

```
输入 x
  ├─ LayerNorm → MultiHeadAttention → + 残差 → x'
  └─────────────────────────────────────────────┘
  ├─ LayerNorm → FFN → + 残差 → x''
  └──────────────────────────────────┘
输出 x''
```

---

## 代码任务

新建 `week12/day4/transformer_encoder.py`，**复用前三天的模块**：

```python
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day3'))
from multihead_attention import MultiHeadAttention
from positional_encoding import SinusoidalPositionalEncoding


# ── 零件 5：Feed-Forward Network ──────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            ___,           # dropout
            nn.Linear(d_ff, d_model),
            ___,           # dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 组装：单个 Encoder Block ──────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attn    = MultiHeadAttention(d_model, num_heads)
        self.ffn     = FeedForward(d_model, d_ff, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-LN 结构：
          x = x + Dropout(Attn(LayerNorm(x)))
          x = x + Dropout(FFN(LayerNorm(x)))
        """
        # 子层 1：Multi-Head Attention（Pre-LN）
        residual = x
        x_norm = ___                          # LayerNorm
        attn_out, attn_weights = ___          # MultiHeadAttention，传入 mask
        x = residual + self.dropout(___)      # 残差连接

        # 子层 2：FFN（Pre-LN）
        residual = x
        x_norm = ___                          # LayerNorm
        ffn_out = ___                         # FFN
        x = residual + self.dropout(___)      # 残差连接

        return x, attn_weights


# ── 堆叠多层：TransformerEncoder ─────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # 最终 LayerNorm（Pre-LN 惯例）

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, L) 整数 token id
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, list]:
        """
        返回：
          x            : (B, L, d_model)  最终隐状态
          all_attn     : list of (B, num_heads, L, L)，每层的注意力权重
        """
        x = self.embedding(input_ids)       # (B, L, d_model)
        x = ___                             # 加位置编码

        all_attn = []
        for layer in ___:                   # 遍历每一层
            x, attn_weights = ___           # 过 EncoderBlock
            all_attn.append(___)            # 收集注意力权重

        x = self.norm(x)                    # 最终 LayerNorm
        return x, all_attn


# ── 验证 ──────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # 蛋白质场景：20 种氨基酸 + 3 个特殊 token
    VOCAB_SIZE = 23
    B, L       = 2, 50
    d_model    = 64
    num_heads  = 4
    num_layers = 3
    d_ff       = 256   # 4 × d_model

    model = TransformerEncoder(
        vocab_size  = VOCAB_SIZE,
        d_model     = d_model,
        num_heads   = num_heads,
        num_layers  = num_layers,
        d_ff        = d_ff,
        max_len     = 512,
        dropout     = 0.0
    )

    input_ids = torch.randint(0, VOCAB_SIZE, (B, L))
    output, all_attn = model(input_ids)

    print(f"输入  shape : {input_ids.shape}")
    print(f"输出  shape : {output.shape}")          # 期望: (2, 50, 64)
    print(f"注意力层数  : {len(all_attn)}")          # 期望: 3
    print(f"每层注意力  : {all_attn[0].shape}")      # 期望: (2, 4, 50, 50)

    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total:,}")

    # 各模块参数量分解
    emb   = sum(p.numel() for p in model.embedding.parameters())
    layer = sum(p.numel() for p in model.layers[0].parameters())
    print(f"  Embedding      : {emb:,}")
    print(f"  单层 Block     : {layer:,}  × {num_layers} 层")
    print(f"  Final LayerNorm: {sum(p.numel() for p in model.norm.parameters()):,}")
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| `output.shape` | `(2, 50, 64)` |
| `len(all_attn)` | `3` |
| `all_attn[0].shape` | `(2, 4, 50, 50)` |
| 总参数量合理 | 约 `~130K` 量级 |

---

## 输出问题

**Q1**：`nn.ModuleList` 和普通 Python `list` 存 layer 有什么区别？如果用普通 list，会出什么问题？

**Q2**：FFN 里为什么中间维度 $$d_{ff} = 4 \times d_{model}$$？这个 4 是怎么来的？

**Q3**：Pre-LN 和 Post-LN 在训练稳定性上的差异，从梯度流动的角度解释一下。