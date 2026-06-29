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
            nn.Dropout(dropout),           # dropout
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),           # dropout
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
        x_norm = self.norm1(x)                          # LayerNorm
        attn_out, attn_weights = self.attn(x_norm, mask)          # MultiHeadAttention，传入 mask
        x = residual + self.dropout(attn_out)      # 残差连接

        # 子层 2：FFN（Pre-LN）
        residual = x
        x_norm = self.norm2(x)                          # LayerNorm
        ffn_out = self.ffn(x_norm)                         # FFN
        x = residual + self.dropout(ffn_out)      # 残差连接

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
        x = self.pos_enc(x)                 # 加位置编码

        all_attn = []
        for layer in self.layers:           # 遍历每一层
            x, attn_weights = layer(x, mask)           # 过 EncoderBlock
            all_attn.append(attn_weights)            # 收集注意力权重

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

# 输出

# 输入  shape : torch.Size([2, 50])
# 输出  shape : torch.Size([2, 50, 64])
# 注意力层数  : 3
# 每层注意力  : torch.Size([2, 4, 50, 50])

# 总参数量: 151,552
#   Embedding      : 1,472
#   单层 Block     : 49,984  × 3 层
#   Final LayerNorm: 128

# Q1：`nn.ModuleList` 和普通 Python `list` 存 layer 有什么区别？如果用普通 list，会出什么问题？
# nn.ModuleList 会将其中的子模块注册到父模块中，这样在调用 model.parameters() 时可以正确返回所有参数。
# 如果是普通的list，PyTorch 不会将其中的子模块注册到父模块中，
# 导致 model.parameters() 无法返回这些子模块的参数，从而无法进行训练和优化。

# Q2：FFN 里为什么中间维度 $d_{ff} = 4 \times d_{model}$？这个 4 是怎么来的？
# 经验值，Transformer 原论文中使用 4 倍的扩展因子，可以增加模型的表达能力，同时不会显著增加计算量。

# Q3：Pre-LN 和 Post-LN 在训练稳定性上的差异，从梯度流动的角度解释一下。
# Pre-LN（LayerNorm 在子层前）有助于梯度在深层网络中更稳定地流动，
# 因为它在每个子层的输入上进行归一化，减少了梯度消失或爆炸的风险。
# Post-LN（LayerNorm 在子层后）可能导致梯度在深层网络中衰减，
# 因为归一化是在残差连接之后进行的，可能会使梯度在传递过程中变得不稳定。
# 因此，Pre-LN 通常在训练深层 Transformer 时表现更好。