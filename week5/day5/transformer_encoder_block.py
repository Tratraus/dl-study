import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        output = self.W_o(attn_out)
        output = self.dropout(output)

        return output, weights


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 前馈网络：d_model → d_ff → d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # 两个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len), True 表示 padding
        """

        # Pre-LN Attention:
        # x = x + Attention(LayerNorm(x))
        attn_input = self.norm1(x)
        attn_out, weights = self.attn(attn_input, mask)
        x = x + self.dropout(attn_out)

        # Pre-LN FFN:
        # x = x + FFN(LayerNorm(x))
        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)

        return x, weights


def smoke_test():
    batch, seq_len = 2, 8
    d_model, num_heads, d_ff = 64, 4, 256

    block = TransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )

    x = torch.randn(batch, seq_len, d_model)

    # 最后两个位置是 padding
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, -2:] = True

    out, weights = block(x, mask)

    print(f"input   shape : {x.shape}")        # (2, 8, 64)
    print(f"output  shape : {out.shape}")      # (2, 8, 64)
    print(f"weights shape : {weights.shape}")  # (2, 4, 8, 8)

    print(f"padding weight: {weights[0, 0, 0, -2:].tolist()}")

    assert out.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, num_heads, seq_len, seq_len)
    print("✅ 通过")

smoke_test()


# 小问题：
# 我写的
# x = x + self.dropout(attn_out)   # ← 对 attn_out 又加了一次 dropout

# 骨架：
# x = x + attn_out                 # ← MultiHeadAttention 内部已经做了 dropout
# 两种写法都是合理的，但你的写法相当于对 attention 输出做了两次 dropout：


# 第一次：MultiHeadAttention.forward() 里的 self.dropout(output)
# 第二次：TransformerEncoderBlock.forward() 里的 self.dropout(attn_out)
# 这在实践中 dropout 率会偏高。标准做法是只在一个地方做。