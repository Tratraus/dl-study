import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
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
        x = x.view(B, L, self.num_heads, self.d_k)   # 填入正确的 shape
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
        K = self.W_k(x)
        V = self.W_v(x)

        # 步骤 2：拆分多头
        Q = self.split_heads(Q)  # (B, num_heads, L, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 步骤 3：每个头做 scaled dot-product attention
        # 注意：scaled_dot_product_attention 接受 (B, L, d_k)
        # 现在 Q/K/V 是 (B, num_heads, L, d_k)
        # → 把 B 和 num_heads 合并：reshape 成 (B*num_heads, L, d_k)
        Q = Q.reshape(B * self.num_heads, L, self.d_k)
        K = K.reshape(B * self.num_heads, L, self.d_k)
        V = V.reshape(B * self.num_heads, L, self.d_k)

        attn_mask = None
        if mask is not None:
            # mask 可能是 (B, 1, 1, L)，先 expand 再 reshape
            attn_mask = mask.expand(B, self.num_heads, 1, L)  # (B, H, 1, L)
            attn_mask = attn_mask.reshape(B * self.num_heads, 1, L)  # (B*H, 1, L)

        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, attn_mask)
        # attn_out    : (B*num_heads, L, d_k)
        # attn_weights: (B*num_heads, L, L)

        # 步骤 4：把多头结果拼回来
        # (B*num_heads, L, d_k) → (B, num_heads, L, d_k) → (B, L, d_model)
        attn_out = attn_out.reshape(B, self.num_heads, L, self.d_k)
        attn_out = attn_out.transpose(1, 2)          # (B, L, num_heads, d_k)
        attn_out = attn_out.reshape(B, L, self.d_model)  # (B, L, d_model)

        # 步骤 5：最终线性投影 W_O
        output = self.W_o(attn_out)  # (B, L, d_model)

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


# 输出

# 输入  shape: torch.Size([2, 10, 64])
# 输出  shape: torch.Size([2, 10, 64])
# 注意力 shape: torch.Size([2, 4, 10, 10])

# 各头注意力行和（应全为 1.0）：
# tensor([[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000]],

#         [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000],
#          [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#           1.0000, 1.0000]]], grad_fn=<SumBackward1>)

# 模型参数量: 16640（含 bias）
# 理论参数量: 16640


# Q1：split_heads 里为什么要先 view 再 transpose？能不能直接 view(B, num_heads, L, d_k)？
# 因为直接 view(B, num_heads, L, d_k) 会改变原始数据在内存中的顺序，导致每个头的维度不连续，从而破坏了注意力机制的计算逻辑。
# 先 view(B, L, num_heads, d_k) 保持了原始顺序，然后 transpose(1, 2) 将 num_heads 移到第二维，确保每个头的数据是连续的。

# Q2：多头注意力的参数量和单头相同，但效果更好。从"子空间"的角度解释：多头到底多了什么？
# 多头注意力通过将输入的特征空间划分为多个子空间（每个头对应一个子空间），使得模型能够在不同的表示子空间中学习不同的注意力模式。
# 每个头可以关注输入序列的不同部分，从而捕捉到更多样化的特征和关系。这种多样性使得模型在处理复杂任务时表现更好，因为它能够综合来自不同子空间的信息。

# Q3：attn_weights 的 shape 是 (B, num_heads, L, L)，和 Week 11 ESM-2 的 outputs.attentions 的 shape 对比一下——它们一样吗？
# 同样是(B, num_heads, L, L)