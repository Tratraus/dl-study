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
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # shape: (B, L, L)

    # 步骤 2：应用 mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 步骤 3：softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)  # shape: (B, L, L)，每行和为 1

    # 步骤 4：加权求和 V
    output = attn_weights @ V  # shape: (B, L, d_v)

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


# 输出
# output shape     : torch.Size([2, 10, 64])
# attn_weights shape: torch.Size([2, 10, 10])
# attn_weights 行和（应全为 1.0）：
# tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#          1.0000],
#         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#          1.0000]])

# Week 11 ESM-2 的注意力 shape：(1, num_heads, L, L)
# 本次实现的 attn_weights shape：torch.Size([2, 10, 10])
# 差异：本次是单头，ESM-2 有多个头（多头将在 Day 2 实现）

# Q1：Q、K、V 三个矩阵都来自同一个输入 X，为什么要分成三个独立的投影矩阵？如果直接用 XX^T 做注意力分数会有什么问题？
# 因为 Q、K、V 的作用不同，Q 用于查询，K 用于键，V 用于值。
# 如果直接用 XX^T 做注意力分数，无法区分查询和键的不同角色，可能导致注意力机制无法有效捕捉输入的不同特征。

# Q2：如果 d_k = 512，不除以 \sqrt{d_k} 的情况下，点积的数量级大约是多少？softmax 会发生什么？
# 会导致点积的数量级大约为 512（因为每个元素的值大约在 [-1, 1] 之间，点积会累加 512 个这样的值），
# 这会使得 softmax 的输入值非常大，从而导致 softmax 输出接近 one-hot 分布，
# 即大部分注意力权重集中在一个位置，其他位置几乎为零，这样会丢失信息。

# Q3：`attn_weights[b, i, j]` 这个值的含义是什么？如果 `attn_weights[0, 3, 7] = 0.8`，用一句话解释它的生物学直觉（在蛋白质序列的语境下）。
# `attn_weights[b, i, j]` 表示在第 b 个样本中，第 i 个位置的查询向量 Q[i] 对第 j 个位置的键向量 K[j] 的注意力权重。
# 如果 `attn_weights[0, 3, 7] = 0.8`，这意味着在第一个样本中，位置 3 的氨基酸对位置 7 的氨基酸的注意力权重为 0.8，
# 即位置 3 的氨基酸对位置 7 的氨基酸的影响较大。