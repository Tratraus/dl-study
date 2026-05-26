import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask = None):
    """
    参数：
        Q : (batch, seq_len, d_k)
        K : (batch, seq_len, d_k)
        V : (batch, seq_len, d_v)
        mask : (batch, seq_len) bool tensor，True 表示该位置是 padding
    返回：
        output  : (batch, seq_len, d_v)
        weights : (batch, seq_len, seq_len)
    """
    # 第一步：计算 scores = Q @ K^T / sqrt(d_k)
    # 提示：K 需要转置最后两个维度，用 K.transpose(-2, -1)
    # 提示：d_k = Q.size(-1)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2,-1) / (d_k **0.5)


    # 第二步：如果有 mask，把 padding 位置的 scores 设为 -1e9
    # 提示：mask 形状是 (batch, seq_len)，需要扩展到 (batch, 1, seq_len)
    #       用 mask.unsqueeze(1)
    #       然后 scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
    if mask is not None :
      scores = scores.masked_fill(mask.unsqueeze(1), -1e9)

    # 第三步：softmax（对最后一个维度）
    weights = torch.softmax(scores, dim = -1)

    # 第四步：weights @ V
    output = weights @ V

    return output, weights

def smoke_test():
    batch, seq_len, d_k, d_v = 2, 6, 16, 16

    Q = torch.randn(batch, seq_len, d_k)
    K = torch.randn(batch, seq_len, d_k)
    V = torch.randn(batch, seq_len, d_v)

    # 模拟 padding mask：每个序列最后 2 个位置是 padding
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, -2:] = True   # 最后两个位置是 padding

    output, weights = scaled_dot_product_attention(Q, K, V, mask)

    print(f"output  shape : {output.shape}")    # (2, 6, 16)
    print(f"weights shape : {weights.shape}")   # (2, 6, 6)

    # 验证权重是概率分布
    print(f"weights sum   : {weights[0].sum(dim=-1).tolist()}")  # 每行加起来 ≈ 1.0

    # 验证 padding 位置的权重为 0
    print(f"padding weight: {weights[0, 0, -2:].tolist()}")      # 应该接近 [0, 0]

    print("✅ 通过" if output.shape == (2, 6, 16) else "❌ 形状错误")

smoke_test()
