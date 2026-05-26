import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接复制昨天的函数
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 要求：d_model 必须能被 num_heads 整除
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        # 四个线性层：W^Q, W^K, W^V, W^O
        # 提示：都是 (d_model → d_model)，不需要 bias 也可以
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # 第一步：线性投影
        Q = self.W_q(x)   # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 第二步：拆分成 h 个头
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        # 提示：用 .view() 拆分，再用 .transpose(1, 2) 移动维度
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 第三步：调用 scaled_dot_product_attention
        # mask 需要从 (batch, seq_len) 扩展到 (batch, 1, 1, seq_len)
        # 提示：mask.unsqueeze(1).unsqueeze(2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (batch, h, seq_len, d_k)

        # 第四步：拼接所有头
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k) → (batch, seq_len, d_model)
        # 提示：先 .transpose(1, 2)，再 .contiguous().view()
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # 第五步：输出线性层
        output = self.W_o(attn_out)

        return output, weights


def smoke_test():
    batch, seq_len, d_model, num_heads = 2, 6, 64, 4

    model = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(batch, seq_len, d_model)

    # 最后 2 个位置是 padding
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, -2:] = True

    output, weights = model(x, mask)

    print(f"output  shape : {output.shape}")    # (2, 6, 64)
    print(f"weights shape : {weights.shape}")   # (2, 4, 6, 6)

    # 验证 padding 位置权重为 0
    print(f"padding weight: {weights[0, 0, 0, -2:].tolist()}")  # 应接近 [0, 0]

    assert output.shape  == (2, 6, 64),    "output 形状错误"
    assert weights.shape == (2, 4, 6, 6),  "weights 形状错误"
    print("✅ 通过")

smoke_test()
