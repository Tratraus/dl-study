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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # TODO 2：定义输出投影层
        # Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale = math.sqrt(d_model)

    def forward(self, query, context):
        # TODO 3：计算 Q / K / V
        # Q shape: (batch, tgt_len, d_model)
        # K shape: (batch, src_len, d_model)
        # V shape: (batch, src_len, d_model)
        Q = self.q_proj(query)
        K = self.k_proj(context)
        V = self.v_proj(context)

        # TODO 4：计算注意力分数
        # scores shape: (batch, tgt_len, src_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # TODO 5：softmax + 加权求和
        # weights shape: (batch, tgt_len, src_len)
        # output  shape: (batch, tgt_len, d_model)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # TODO 6：过输出投影
        # output shape: (batch, tgt_len, d_model)

        output = self.out_proj(output)

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


# Q1：scores = Q @ K^T 这一步，K^T 在 PyTorch 里怎么写？（提示：@ 是矩阵乘法，K 的形状是 (batch, src_len, d_model)，你需要转置哪两个维度？）
# 答：K.transpose(-2, -1) 或 K.transpose(1, 2)，都可以。
# 因为 K 的形状是 (batch, src_len, d_model)，
# 我们需要把 src_len 和 d_model 这两个维度交换，才能得到 (batch, d_model, src_len)，
# 这样才能和 Q 的形状 (batch, tgt_len, d_model) 进行矩阵乘法。

# Q2：为什么要除以 sqrt(d_model)（缩放）？如果不缩放会发生什么？
# 答：除以 sqrt(d_model) 是为了防止 scores 的数值过大，导致 softmax 后的梯度消失。

# Q3：Cross-Attention 的输出形状是 (batch, tgt_len, d_model)，和 src_len 无关。用一句话解释为什么。
# 答：因为 Cross-Attention 的输出是对每个 query（tgt_len）进行加权求和得到的，而加权求和的结果维度是 d_model，与 src_len 无关。