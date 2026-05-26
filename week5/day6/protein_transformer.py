import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ── 复制 Day2~5 的组件 ──────────────────────────────────────
# scaled_dot_product_attention()

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
      scores = scores.masked_fill(mask, -1e9)

    # 第三步：softmax（对最后一个维度）
    weights = torch.softmax(scores, dim = -1)

    # 第四步：weights @ V
    output = weights @ V

    return output, weights

# MultiHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):  # ← 添加 dropout 参数
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)  # ← 可选，注意力权重上的 dropout

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

# PositionalEncoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        d_model : 向量维度
        max_len : 支持的最大序列长度
        dropout : 防止过拟合
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算所有位置的编码，形状：(max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position：(max_len, 1)，每个位置的索引
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term：(d_model/2,)，每个时钟的频率
        # 公式：1 / 10000^(2i/d_model) = exp(-2i/d_model * log(10000))
        # 提示：torch.arange(0, d_model, 2) 生成 [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        # 提示：pe[:, 0::2] 取所有行的偶数列
        #       pe[:, 1::2] 取所有行的奇数列
        pe[:, 0::2] = torch.sin(position * div_term)   # sin
        pe[:, 1::2] = torch.cos(position * div_term)   # cos

        # 注册为 buffer（不是参数，不参与梯度更新，但会随模型保存）
        # 增加 batch 维度：(1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
        """
        # 把位置编码加到输入上
        # 注意：只取前 seq_len 个位置
        # 提示：self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# TransformerEncoderBlock

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
        x = x + attn_out

        # Pre-LN FFN:
        # x = x + FFN(LayerNorm(x))
        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)

        return x, weights

# （直接粘贴，不需要修改）
# ────────────────────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0

SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {0: "H", 1: "E", 2: "C"}


def tokenize(sequences, max_len=None):
    """
    把氨基酸字符串列表转成 token tensor 和 mask tensor

    参数：
        sequences : List[str]，比如 ["MKLVF", "ACDE"]
        max_len   : 如果为 None，自动取最长序列的长度

    返回：
        tokens : (batch, max_len)  LongTensor
        mask   : (batch, max_len)  BoolTensor，True 表示 padding
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    batch = len(sequences)
    tokens = torch.zeros(batch, max_len, dtype=torch.long)
    mask   = torch.ones(batch, max_len, dtype=torch.bool)   # 默认全是 padding

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            tokens[i, j] = AA_TO_IDX.get(aa, 0)
        mask[i, :len(seq)] = False   # 真实位置不是 padding

    return tokens, mask


class ProteinTransformer(nn.Module):
    def __init__(
        self,
        d_model    = 64,
        num_heads  = 4,
        d_ff       = 256,
        num_layers = 3,
        dropout    = 0.1,
        num_classes= 3,     # H / E / C
        max_len    = 512,
        vocab_size = 21,
    ):
        super().__init__()

        # 1. Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 3. N 层 TransformerEncoderBlock
        # 提示：用 nn.ModuleList 存放多个 Block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 4. 分类头：d_model → num_classes
        self.classifier = nn.Linear(d_model, num_classes)

        # 5. dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        """
        tokens : (batch, seq_len)  LongTensor
        mask   : (batch, seq_len)  BoolTensor，True 表示 padding
        返回：
            logits : (batch, seq_len, num_classes)
        """
        # 第一步：Embedding
        x = self.embedding(tokens)      # (batch, seq_len, d_model)
        x = self.dropout(x)

        # 第二步：Positional Encoding
        x = self.pos_enc(x)

        # 第三步：逐层过 TransformerEncoderBlock
        for block in self.blocks:
            x, _ = block(x, mask)

        # 第四步：分类头
        logits = self.classifier(x)

        return logits

def smoke_test():
    sequences = [
        "MKLVFGRELEK",
        "ACDEFGHIKL",
        "MNPQRSTVWY",
    ]
    tokens, mask = tokenize(sequences)
    print(f"tokens shape : {tokens.shape}")    # (3, 11)
    print(f"mask   shape : {mask.shape}")      # (3, 11)
    print(f"tokens[0]    : {tokens[0].tolist()}")
    print(f"mask[0]      : {mask[0].tolist()}")

    model = ProteinTransformer(
        d_model=64, num_heads=4, d_ff=256,
        num_layers=3, dropout=0.0
    )

    logits = model(tokens, mask)
    print(f"logits shape : {logits.shape}")    # (3, 11, 3)

    preds = logits.argmax(dim=-1)
    pred_ss = [[IDX_TO_SS[p.item()] for p in seq] for seq in preds]
    print(f"pred[0] : {''.join(pred_ss[0])}")

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量     : {total:,}")
    print(f"可训练参数量 : {trainable:,}")

    assert logits.shape == (3, 11, 3)
    print("✅ 通过")

smoke_test()
