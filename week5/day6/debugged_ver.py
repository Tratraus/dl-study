import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ════════════════════════════════════════════════════════════
# Core Components
# ════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q : (..., seq_len, d_k)
    K : (..., seq_len, d_k)
    V : (..., seq_len, d_v)
    mask : (..., seq_len)  Bool，True = padding
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    weights = torch.softmax(scores, dim=-1)
    output  = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # FIX 2: 删除未使用的 self.dropout

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch,1,1,seq_len)

        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output   = self.W_o(attn_out)
        # FIX 2: 不在 MHA 内部做 dropout，交给 Block 统一管理
        return output, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN Attention + 残差
        attn_out, weights = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)   # FIX 3: 对称地加 dropout

        # Pre-LN FFN + 残差
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x, weights


# ════════════════════════════════════════════════════════════
# Tokenizer & Vocab
# ════════════════════════════════════════════════════════════

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0

SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {0: "H", 1: "E", 2: "C"}


def tokenize(sequences, max_len=None):
    """
    sequences : List[str]
    返回：
        tokens : (batch, max_len)  LongTensor
        mask   : (batch, max_len)  BoolTensor，True = padding
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    batch  = len(sequences)
    tokens = torch.zeros(batch, max_len, dtype=torch.long)
    mask   = torch.ones(batch, max_len,  dtype=torch.bool)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            tokens[i, j] = AA_TO_IDX.get(aa, 0)
        mask[i, :len(seq)] = False

    return tokens, mask


# ════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════

class ProteinTransformer(nn.Module):
    def __init__(
        self,
        d_model     = 64,
        num_heads   = 4,
        d_ff        = 256,
        num_layers  = 3,
        dropout     = 0.1,
        num_classes = 3,
        max_len     = 512,
        vocab_size  = 21,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)
        # FIX 1: 删除 self.dropout，不在 Embedding 后额外做 dropout

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)   # (batch, seq_len, d_model)
        x = self.pos_enc(x)          # FIX 1: 只保留 pos_enc 内部的 dropout

        for block in self.blocks:
            x, _ = block(x, mask)

        logits = self.classifier(x)  # (batch, seq_len, num_classes)
        return logits


# ════════════════════════════════════════════════════════════
# Smoke Test
# ════════════════════════════════════════════════════════════

def smoke_test():
    sequences = [
        "MKLVFGRELEK",
        "ACDEFGHIKL",
        "MNPQRSTVWY",
    ]
    tokens, mask = tokenize(sequences)
    print(f"tokens shape : {tokens.shape}")
    print(f"mask   shape : {mask.shape}")
    print(f"tokens[0]    : {tokens[0].tolist()}")
    print(f"mask[0]      : {mask[0].tolist()}")

    model = ProteinTransformer(
        d_model=64, num_heads=4, d_ff=256,
        num_layers=3, dropout=0.0
    )

    logits = model(tokens, mask)
    print(f"logits shape : {logits.shape}")

    preds  = logits.argmax(dim=-1)
    pred_ss = [[IDX_TO_SS[p.item()] for p in seq] for seq in preds]
    print(f"pred[0] : {''.join(pred_ss[0])}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量     : {total:,}")
    print(f"可训练参数量 : {trainable:,}")

    assert logits.shape == (3, 11, 3)
    print("✅ 通过")

smoke_test()