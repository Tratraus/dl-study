import torch
import math
import torch.nn as nn

# ── 把 train_v4.py 里以下内容复制过来 ──────────────────────
# AMINO_ACIDS, SS_LABELS, HELIX_AA, STRAND_AA

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SS_LABELS   = "HEC"

HELIX_AA  = set("AVILM")
STRAND_AA = set("FYWST")

# AA_TO_IDX, SS_TO_IDX, IDX_TO_SS

AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0
SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {0: "H", 1: "E", 2: "C"}

# scaled_dot_product_attention

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

# MultiHeadAttention

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

# PositionalEncoding

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

# TransformerEncoderBlock

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),           # ← 改这里，ReLU → GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, w = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, w

# ProteinTransformer

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

# ────────────────────────────────────────────────────────────


def predict(seq_str, model, device, max_len=128):
    """
    对单条氨基酸序列进行二级结构预测

    参数：
        seq_str : str，氨基酸序列，例如 "MKLVFAVILM"
        model   : 已加载权重的 ProteinTransformer
        device  : torch.device
        max_len : int，截断长度

    返回：
        pred_ss : str，预测的二级结构，例如 "CCHHHHHHHH"
    """
    model.eval()

    # 截断
    seq_str = seq_str[:max_len]

    # TODO 1: 把序列转成 token tensor，shape=(1, seq_len)
    # 提示：
    #   tokens = [AA_TO_IDX.get(aa, 0) for aa in seq_str]
    #   然后转成 torch.LongTensor，unsqueeze(0) 加 batch 维度

    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq_str]
    tokens = torch.LongTensor(tokens).unsqueeze(0)


    # TODO 2: 构造 mask，shape=(1, seq_len)，全为 False（没有 padding）
    # 提示：torch.zeros(1, len(seq_str), dtype=torch.bool)

    mask = torch.zeros(1, len(seq_str), dtype=torch.bool)

    # TODO 3: 移动到 device

    tokens = tokens.to(device)
    mask   = mask.to(device)

    # TODO 4: 前向传播，得到 logits，shape=(1, seq_len, 3)
    # 提示：with torch.no_grad(): logits = model(tokens, mask)

    with torch.no_grad():
        logits = model(tokens, mask)

    # TODO 5: 取 argmax，得到每个位置的预测类别，shape=(1, seq_len)
    # 提示：preds = logits.argmax(dim=-1)
        preds = logits.argmax(dim=-1)

    # TODO 6: 把预测的 index 转回字符串
    # 提示：IDX_TO_SS[idx.item()] for idx in preds[0]
        pred_ss = ''.join([IDX_TO_SS[idx.item()] for idx in preds[0]])
    return pred_ss


def load_model(checkpoint_path, device):
    """
    从 checkpoint 文件加载模型

    参数：
        checkpoint_path : str，best_model.pt 的路径
        device          : torch.device

    返回：
        model : 已加载权重的 ProteinTransformer
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = ProteinTransformer(
        d_model=64, num_heads=4, d_ff=256, num_layers=3, dropout=0.1
    ).to(device)

    # TODO: 加载模型权重
    # 提示：model.load_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    ckpt  = torch.load("week6/day5/best_model.pt", map_location=device)
    model = load_model("week6/day5/best_model.pt", device)
    print(f"模型加载成功，来自 best_epoch={ckpt['epoch']}")   # TODO: 打印 ckpt["epoch"]

    # 测试序列
    test_seqs = [
        "AVILMAVILMAVILM",          # 全是疏水氨基酸，应该全预测为 H
        "FYWSTFYWST",               # 全是芳香/极性，应该全预测为 E
        "ACDEFGHIKLMNPQRSTVWY",     # 混合序列
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    ]

    print("\n── 预测结果 ──")
    for seq in test_seqs:
        pred = predict(seq, model, device)
        # 对齐打印
        print(f"序列({len(seq):3d}): {seq[:40]}{'...' if len(seq)>40 else ''}")
        print(f"预测({len(pred):3d}): {pred[:40]}{'...' if len(pred)>40 else ''}")
        print()

main()
