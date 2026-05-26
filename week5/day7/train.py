# ── 把 day6 所有代码粘贴到这里 ──────────────────────────────
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

# ────────────────────────────────────────────────────────────

import random
import numpy as np
import matplotlib.pyplot as plt

# ── 固定随机种子，保证结果可复现 ────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ════════════════════════════════════════════════════════════
# 1. 合成数据集
# ════════════════════════════════════════════════════════════

HELIX_AA  = set("AVILM")    # 疏水性，倾向 Helix
STRAND_AA = set("FYWST")    # 芳香/极性，倾向 Strand
# 其余氨基酸 → Coil

def generate_sequence(min_len=20, max_len=50):
    """
    生成一条合成氨基酸序列和对应的二级结构标签
    返回：(seq_str, label_str)
    例如：("MKLVF...", "HHHCC...")
    """
    length = random.randint(min_len, max_len)
    seq, labels = [], []

    i = 0
    while i < length:
        r = random.random()

        if r < 0.35:
            # 生成一段 Helix（3~8 个疏水氨基酸）
            seg_len = random.randint(3, 8)
            for _ in range(min(seg_len, length - i)):
                aa = random.choice(list(HELIX_AA))
                seq.append(aa)
                labels.append("H")
            i += seg_len

        elif r < 0.65:
            # 生成一段 Strand（2~5 个芳香/极性氨基酸）
            seg_len = random.randint(2, 5)
            for _ in range(min(seg_len, length - i)):
                aa = random.choice(list(STRAND_AA))
                seq.append(aa)
                labels.append("E")
            i += seg_len

        else:
            # 生成一个 Coil（随机氨基酸）
            aa = random.choice(list(AMINO_ACIDS))
            seq.append(aa)
            labels.append("C")
            i += 1

    return "".join(seq), "".join(labels)


def make_dataset(n_samples=1000, min_len=20, max_len=50):
    """
    生成 n_samples 条数据
    返回：List of (seq_str, label_str)
    """
    return [generate_sequence(min_len, max_len) for _ in range(n_samples)]


def collate_batch(batch):
    """
    把一个 batch 的 (seq, label) 列表转成 tensor

    参数：
        batch : List of (seq_str, label_str)

    返回：
        tokens : (batch, max_len)  LongTensor
        labels : (batch, max_len)  LongTensor，padding 位置 = -1
        mask   : (batch, max_len)  BoolTensor
    """
    seqs   = [item[0] for item in batch]
    labels_str = [item[1] for item in batch]

    max_len = max(len(s) for s in seqs)
    batch_size = len(seqs)

    tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -1, dtype=torch.long)  # 默认 -1
    mask   = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (seq, lab) in enumerate(zip(seqs, labels_str)):
        for j, aa in enumerate(seq):
            tokens[i, j] = AA_TO_IDX.get(aa, 0)
            labels[i, j] = SS_TO_IDX[lab[j]]
        mask[i, :len(seq)] = False

    return tokens, labels, mask


# ════════════════════════════════════════════════════════════
# 2. 训练循环
# ════════════════════════════════════════════════════════════

def train_one_epoch(model, data, optimizer, loss_fn, batch_size=32):
    """
    跑一个 epoch，返回平均 loss
    """
    model.train()
    random.shuffle(data)
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        tokens, labels, mask = collate_batch(batch)

        optimizer.zero_grad()

        # 前向传播
        logits = model(tokens, mask)   # (batch, seq_len, 3)

        # 计算 loss
        # logits 需要 reshape 成 (batch*seq_len, 3)
        # labels 需要 reshape 成 (batch*seq_len,)
        loss = loss_fn(
            logits.view(-1, 3),
            labels.view(-1)
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, loss_fn, batch_size=32):
    """
    在验证集上计算 loss 和准确率（忽略 padding 位置）
    返回：(avg_loss, accuracy)
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    n_batches     = 0

    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        tokens, labels, mask = collate_batch(batch)

        logits = model(tokens, mask)   # (batch, seq_len, 3)

        loss = loss_fn(
            logits.view(-1, 3),
            labels.view(-1)
        )
        total_loss += loss.item()
        n_batches  += 1

        # 计算准确率（只统计非 padding 位置）
        preds = logits.argmax(dim=-1)          # (batch, seq_len)
        valid = (labels != -1)                 # (batch, seq_len)，True = 真实位置
        total_correct += (preds[valid] == labels[valid]).sum().item()
        total_tokens  += valid.sum().item()

    avg_loss = total_loss / n_batches
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


# ════════════════════════════════════════════════════════════
# 3. 主训练流程
# ════════════════════════════════════════════════════════════

def main():
    set_seed(42)

    # ── 数据 ────────────────────────────────────────────────
    all_data  = make_dataset(n_samples=1000)
    train_data = all_data[:800]
    val_data   = all_data[800:]
    print(f"训练集：{len(train_data)} 条，验证集：{len(val_data)} 条")
    print(f"示例序列：{train_data[0][0]}")
    print(f"示例标签：{train_data[0][1]}")

    # ── 模型 ────────────────────────────────────────────────
    model = ProteinTransformer(
        d_model    = 64,
        num_heads  = 4,
        d_ff       = 256,
        num_layers = 3,
        dropout    = 0.1,
        num_classes= 3,
        max_len    = 512,
        vocab_size = 21,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}")

    # ── 损失函数 & 优化器 ────────────────────────────────────
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── 训练 ────────────────────────────────────────────────
    n_epochs = 50
    train_losses, val_losses, val_accs = [], [], []

    print("\n开始训练...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
    print("-" * 42)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_data, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_data, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % 5 == 0:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_acc:>8.2%}")

    print("\n训练完成！")

    # ── 可视化 ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accs, color="green", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("图表已保存：training_curve.png")

    # ── 推理示例 ─────────────────────────────────────────────
    print("\n── 推理示例 ──")
    test_seqs = [
        "AVILMAVILMAVILM",   # 全是疏水氨基酸，应该预测为 H
        "FYWSTFYWSTFYWST",   # 全是芳香/极性，应该预测为 E
        "ACDEFGHIKLMNPQR",   # 混合序列
    ]
    tokens, mask = tokenize(test_seqs)
    with torch.no_grad():
        logits = model(tokens, mask)
    preds = logits.argmax(dim=-1)
    for seq, pred in zip(test_seqs, preds):
        pred_str = "".join(IDX_TO_SS[p.item()] for p in pred)
        print(f"  输入：{seq}")
        print(f"  预测：{pred_str}")
        print()


main()
