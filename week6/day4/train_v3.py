import torch
from torch.utils.data import Dataset, DataLoader
import random
import math
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════
# 1. 生成模拟 FASTA 文件
# ════════════════════════════════════════════════════════════

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SS_LABELS   = "HEC"

HELIX_AA  = set("AVILM")
STRAND_AA = set("FYWST")
def aa_to_ss(aa):
    if aa in HELIX_AA:  return "H"
    if aa in STRAND_AA: return "E"
    return "C"

def generate_ss_with_context(seq):
    """
    规则：
      连续 ≥3 个 AVILM → 整段标为 H（需要看左右邻居）
      连续 ≥2 个 FYWST → 整段标为 E（需要看左右邻居）
      其他             → C

    这个规则必须看上下文才能判断，单个字符无法决定标签。
    """
    n   = len(seq)
    ss  = ["C"] * n

    # 标记 Helix：找连续 ≥3 个疏水氨基酸的片段
    i = 0
    while i < n:
        if seq[i] in HELIX_AA:
            j = i
            while j < n and seq[j] in HELIX_AA:
                j += 1
            if j - i >= 3:          # 连续长度 ≥ 3 才算 Helix
                for k in range(i, j):
                    ss[k] = "H"
            i = j
        else:
            i += 1

    # 标记 Strand：找连续 ≥2 个芳香/极性氨基酸的片段
    i = 0
    while i < n:
        if seq[i] in STRAND_AA and ss[i] == "C":   # 不覆盖已标记的 H
            j = i
            while j < n and seq[j] in STRAND_AA:
                j += 1
            if j - i >= 2:          # 连续长度 ≥ 2 才算 Strand
                for k in range(i, j):
                    ss[k] = "E"
            i = j
        else:
            i += 1

    return "".join(ss)


def generate_fasta_files(seq_path, ss_path, n_samples=500, seed=42):
    """
    生成两个 FASTA 格式的文件：
      seq_path : 氨基酸序列文件
      ss_path  : 二级结构标签文件

    格式：
      >SEQ_0001
      MKLVF...
      >SEQ_0002
      ACDEF...
    """
    random.seed(seed)

    with open(seq_path, "w") as f_seq, open(ss_path, "w") as f_ss:
        for i in range(n_samples):
            length = random.randint(20, 100)

            # 生成序列
            seq = "".join(random.choice(AMINO_ACIDS) for _ in range(length))

            # # 生成标签（随机，模拟真实数据的复杂性）
            # ss  = "".join(random.choice(SS_LABELS)   for _ in range(length))
            # # 改为固定规则
            # ss = "".join(aa_to_ss(aa) for aa in seq)
            # 新的生成策略
            ss = generate_ss_with_context(seq)

            seq_id = f"SEQ_{i+1:04d}"
            f_seq.write(f">{seq_id}\n{seq}\n")
            f_ss.write( f">{seq_id}\n{ss}\n")

    print(f"已生成：{seq_path}（{n_samples} 条序列）")
    print(f"已生成：{ss_path}（{n_samples} 条标签）")


# ════════════════════════════════════════════════════════════
# 2. 解析 FASTA 文件
# ════════════════════════════════════════════════════════════

def parse_fasta(filepath):
    """
    解析 FASTA 文件，返回 dict：{seq_id: sequence_str}

    参数：
        filepath : str，FASTA 文件路径

    返回：
        records : dict，key = seq_id，value = 序列字符串

    提示：
        - 遇到以 ">" 开头的行 → 这是 ID 行，去掉 ">" 和换行符
        - 其他行 → 序列内容，拼接到当前 ID 对应的序列上
        - 注意：序列可能跨多行（本任务生成的是单行，但要写成通用形式）
    """
    records = {}
    current_id  = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # TODO: 保存上一条记录（如果有）
                # TODO: 解析新的 ID
                if current_id is not None:
                    records[current_id] = "".join(current_seq)  # 保存上一条记录
                current_id = line[1:]  # 去掉 ">"
                current_seq = []       # 重置序列列表

            else:
                # TODO: 把这一行拼接到当前序列
                current_seq.append(line)

        # TODO: 保存最后一条记录
        if current_id is not None:
            records[current_id] = "".join(current_seq)

    return records


# ════════════════════════════════════════════════════════════
# 3. Dataset 类
# ════════════════════════════════════════════════════════════

AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0
SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {0: "H", 1: "E", 2: "C"}

class ProteinDataset(Dataset):
    def __init__(self, seq_fasta, ss_fasta):
        """
        参数：
            seq_fasta : 氨基酸序列 FASTA 文件路径
            ss_fasta  : 二级结构标签 FASTA 文件路径
        """
        seq_records = parse_fasta(seq_fasta)
        ss_records  = parse_fasta(ss_fasta)

        # 只保留两个文件都有的 ID
        common_ids = sorted(set(seq_records) & set(ss_records))

        self.samples = []
        for seq_id in common_ids:
            seq = seq_records[seq_id]
            ss  = ss_records[seq_id]
            if len(seq) == len(ss):   # 长度必须一致
                self.samples.append((seq, ss))

        print(f"加载完成：{len(self.samples)} 条有效样本")

    def __len__(self):
        # TODO: 返回样本数量
        return len(self.samples)

    def __getitem__(self, i):
        """
        返回第 i 条样本：(seq_str, ss_str)
        不在这里做 padding，交给 collate_fn 统一处理
        """
        # TODO: 返回 self.samples[i]
        return self.samples[i]


# ════════════════════════════════════════════════════════════
# 4. collate_fn（变长序列 → 对齐的 batch）
# ════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    把一个 batch 的 (seq_str, ss_str) 转成 tensor

    参数：
        batch : List of (seq_str, ss_str)

    返回：
        tokens : (batch, max_len)  LongTensor
        labels : (batch, max_len)  LongTensor，padding 位置 = -1
        mask   : (batch, max_len)  BoolTensor，True = padding

    提示：
        - max_len = 这个 batch 里最长序列的长度
        - tokens 初始化为 0（PAD token）
        - labels 初始化为 -1（ignore_index）
        - mask   初始化为 True（全是 padding）
        - 然后逐条填入真实数据
    """
    # TODO: 实现 collate_fn
    seqs = [item[0] for item in batch]
    ss_labels = [item[1] for item in batch]

    max_len = max(len(seq) for seq in seqs)
    batch_size = len(batch)

    tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)
    mask   = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (seq, ss) in enumerate(zip(seqs, ss_labels)):
        for j, aa in enumerate(seq):
            tokens[i, j] = AA_TO_IDX.get(aa, 0)  # 不认识的 AA 当 PAD 处理
            labels[i, j] = SS_TO_IDX.get(ss[j], -1)  # 不认识的标签当 ignore_index 处理
        mask[i, :len(seq)] = False  # 前 len(seq) 个位置不是 padding

    return tokens, labels, mask

# ════════════════════════════════════════════════════════════
# 新增：支持截断的 ProteinDataset
# ════════════════════════════════════════════════════════════

class ProteinDatasetV2(Dataset):
    def __init__(self, seq_fasta, ss_fasta, max_len=128):
        """
        在 V1 基础上增加：
          max_len : 超过此长度的序列截断到 max_len
        """
        seq_records = parse_fasta(seq_fasta)
        ss_records  = parse_fasta(ss_fasta)
        common_ids  = sorted(set(seq_records) & set(ss_records))

        self.max_len = max_len
        self.samples = []

        for seq_id in common_ids:
            seq = seq_records[seq_id]
            ss  = ss_records[seq_id]
            if len(seq) == len(ss) and len(seq) > 0:
                # TODO: 如果序列长度超过 max_len，截断到 max_len
                # 提示：seq = seq[:max_len]，ss 同理
                if len(seq) > max_len:
                    seq = seq[:max_len]
                    ss = ss[:max_len]
                self.samples.append((seq, ss))

        print(f"加载完成：{len(self.samples)} 条，max_len={max_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ════════════════════════════════════════════════════════════
# 新增：数据集统计信息
# ════════════════════════════════════════════════════════════

def dataset_stats(dataset):
    """
    打印数据集的基本统计信息：
      - 样本数量
      - 序列长度分布（min / max / mean）
      - 标签分布（H / E / C 各占多少比例）

    提示：
      - 遍历 dataset，收集每条序列的长度和标签
      - 用 sum() / len() 计算均值
      - 标签计数：对每条 ss 字符串统计 H/E/C 的个数
    """
    lengths  = []
    label_counts = {"H": 0, "E": 0, "C": 0}

    for seq, ss in dataset:
        # TODO: 收集长度
        lengths.append(len(seq))
        # TODO: 统计标签
        for label in ss:
            if label in label_counts:
                label_counts[label] += 1

    total_labels = sum(label_counts.values())

    print(f"样本数量 : {len(dataset)}")
    print(f"序列长度 : min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"标签分布 :")
    for label, count in label_counts.items():
        print(f"  {label} : {count:6d}  ({count/total_labels:.1%})")



# ── 从 Week 5 复制的组件（GELU 版本）────────────────────────
# 把所有 nn.ReLU() 替换为 nn.GELU()
# 其余代码和 Week 5 完全相同

# scaled_dot_product_attention  （直接复制，不需要改）
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
# MultiHeadAttention             （直接复制，不需要改）

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

# PositionalEncoding             （直接复制，不需要改）

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

# ProteinTransformer  （直接复制，不需要改）

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
# Early Stopping
# ════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss):
        """
        返回 True  → val_loss 有改善，调用方应保存模型
        返回 False → 没有改善，counter +1
        """
        # TODO: 实现上面描述的逻辑
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ════════════════════════════════════════════════════════════
# 训练 / 验证函数
# ════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for tokens, labels, mask in loader:
        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss   = loss_fn(logits.view(-1, 3), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    n_batches     = 0
    for tokens, labels, mask in loader:
        logits = model(tokens, mask)
        loss   = loss_fn(logits.view(-1, 3), labels.view(-1))
        total_loss += loss.item()
        n_batches  += 1
        preds = logits.argmax(dim=-1)
        valid = (labels != -1)
        total_correct += (preds[valid] == labels[valid]).sum().item()
        total_tokens  += valid.sum().item()
    return total_loss / n_batches, total_correct / total_tokens

# ════════════════════════════════════════════════════════════
# 新增：Warmup + Cosine Decay 调度器
# ════════════════════════════════════════════════════════════

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    组合 Warmup + Cosine Decay

    参数：
        optimizer     : 优化器
        warmup_epochs : 预热阶段的 epoch 数
        total_epochs  : 总训练 epoch 数

    返回：
        scheduler : SequentialLR，先 warmup 再 cosine decay
    """
    # 第一阶段：LinearLR，从 start_factor 线性增长到 1.0
    # start_factor=1/warmup_epochs 意味着第一个 epoch 的 lr = max_lr / warmup_epochs
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1.0 / warmup_epochs,
        end_factor   = 1.0,
        total_iters  = warmup_epochs
    )

    # 第二阶段：CosineAnnealingLR，从 max_lr 降到 eta_min
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = total_epochs - warmup_epochs,
        eta_min = 1e-5
    )

    # 用 SequentialLR 把两个阶段串联起来
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_epochs]   # 在第 warmup_epochs 个 epoch 切换
    )
    return scheduler


# ════════════════════════════════════════════════════════════
# 主训练流程
# ════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    from torch.utils.data import random_split

    # ── 数据 ────────────────────────────────────────────────
    generate_fasta_files("seq.fasta", "ss.fasta", n_samples=500)
    dataset  = ProteinDatasetV2("seq.fasta", "ss.fasta", max_len=128)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ── 模型 ────────────────────────────────────────────────
    model     = ProteinTransformer(d_model=64, num_heads=4, d_ff=256, num_layers=3, dropout=0.1)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── 新增：调度器 ─────────────────────────────────────────
    warmup_epochs = 5
    max_epochs    = 100
    scheduler = get_scheduler(optimizer, warmup_epochs, max_epochs)

    # ── Early Stopping ───────────────────────────────────────
    early_stopping   = EarlyStopping(patience=10, min_delta=1e-4)
    best_model_state = None

    # ── 训练循环 ─────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}  {'LR':>8}  {'Best':>5}")
    print("-" * 62)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        # ── 新增：调度器 step ────────────────────────────────
        scheduler.step()

        # 获取当前 lr（用于打印）
        current_lr = optimizer.param_groups[0]['lr']

        improved = early_stopping.step(val_loss)
        if improved:
            import copy
            best_model_state = copy.deepcopy(model.state_dict())

        flag = "✅" if improved else ""
        if epoch % 5 == 0 or early_stopping.should_stop:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}"
                  f"  {val_acc:>8.2%}  {current_lr:>8.2e}  {flag}")

        if early_stopping.should_stop:
            print(f"\nEarly Stopping 触发！在 Epoch {epoch} 停止")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最优模型（best val_loss = {early_stopping.best_loss:.4f}）")

    final_loss, final_acc = evaluate(model, val_loader, loss_fn)
    print(f"\n最终验证集结果：Loss={final_loss:.4f}，Acc={final_acc:.2%}")

main()
