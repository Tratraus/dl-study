# day6_bilstm.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 基础设置
# ============================================================

AMINO_ACIDS   = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE    = len(AMINO_ACIDS)   # 20
SEQ_LEN       = 21                 # 固定长度，中心位置 = 10
CENTER_POS    = SEQ_LEN // 2      # = 10

# 带电荷的氨基酸（正电：K R H；负电：D E）
CHARGED       = set("KRHDE")
char2idx      = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}

print(f"序列长度：{SEQ_LEN}，中心位置：{CENTER_POS}")
print(f"带电荷氨基酸：{sorted(CHARGED)}")
print(f"不带电荷氨基酸：{sorted(set(AMINO_ACIDS) - CHARGED)}")

def onehot_encode(seq):
    matrix = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq):
        matrix[i, char2idx[ch]] = 1.0
    return matrix


# ============================================================
# Part 2: 生成数据集
# ============================================================

def generate_charged_dataset(n_samples=4000, random_state=42):
    """
    生成数据集：
    正样本（label=1）：中心氨基酸属于 KRHDE
    负样本（label=0）：中心氨基酸不属于 KRHDE
    各占 50%

    关键设计：
    中心氨基酸的左右各 5 个位置，
    如果左边含有 "P"（脯氨酸），则中心更可能是带电荷的（模拟真实规律）
    → 这个规律需要双向信息才能完整捕捉
    """
    rng = np.random.RandomState(random_state)
    sequences, labels = [], []
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    charged_list     = sorted(CHARGED)
    non_charged_list = sorted(set(AMINO_ACIDS) - CHARGED)

    for _ in range(n_pos):
        seq = list(rng.choice(AMINO_ACIDS, size=SEQ_LEN))
        # 中心设为带电荷氨基酸
        seq[CENTER_POS] = rng.choice(charged_list)
        # 在左边 [5,9] 范围内随机插入 "P"（增加规律性）
        p_pos = rng.randint(CENTER_POS - 5, CENTER_POS)
        seq[p_pos] = 'P'
        sequences.append(''.join(seq))
        labels.append(1)

    for _ in range(n_neg):
        seq = list(rng.choice(AMINO_ACIDS, size=SEQ_LEN))
        # 中心设为不带电荷氨基酸
        seq[CENTER_POS] = rng.choice(non_charged_list)
        sequences.append(''.join(seq))
        labels.append(0)

    combined = list(zip(sequences, labels))
    rng.shuffle(combined)
    sequences, labels = zip(*combined)
    return list(sequences), list(labels)


sequences, labels = generate_charged_dataset(n_samples=4000)
print(f"\n数据集大小：{len(sequences)}")
print(f"类别分布：0={labels.count(0)}, 1={labels.count(1)}")
print(f"正样本示例：{sequences[0]}  中心={sequences[0][CENTER_POS]}")
print(f"负样本示例：{sequences[-1]}  中心={sequences[-1][CENTER_POS]}")


# ============================================================
# Part 3: Dataset + DataLoader
# ============================================================

class ChargedDataset(Dataset):
    def __init__(self, sequences, labels):
        encoded      = np.array([onehot_encode(seq) for seq in sequences])
        self.X       = torch.tensor(encoded, dtype=torch.float32)
        self.Y       = torch.tensor(labels,  dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

seq_train, seq_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.3, random_state=42, stratify=labels)
seq_val, seq_test, y_val, y_test = train_test_split(
    seq_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_loader = DataLoader(ChargedDataset(seq_train, y_train),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(ChargedDataset(seq_val,   y_val),
                          batch_size=64, shuffle=False)
test_loader  = DataLoader(ChargedDataset(seq_test,  y_test),
                          batch_size=64, shuffle=False)

print(f"\nTrain: {len(seq_train)}, Val: {len(seq_val)}, Test: {len(seq_test)}")


# ============================================================
# Part 4: 三个模型
# ============================================================

class UniLSTM(nn.Module):
    """单向 LSTM，只用最后时间步的隐藏状态"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(VOCAB_SIZE, hidden_size, batch_first=True,
                            bidirectional=False)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return torch.sigmoid(self.fc(hn.squeeze(0))).squeeze(1)


class BiLSTM(nn.Module):
    """
    双向 LSTM
    你来写 forward：
    1. 初始化 h0, c0（注意双向时第一维是 2）
    2. 运行 self.lstm，得到 output 和 (hn, cn)
    3. 拼接前向和后向的最终隐藏状态
    4. 经过 fc + sigmoid
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(VOCAB_SIZE, hidden_size, batch_first=True,
                            bidirectional=True)
        self.fc   = nn.Linear(hidden_size * 2, 1)   # 注意：输入维度是 2×hidden

    def forward(self, x):
        # 你来写
        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # hn 形状：(num_layers * num_directions, batch, hidden_size)
        # 因为 num_layers=1，所以 hn 形状是 (2, batch, hidden_size)，其中 hn[0] 是前向最后隐藏状态，hn[1] 是后向最后隐藏状态
        hn_cat = torch.cat([hn[0], hn[1]], dim=1)  # (batch, hidden_size * 2)
        out = torch.sigmoid(self.fc(hn_cat)).squeeze(1)  # (batch,)
        return out


class BiLSTMCenter(nn.Module):
    """
    双向 LSTM，但只用中心位置（位置10）的输出，而不是最后时间步
    这是更合理的做法：中心位置的输出同时包含了左右两侧的信息
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(VOCAB_SIZE, hidden_size, batch_first=True,
                            bidirectional=True)
        self.fc   = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        output, _ = self.lstm(x, (h0, c0))
        # output 形状：(batch, SEQ_LEN, hidden_size * 2)
        # 取中心位置的输出
        center_out = output[:, CENTER_POS, :]   # (batch, hidden_size * 2)
        return torch.sigmoid(self.fc(center_out)).squeeze(1)


# 测试三个模型的输出形状
for name, model in [("UniLSTM",      UniLSTM()),
                    ("BiLSTM",       BiLSTM()),
                    ("BiLSTMCenter", BiLSTMCenter())]:
    xb, _ = next(iter(train_loader))
    try:
        out = model(xb)
        print(f"{name} 输出形状：{out.shape}，值范围：[{out.min():.3f}, {out.max():.3f}]")
    except Exception as e:
        print(f"{name} 报错：{e}")


# ============================================================
# Part 5: 训练 + 评估 + Early Stopping（直接复用）
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        output = model(xb)
        loss   = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(yb)
        preds = (output > 0.5).float()
        total_correct += (preds == yb).sum().item()
        total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            output = model(xb)
            loss   = criterion(output, yb)
            total_loss    += loss.item() * len(yb)
            preds = (output > 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None
    def step(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience
    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def run_experiment(model, name, epochs=80, lr=0.001, patience=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper   = EarlyStopping(patience=patience)
    print(f"\n{'='*60}\n训练 {name}\n{'='*60}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f}")
        if stopper.step(val_acc, model):
            print(f"  ← Early stop! epoch={epoch}, best val={stopper.best_acc:.4f}")
            break
    stopper.restore(model)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"→ Test Acc: {test_acc:.4f}")
    return test_acc

results = {}
results['UniLSTM']      = run_experiment(UniLSTM(),      'UniLSTM')
results['BiLSTM']       = run_experiment(BiLSTM(),       'BiLSTM')
results['BiLSTMCenter'] = run_experiment(BiLSTMCenter(), 'BiLSTMCenter')

print("\n" + "=" * 45)
print("三模型 Test Acc 汇总")
print("=" * 45)
for name, acc in results.items():
    print(f"  {name:<20} {acc:.4f}")


# ============================================================
# Part 6: 参数量对比
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n参数量对比：")
for name, model in [("UniLSTM",      UniLSTM()),
                    ("BiLSTM",       BiLSTM()),
                    ("BiLSTMCenter", BiLSTMCenter())]:
    print(f"  {name:<20} {count_params(model):,} 参数")


# ============================================================
# Questions
# ============================================================

# 问题 1：
# BiLSTM 的参数量是 UniLSTM 的几倍？
# 为什么不是恰好 2 倍？（提示：想想 fc 层的维度变化）

# 你的回答：
# 参数量对比：
#   UniLSTM              22,081 参数
#   BiLSTM               44,161 参数
#   BiLSTMCenter         44,161 参数
# Answer:
# UniLSTM 参数量拆解：
#   LSTM 层：4 × (hidden × input + hidden × hidden + hidden)
#          = 4 × (64×20 + 64×64 + 64)
#          = 4 × (1280 + 4096 + 64)
#          = 4 × 5440 = 21,760
#   fc 层：  hidden × 1 + 1 = 64 + 1 = 65
#   合计：   21,760 + 65 = 21,825

# 等等，实际输出是 22,081，差了 256？
# → 因为 LSTM 有两套偏置（input bias + hidden bias），
#   实际是 4 × (64×20 + 64×64 + 64 + 64) = 22,016，加上 fc 的 65 = 22,081 ✅

# BiLSTM 参数量拆解：
#   LSTM 层（前向）：同 UniLSTM LSTM 层 = 22,016
#   LSTM 层（后向）：同上              = 22,016
#   fc 层：          (64×2) × 1 + 1   = 129
#   合计：           22,016 + 22,016 + 129 = 44,161 ✅

# 倍数：44,161 / 22,081 ≈ 1.9997 ≈ 2倍

# 为什么不是恰好 2 倍？
#   LSTM 层：精确 2 倍（前向 + 后向各一套）
#   fc 层：  UniLSTM fc = 65，BiLSTM fc = 129
#            129 / 65 ≈ 1.98 倍，不是 2 倍
#            （因为偏置项 b 只有 1 个，不随 hidden 翻倍而翻倍）

# 你说"fc 层是 4 倍"是错的——
#   fc 权重：64→1 变成 128→1，权重矩阵从 64 变成 128，是 2 倍
#   fc 偏置：1 个，不变
#   所以 fc 整体是 (128+1)/(64+1) ≈ 1.98 倍，不是 4 倍


# 问题 2：
# BiLSTMCenter 用的是中心位置（位置10）的输出，
# 而 BiLSTM 用的是最后时间步（位置20）的输出。
# 对于这个任务，哪种方式更合理？为什么？

# 你的回答：中间位置，因为中心位置的输出同时包含了左右两侧的信息，
# 而最后时间步的输出主要包含了序列末尾的信息，
# 可能无法充分利用左右两侧的上下文信息。

# 问题 3：
# 双向 LSTM 在哪些生物信息学任务中特别有用？
# 举两个真实的例子，并解释为什么需要双向信息。

# 你的回答：
# 1. 蛋白质二级结构预测：蛋白质的氨基酸序列中，某个氨基酸的结构不仅受其前面的氨基酸影响，也受后面的氨基酸影响。双向 LSTM 可以同时捕捉前向和后向的上下文信息，从而提高预测准确性。
# 2. 基因调控元件识别：在 DNA 序列中，某个位置的功能可能依赖于其上下游的序列信息。双向 LSTM 可以同时考虑上下游的序列特征，有助于更准确地识别调控元件。
