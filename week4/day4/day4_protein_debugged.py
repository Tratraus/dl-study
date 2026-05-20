# day4_protein.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 蛋白质序列的编码
# ============================================================

# 20 种标准氨基酸
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE  = len(AMINO_ACIDS)   # 20
SEQ_LEN     = 50
MOTIF       = "LSEE"

char2idx = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}
print(f"氨基酸词表大小：{VOCAB_SIZE}")
print(f"char2idx 示例：L={char2idx['L']}, S={char2idx['S']}, "
      f"E={char2idx['E']}")

def onehot_encode(seq):
    matrix = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq):
        matrix[i, char2idx[ch]] = 1.0
    return matrix

# ============================================================
# Part 2: 生成蛋白质数据集
# ============================================================

def generate_protein_dataset(n_samples=2000, seq_len=SEQ_LEN,
                              motif=MOTIF, random_state=42):
    """
    生成蛋白质序列数据集
    正样本（label=1）：随机序列中随机位置插入 motif
    负样本（label=0）：纯随机序列（确保不含 motif）
    各占 50%
    """
    rng = np.random.RandomState(random_state)
    sequences, labels = [], []
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # 生成正样本
    for _ in range(n_pos):
        while True:
            seq = list(rng.choice(AMINO_ACIDS, size=seq_len))
            # 随机选一个插入位置
            insert_pos = rng.randint(0, seq_len - len(motif) + 1)
            for j, ch in enumerate(motif):
                seq[insert_pos + j] = ch
            seq_str = ''.join(seq)
            # 确保只含一个 motif（避免随机生成了多个）
            if seq_str.count(motif) >= 1:
                sequences.append(seq_str)
                labels.append(1)
                break

    # 生成负样本（确保不含 motif）
    for _ in range(n_neg):
        while True:
            seq = ''.join(rng.choice(AMINO_ACIDS, size=seq_len))
            if motif not in seq:
                sequences.append(seq)
                labels.append(0)
                break

    # 打乱顺序
    combined = list(zip(sequences, labels))
    rng.shuffle(combined)
    sequences, labels = zip(*combined)
    return list(sequences), list(labels)


sequences, labels = generate_protein_dataset(n_samples=2000)
print(f"\n数据集大小：{len(sequences)}")
print(f"类别分布：0={labels.count(0)}, 1={labels.count(1)}")
print(f"正样本示例：{sequences[0]}")
print(f"  含 motif：{MOTIF in sequences[0]}")
print(f"负样本示例：{sequences[-1]}")
print(f"  含 motif：{MOTIF in sequences[-1]}")


# ============================================================
# Part 3: Dataset + DataLoader
# ============================================================

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        encoded = np.array([onehot_encode(seq) for seq in sequences])
        self.X = torch.tensor(encoded, dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

seq_train, seq_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.3, random_state=42, stratify=labels)
seq_val, seq_test, y_val, y_test = train_test_split(
    seq_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_loader = DataLoader(ProteinDataset(seq_train, y_train),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(ProteinDataset(seq_val,   y_val),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(ProteinDataset(seq_test,  y_test),
                          batch_size=32, shuffle=False)

print(f"\nTrain: {len(seq_train)}, Val: {len(seq_val)}, Test: {len(seq_test)}")


# ============================================================
# Part 4: 三个模型（直接复用，只改输入维度）
# ============================================================

class FlattenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * VOCAB_SIZE, 128), nn.ReLU(),
            nn.Dropout(0.5),                       # ← 新增
            nn.Linear(128, 64),                    nn.ReLU(),
            nn.Dropout(0.3),                       # ← 新增
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0),
                         self.rnn.hidden_size, device=x.device)
        _, hn = self.rnn(x, h0)
        return torch.sigmoid(self.fc(hn.squeeze(0))).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size, device=x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return torch.sigmoid(self.fc(hn.squeeze(0))).squeeze(1)


# ============================================================
# Part 5: 训练 + 评估（直接复用）
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
# ============================================================
# Part 5.5: Early Stopping（加在 evaluate 函数之后）
# ============================================================

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience   = patience    # 连续多少个 epoch 没有提升就停止
        self.min_delta  = min_delta   # 提升幅度小于这个值不算"提升"
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None        # 保存最佳权重

    def step(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience  # 返回 True = 该停了

    def restore(self, model):
        """训练结束后，把模型权重恢复到 val_acc 最高的那个 epoch"""
        if self.best_state:
            model.load_state_dict(self.best_state)

# ============================================================
# Part 6: 对比实验
# ============================================================

def run_experiment(model, name, epochs=60, lr=0.0005, patience=8):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper   = EarlyStopping(patience=patience)   # ← 每次实验新建一个

    print(f"\n{'='*60}\n训练 {name}\n{'='*60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f}")

        # ↓ 每个 epoch 结束后调用一次
        if stopper.step(val_acc, model):
            print(f"  ← Early stop! epoch={epoch}, best val acc={stopper.best_acc:.4f}")
            break  # 跳出训练循环

    # ↓ 训练结束后恢复最佳权重（无论是正常结束还是早停）
    stopper.restore(model)

    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"→ Test Acc: {test_acc:.4f}")
    return test_acc


results = {}
results['FlattenMLP']    = run_experiment(FlattenMLP(), 'FlattenMLP')
results['RNN(h=64)']     = run_experiment(
    RNNClassifier(VOCAB_SIZE, hidden_size=64), 'RNN(h=64)')
results['LSTM(h=64)']    = run_experiment(
    LSTMClassifier(VOCAB_SIZE, hidden_size=64), 'LSTM(h=64)')

print("\n" + "=" * 40)
print("三模型 Test Acc 汇总")
print("=" * 40)
for name, acc in results.items():
    print(f"  {name:<20} {acc:.4f}")


# ============================================================
# Part 7: 可解释性实验——LSTM 在哪个位置"发现"了 motif？
# ============================================================

print("\n" + "=" * 50)
print("Part 7: 隐藏状态可视化")
print("=" * 50)

# 取一条正样本，找到 motif 的位置
lstm_model = LSTMClassifier(VOCAB_SIZE, hidden_size=64)
# 用已训练的模型（重新训练一次）
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005)
for epoch in range(1, 41):
    train_one_epoch(lstm_model, train_loader, criterion, optimizer)

# 找一条含 motif 的序列
pos_seq = next(s for s in sequences if MOTIF in s)
motif_pos = pos_seq.index(MOTIF)
print(f"分析序列（前20个氨基酸）：{pos_seq[:20]}...")
print(f"motif '{MOTIF}' 位于位置：{motif_pos}~{motif_pos+len(MOTIF)-1}")

# 提取每个时间步的隐藏状态范数
lstm_model.eval()
x = torch.tensor(onehot_encode(pos_seq)).unsqueeze(0)  # (1, 50, 20)
h0 = torch.zeros(1, 1, 64)
c0 = torch.zeros(1, 1, 64)
with torch.no_grad():
    output, (hn, cn) = lstm_model.lstm(x, (h0, c0))
    # output 形状：(1, 50, 64)

# 计算每个时间步隐藏状态的 L2 范数（代表"激活强度"）
hidden_norms = output.squeeze(0).norm(dim=1).numpy()  # (50,)

# 你来写：
# 用 matplotlib 画一个折线图：
#   x 轴：时间步（0~49）
#   y 轴：hidden_norms
#   在 motif 位置（motif_pos ~ motif_pos+3）画一个红色竖线或阴影
#   标题：f"LSTM Hidden State Norm - motif '{MOTIF}' at pos {motif_pos}"
# 保存为 hidden_norm.png

import matplotlib.pyplot as plt
# 你来写
plt.figure(figsize=(10, 4))
plt.plot(hidden_norms, marker='o')
plt.axvspan(motif_pos, motif_pos + len(MOTIF) - 1, color='red', alpha=0.3)
plt.title(f"LSTM Hidden State Norm - motif '{MOTIF}' at pos {motif_pos}")
plt.xlabel("Time Step")
plt.ylabel("Hidden State L2 Norm")
plt.grid()
plt.savefig("./week4/day4/hidden_norm_debugged_version.png")
plt.show()

# ============================================================
# Questions
# ============================================================

# 问题 1：
# 这个任务（motif 检测）和 Day 2~3 的 GC 含量任务，
# 哪个更需要位置信息？为什么 MLP 在这个任务上表现更差？

# 你的回答：
# FlattenMLP           0.6767
# RNN(h=64)            0.8200
# LSTM(h=64)           0.9867
# 这个任务更需要位置信息，因为 motif 的位置对于分类结果非常关键，
# 而 MLP 直接展平输入，无法捕捉序列中的位置信息，因此表现更差。

# 问题 2：
# Part 7 的可视化里，LSTM 的隐藏状态范数在 motif 位置附近有什么变化？
# 你怎么解释这个现象？

# 你的回答：（运行后回答）
# 在 motif 位置附近，LSTM 的隐藏状态范数明显增大，说明 LSTM 在这些位置对输入序列的关注度更高。
# 这表明 LSTM 能够识别出 motif 的存在，并在这些位置产生更强的激活，从而影响最终的分类结果。

# 问题 3：
# 如果把 MOTIF 从 "LSEE"（长度4）改成 "LS"（长度2），
# 你预测三个模型的相对表现会怎么变化？为什么？

# 你的回答：MLP会上升，RNN和LSTM也会上升，但MLP提升更大。
# 因为 MOTIF 变短后，随机序列中出现 MOTIF 的概率增加，
# MLP 可能会误判更多负样本为正样本，从而提升了准确率。
# 而 RNN 和 LSTM 仍然需要捕捉序列中的位置信息，虽然也会提升，但相对来说提升幅度可能不如 MLP 明显。
# Answer:
# MOTIF = "LS"（长度 2）时：

# 对 LSTM：
#   只需要记住"上一步是 L，这一步是 S"就能检测
#   → 更短的依赖距离，LSTM 更容易学，准确率会更高

# 对 MLP：
#   "LS" 在展平向量里的信号比 "LSEE" 更模糊
#   （4个字符的组合比2个字符更独特，更容易被统计特征区分）
#   → MLP 反而更难，因为 "LS" 单独出现的概率更高，噪声更多

# 结论：MOTIF 越短 → LSTM 相对优势越大
#       MOTIF 越长 → 需要更长记忆，LSTM 仍然更好，但训练更难
