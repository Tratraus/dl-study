# day3_lstm.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1~2: 数据生成（直接复用）
# ============================================================

VOCAB = ['A', 'C', 'G', 'T']
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 20
char2idx = {ch: idx for idx, ch in enumerate(VOCAB)}

def onehot_encode(seq):
    matrix = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq):
        matrix[i, char2idx[ch]] = 1.0
    return matrix

def generate_dataset(n_samples=1000, seq_len=SEQ_LEN, random_state=42):
    rng = np.random.RandomState(random_state)
    sequences, labels = [], []
    for _ in range(n_samples):
        seq = ''.join(rng.choice(VOCAB, size=seq_len))
        gc_content = (seq.count('G') + seq.count('C')) / seq_len
        labels.append(1 if gc_content > 0.5 else 0)
        sequences.append(seq)
    return sequences, labels

class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        encoded = np.array([onehot_encode(seq) for seq in sequences])
        self.X = torch.tensor(encoded, dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

sequences, labels = generate_dataset(n_samples=1000)
seq_train, seq_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.3, random_state=42, stratify=labels)
seq_val, seq_test, y_val, y_test = train_test_split(
    seq_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_loader = DataLoader(DNADataset(seq_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(DNADataset(seq_val,   y_val),   batch_size=32, shuffle=False)
test_loader  = DataLoader(DNADataset(seq_test,  y_test),  batch_size=32, shuffle=False)

# ============================================================
# Part 3: 理解 nn.LSTM 和 nn.RNN 的区别
# ============================================================

print("=" * 50)
print("Part 3: nn.LSTM 形状实验")
print("=" * 50)

lstm_test = nn.LSTM(input_size=VOCAB_SIZE, hidden_size=32,
                    num_layers=1, batch_first=True)

x_dummy = torch.zeros(8, SEQ_LEN, VOCAB_SIZE)
h0 = torch.zeros(1, 8, 32)
c0 = torch.zeros(1, 8, 32)   # LSTM 比 RNN 多这一个

output, (hn, cn) = lstm_test(x_dummy, (h0, c0))

print(f"输入 x 形状：        {x_dummy.shape}")
print(f"output 形状：        {output.shape}")
print(f"hn 形状：            {hn.shape}")
print(f"cn 形状：            {cn.shape}")   # 新增

# 统计参数量对比
rnn_test  = nn.RNN(input_size=VOCAB_SIZE,  hidden_size=32, batch_first=True)
rnn_params  = sum(p.numel() for p in rnn_test.parameters())
lstm_params = sum(p.numel() for p in lstm_test.parameters())
print(f"\nRNN  参数量：{rnn_params}")
print(f"LSTM 参数量：{lstm_params}")
print(f"LSTM 参数量是 RNN 的 {lstm_params / rnn_params:.1f} 倍")
# 你觉得为什么是这个倍数？（提示：LSTM 有几个门？）

# ============================================================
# Part 4: 构建 LSTM 分类模型
# ============================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # 你来写：
        # 1. 定义 self.lstm = nn.LSTM(...)，batch_first=True
        # 2. 定义 self.fc  = nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            )
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 你来写：
        # 1. 初始化 h0 和 c0（都是全零，注意 device=x.device）
        # 2. 调用 self.lstm(x, (h0, c0))，得到 output 和 (hn, cn)
        # 3. 取最后一步隐藏状态
        # 4. 经过 fc + sigmoid，返回 (batch,) 形状
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        last_hidden = hn.squeeze(0)  # (batch, hidden_size)
        logits = self.fc(last_hidden) # (batch, 1)
        probs = torch.sigmoid(logits).squeeze(1)  # (batch,)
        return probs

# 测试
model = LSTMClassifier(input_size=VOCAB_SIZE, hidden_size=32)
x_test = torch.zeros(4, SEQ_LEN, VOCAB_SIZE)
out = model(x_test)
print(f"\nLSTMClassifier 输出形状：{out.shape}")   # (4,)
print(f"LSTMClassifier 输出值：  {out}")

# ============================================================
# Part 5: 训练 + 评估函数（直接复用）
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
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
            loss = criterion(output, yb)
            total_loss    += loss.item() * len(yb)
            preds = (output > 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

# ============================================================
# Part 6: 三模型对比实验
# ============================================================

# 复用 Day 2 的 RNN 和 MLP
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

class FlattenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * VOCAB_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def run_experiment(model, name, epochs=30):
    """训练一个模型并返回 test_acc"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"\n{'='*60}")
    print(f"训练 {name}")
    print(f"{'='*60}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"→ Test Acc: {test_acc:.4f}")
    return test_acc

results = {}
results['FlattenMLP']     = run_experiment(FlattenMLP(), 'FlattenMLP')
results['RNNClassifier']  = run_experiment(RNNClassifier(VOCAB_SIZE, 32), 'RNNClassifier')
results['LSTMClassifier'] = run_experiment(LSTMClassifier(VOCAB_SIZE, 32), 'LSTMClassifier')

print("\n" + "=" * 40)
print("三模型 Test Acc 汇总")
print("=" * 40)
for name, acc in results.items():
    print(f"  {name:<20} {acc:.4f}")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# Part 3 里打印了 RNN 和 LSTM 的参数量，LSTM 是 RNN 的约 4 倍。
# 请解释为什么是 4 倍？（提示：LSTM 有几个门？每个门的结构是什么？）

# 你的回答：因为LSTM共有3个门，其中输入门有两个权重矩阵（输入和隐藏状态），
# 遗忘门和输出门各有一个权重矩阵，所以总共有4个权重矩阵，而RNN只有一个权重矩阵。
# Answer：关键纠正：候选值是独立的第 3 个计算单元，不属于输入门。输入门只控制"写多少"，候选值才是"写什么内容"，两者是分开的。

# 问题 2：
# LSTM 的细胞状态更新公式是：c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
# 如果遗忘门 f_t 全部输出 1，输入门 i_t 全部输出 0，会发生什么？
# 这在生物序列分析中对应什么场景？

# 你的回答：f_t全输出1代表信息完全保留，而i_t全输出0代表无新信息的加入
# 这意味着细胞状态没有进行任何更新，完全依赖于之前的状态。这在生物序列分析中可能对应一个位置对分类结果没有任何贡献的场景。

# 问题 3：
# 三模型对比中，LSTM 的 Test Acc 和 RNN 相比如何？
# 结合 Day 2 的结论（GC 含量是位置无关任务），解释你观察到的结果。

# 你的回答：（运行后回答）
#  FlattenMLP           0.9933
#  RNNClassifier        0.9533
#  LSTMClassifier       1.0000
# LSTM的表现比RNN更好，因为LSTM能够更好地捕捉序列中的长期依赖关系，而GC含量是位置无关的任务，LSTM能够更有效地利用整个序列的信息来做出预测。0
# Answer："LSTM 能更好捕捉长期依赖"——这个解释对这个任务不适用。
# GC 含量任务序列长度只有 20，根本不存在"长期依赖"问题，RNN 在长度 20 上完全不会梯度消失。
# 这次结果（LSTM 1.0000 > MLP 0.9933 > RNN 0.9533）
# 是一次随机种子下的结果，存在偶然性。

# 更重要的观察是：
#   - 三个模型在这个任务上都能达到 95%+ 的准确率
#   - LSTM 的 Val Acc 在 Epoch 30 达到了 1.0000（150个样本全对）
#   - 这可能是测试集恰好"容易"，而不是 LSTM 本质上更强

# 真正的结论（和 Day 2 一致）：
#   GC 含量是位置无关的全局统计任务
#   → 三个模型都能解决，差异来自随机性和超参数
#   → 不能从这个任务判断 LSTM > RNN
#   → 需要一个真正依赖位置信息的任务才能体现差异（Day 4）
