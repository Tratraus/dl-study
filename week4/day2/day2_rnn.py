# day2_rnn.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1~2: 复用 Day 1 的数据生成（直接复制）
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
        self.Y = torch.tensor(labels, dtype=torch.float32)  # ← 注意：BCE 需要 float
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
# Part 3: 理解 nn.RNN 的输入输出形状
# ============================================================

print("=" * 50)
print("Part 3: nn.RNN 形状实验")
print("=" * 50)

rnn_test = nn.RNN(input_size=VOCAB_SIZE, hidden_size=32,
                  num_layers=1, batch_first=True)

x_dummy = torch.zeros(8, SEQ_LEN, VOCAB_SIZE)   # (batch=8, seq=20, input=4)
h0      = torch.zeros(1, 8, 32)                 # (layers=1, batch=8, hidden=32)

output, hn = rnn_test(x_dummy, h0)

print(f"输入 x 形状：       {x_dummy.shape}")
print(f"初始隐藏状态 h0：   {h0.shape}")
print(f"输出 output 形状：  {output.shape}")   # (8, 20, 32)
print(f"最终隐藏状态 hn：   {hn.shape}")       # (1, 8, 32)

# 验证两种取最后隐藏状态的方式等价
method1 = hn.squeeze(0)          # (8, 32)
method2 = output[:, -1, :]       # (8, 32)
print(f"\nmethod1 (hn.squeeze) 形状：    {method1.shape}")
print(f"method2 (output[:,-1,:]) 形状：{method2.shape}")
print(f"两种方法结果是否相同：{torch.allclose(method1, method2)}")

# ============================================================
# Part 4: 构建 RNN 分类模型
# ============================================================

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # 你来写：
        # 1. 定义 self.rnn = nn.RNN(...)，记得 batch_first=True
        # 2. 定义 self.fc  = nn.Linear(hidden_size, 1)
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # x 形状：(batch, seq_len, input_size)
        # 你来写：
        # 1. 初始化 h0（全零，形状：(num_layers, batch_size, hidden_size)）
        #    提示：h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # 2. 调用 self.rnn(x, h0)，得到 output 和 hn
        # 3. 取最后一步的隐藏状态（用 hn.squeeze(0) 或 output[:, -1, :]）
        # 4. 经过 self.fc，再经过 sigmoid，返回 (batch,) 形状的概率

        batch_size = x.size(0)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=x.device)
        output, hn = self.rnn(x, h0)
        last_hidden = hn.squeeze(0)  # (batch, hidden_size)
        # last_hidden = output[:, -1, :]  # (batch, hidden_size)  # 这两种方式等价
        logits = self.fc(last_hidden) # (batch, 1)
        probs = torch.sigmoid(logits).squeeze(1)  # (batch,)
        return probs

# 测试模型
model = RNNClassifier(input_size=VOCAB_SIZE, hidden_size=32)
x_test = torch.zeros(4, SEQ_LEN, VOCAB_SIZE)
out = model(x_test)
print(f"\nRNNClassifier 输出形状：{out.shape}")   # 应该是 (4,)
print(f"RNNClassifier 输出值：  {out}")           # 应该是 4 个 0~1 之间的数

# ============================================================
# Part 5: 训练函数（今天新增 val 评估，修复 Day 1 的遗漏）
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
    # 你来写（和 Week 3 一样，注意 BCE 的 acc 计算方式）
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
# Part 6: 完整训练
# ============================================================

NUM_EPOCHS = 30
model     = RNNClassifier(input_size=VOCAB_SIZE, hidden_size=32)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n" + "=" * 60)
print("开始训练 RNNClassifier")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ============================================================
# Part 7: 对比实验（复用 Day 1 的 FlattenMLP）
# ============================================================

class FlattenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * VOCAB_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

print("\n" + "=" * 60)
print("对比训练 FlattenMLP")
print("=" * 60)

mlp_model = FlattenMLP()
mlp_criterion = nn.BCELoss()
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(mlp_model, train_loader, mlp_criterion, mlp_optimizer)
    val_loss,   val_acc   = evaluate(mlp_model, val_loader, mlp_criterion)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

mlp_test_loss, mlp_test_acc = evaluate(mlp_model, test_loader, mlp_criterion)
print(f"\nFlattenMLP  Test Acc: {mlp_test_acc:.4f}")
print(f"RNNClassifier Test Acc: {test_acc:.4f}")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# nn.RNN 的 output 形状是 (batch, seq_len, hidden_size)，
# hn 的形状是 (num_layers, batch, hidden_size)。
# 对于序列分类任务，我们为什么只取最后一步，而不是用所有时间步的 output？

# 你的回答：因为在分类任务中，信息一般已经聚集于最后一步，取最后一步即可获取分类结果

# 问题 2：
# RNN 的公式是 h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)。
# 如果序列长度是 1000，h_0 里的信息经过 1000 次 tanh 运算后还剩多少？
# 这会导致什么问题？

# 你的回答：会被大量压缩，导致梯度消失问题

# 问题 3：
# 对比实验里，RNNClassifier 和 FlattenMLP 在这个任务上的 Test Acc 谁更高？
# 你觉得为什么？（提示：想想这个任务的标签是怎么定义的）

# 你的回答：（运行后根据结果回答）
# FlattenMLP Test Acc: 0.9867
# RNNClassifier Test Acc: 0.9667
# 可能是因为这个任务的标签是根据整个序列的 GC 含量定义的，MLP 直接看到整个序列的展平向量，
# 可能更容易捕捉到全局特征，而 RNN 需要通过时间步的递归来捕捉全局信息，可能更难训练和泛化。
