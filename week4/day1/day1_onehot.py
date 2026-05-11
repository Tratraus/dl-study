# day1_onehot.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 定义词表和 one-hot 编码函数
# ============================================================

VOCAB = ['A', 'C', 'G', 'T']
VOCAB_SIZE = len(VOCAB)           # 4
SEQ_LEN = 20

# 字母 → 索引的映射字典
char2idx = {ch: idx for idx, ch in enumerate(VOCAB)}
print("char2idx:", char2idx)

def onehot_encode(seq):
    """
    输入：字符串，如 "ATGCATGC..."
    输出：numpy array，形状 (SEQ_LEN, VOCAB_SIZE)，dtype float32
    """
    matrix = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq):
        matrix[i, char2idx[ch]] = 1.0
    return matrix

# 测试一下
test_seq = "ATGC"
encoded = onehot_encode(test_seq)
print(f"\n序列 '{test_seq}' 的 one-hot 编码：")
print(encoded)
print(f"形状：{encoded.shape}")   # 应该是 (4, 4)


# ============================================================
# Part 2: 生成模拟数据集
# ============================================================

def generate_dataset(n_samples=1000, seq_len=SEQ_LEN, random_state=42):
    """
    生成 n_samples 条随机 DNA 序列
    标签规则：GC 含量 > 0.5 → 1，否则 → 0
    返回：sequences（字符串列表），labels（int 列表）
    """
    rng = np.random.RandomState(random_state)
    sequences = []
    labels = []
    for _ in range(n_samples):
        seq = ''.join(rng.choice(VOCAB, size=seq_len))
        gc_content = (seq.count('G') + seq.count('C')) / seq_len
        label = 1 if gc_content > 0.5 else 0
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

sequences, labels = generate_dataset(n_samples=1000)
print(f"\n数据集大小：{len(sequences)}")
print(f"前3条序列：{sequences[:3]}")
print(f"前3条标签：{labels[:3]}")
print(f"类别分布：0={labels.count(0)}, 1={labels.count(1)}")


# ============================================================
# Part 3: 构建 Dataset
# ============================================================

class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        # 你来写：
        # 1. 对每条序列调用 onehot_encode，得到 (SEQ_LEN, VOCAB_SIZE) 的矩阵
        # 2. 把所有矩阵堆叠成 tensor，形状 (N, SEQ_LEN, VOCAB_SIZE)，dtype=float32
        # 3. 把 labels 转成 tensor，dtype=long
        self.X = torch.tensor([onehot_encode(seq) for seq in sequences], dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # 你来写
        return len(self.X)

    def __getitem__(self, idx):
        # 你来写
        return self.X[idx], self.Y[idx]

# ============================================================
# Part 4: 划分数据集 + 构建 DataLoader
# ============================================================

# 你来写：
# 1. 用 train_test_split 划分 train(70%) / val(15%) / test(15%)
# 2. 构建三个 DataLoader，batch_size=32
# 3. 打印 train/val/test 的样本数量
# 4. 取一个 batch，打印 xb.shape 和 yb.shape

X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_dataset = DNADataset(X_train, y_train)
val_dataset = DNADataset(X_val, y_val)
test_dataset = DNADataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 取一个 batch，打印 xb.shape 和 yb.shape
xb, yb = next(iter(train_loader))
print(f"xb.shape: {xb.shape}, yb.shape: {yb.shape}")

# ============================================================
# Part 5: 一个最简单的分类模型（先用 MLP，不用 RNN）
# ============================================================

# 注意：MLP 不能直接处理 (SEQ_LEN, VOCAB_SIZE) 的矩阵
# 需要先把它"展平"成一个向量：(SEQ_LEN * VOCAB_SIZE,) = (80,)
# 这叫做 flatten，是今天的关键操作

class FlattenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                    # (batch, 20, 4) → (batch, 80)
            nn.Linear(SEQ_LEN * VOCAB_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),               # 二分类，输出 1 个 logit
            nn.Sigmoid()                    # 输出概率
        )

    def forward(self, x):
        return self.net(x).squeeze(1)       # 形状：(batch,)

model = FlattenMLP()
print(f"\n模型结构：")
print(model)

# 测试 forward pass
dummy_input = torch.zeros(4, SEQ_LEN, VOCAB_SIZE)   # 模拟 4 个样本
output = model(dummy_input)
print(f"\ndummy input 形状：{dummy_input.shape}")
print(f"模型输出形状：{output.shape}")    # 应该是 (4,)
print(f"模型输出值：{output}")            # 应该是 4 个 0~1 之间的数


# ============================================================
# Part 6: 完整训练（复用 Week 3 的训练框架）
# ============================================================

# 你来写：
# 1. criterion = BCELoss（二分类用 BCE，不是 CrossEntropy）
# 2. optimizer = Adam, lr=0.001
# 3. 训练 30 个 epoch
# 4. 每 5 个 epoch 打印：Epoch / Train Loss / Train Acc / Val Loss / Val Acc
# 5. 训练结束后打印 Test Acc

# 提示：BCE 的 accuracy 计算方式和 CrossEntropy 不同：
#   preds = (output > 0.5).float()
#   correct = (preds == yb.float()).sum().item()
NUM_EPOCHS = 30
model = FlattenMLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        preds = (output > 0.5).float()
        train_correct += (preds == yb.float()).sum().item()
        train_total += xb.size(0)
    model.eval()
    train_loss /= len(train_loader.dataset)
    if (epoch + 1) % 5 == 0:
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# one-hot 编码后，序列 "AAAA" 和 "TTTT" 的矩阵有什么不同？
# 它们的形状一样吗？

# 你的回答：形状一样，都是 (4, 4) 的矩阵，但内容不同。AAAA 的 one-hot 编码是一个第一列全 1 的矩阵，而 TTTT 的 one-hot 编码是一个第四列全1 的矩阵。

# 问题 2：
# FlattenMLP 用 nn.Flatten() 把 (batch, 20, 4) 展平成 (batch, 80)。
# 这样做丢失了什么信息？

# 你的回答：丢失了二维结构，序列的位置信息和相邻碱基之间的关系。

# 问题 3：
# 为什么这个任务用 BCELoss 而不是 CrossEntropyLoss？
# 什么情况下用哪个？

# 你的回答：BCELoss 用于二分类任务，而 CrossEntropyLoss 用于多分类任务。