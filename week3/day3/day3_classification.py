# day3_classification.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 数据读取与划分
# ============================================================

df = pd.read_csv("week3/data/iris.csv")
X_np = df.iloc[:, :4].values.astype(np.float32)
Y_np = df["species"].values.astype(np.int64)

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_np, Y_np, test_size=0.15, random_state=42, stratify=Y_np
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.176, random_state=42, stratify=Y_temp
)

# ============================================================
# Part 2: 标准化
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# Part 3: Dataset & DataLoader
# ============================================================

class IrisDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(IrisDataset(X_train_scaled, Y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(IrisDataset(X_val_scaled,   Y_val),   batch_size=16, shuffle=False)
test_loader  = DataLoader(IrisDataset(X_test_scaled,  Y_test),  batch_size=16, shuffle=False)

# ============================================================
# Part 4: 模型定义
# ============================================================

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)   # 输出 3 个 logits，对应 3 个类别
        )

    def forward(self, x):
        return self.net(x)

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("模型结构：")
print(model)

# ============================================================
# Part 5: 训练函数
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * len(yb)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += len(yb)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ============================================================
# Part 6: 验证函数（你来补全）
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            # 你来写：
            # 1. 累加 loss（参考 train_one_epoch 的写法）
            total_loss += loss.item() * len(yb)
            # 2. 计算 preds（取 logits 中最大值的索引）
            preds = logits.argmax(dim=1)
            # 3. 累加 correct 和 samples
            total_correct += (preds == yb).sum().item()
            total_samples += len(yb)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ============================================================
# Part 7: 训练主循环
# ============================================================

NUM_EPOCHS = 50

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
print("-" * 55)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)

    if epoch % 5 == 0:
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f}")

# ============================================================
# Part 8: 测试集最终评估
# ============================================================

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n========== Test Result ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# 模型最后一层是 nn.Linear(16, 3)，输出 3 个数（logits）。
# 为什么不在最后加一个 Softmax？

# 你的回答：因为nn.CrossEntropyLoss()这个损失函数内部已经包含了Softmax的计算，所以我们在模型的最后一层不需要再加一个Softmax。

# 问题 2：
# train_one_epoch 里有 model.train()，evaluate 里有 model.eval()。
# 如果两个函数都不写这两行，会有什么潜在问题？

# 你的回答：如果不写 model.train()，模型在训练时可能
# 不会启用 dropout 和 batch normalization 的训练模式，
# 导致训练效果不佳。如果不写 model.eval()，
# 在验证或测试时，模型可能仍然启用 dropout 和 batch normalization 的训练模式，
# 导致评估结果不准确。

# 问题 3：
# logits.argmax(dim=1) 是什么意思？dim=1 是什么方向？

# 你的回答：logits.argmax(dim=1) 的意思是对 logits 张量的第 1 维（即列方向，每行）进行 argmax 操作，返回每行中最大值的索引。
# dim=1 表示我们在每个样本的类别维度上寻找最大值的索引，这样就得到了模型预测的类别标签。
