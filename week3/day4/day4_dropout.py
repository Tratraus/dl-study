# day4_dropout.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1~3: 数据准备（和 Day 3 完全一样，直接复用）
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

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

class IrisDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

train_loader = DataLoader(IrisDataset(X_train_s, Y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(IrisDataset(X_val_s,   Y_val),   batch_size=16, shuffle=False)
test_loader  = DataLoader(IrisDataset(X_test_s,  Y_test),  batch_size=16, shuffle=False)

# ============================================================
# Part 4: 模型定义
# ============================================================

# Model A：无 Dropout（你来补全，参考上面的结构说明）
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 你来写：4层，宽度64，激活函数ReLU，无Dropout
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

# Model B：有 Dropout（你来补全）
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Part 5: 训练与评估函数（和 Day 3 完全一样）
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss    += loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

# ============================================================
# Part 6: 训练两个模型并对比
# ============================================================

def run_experiment(model, model_name, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0:
            print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return history

model_a = ModelA()
model_b = ModelB()

history_a = run_experiment(model_a, "Model A: No Dropout")
history_b = run_experiment(model_b, "Model B: With Dropout(0.3)")

# ============================================================
# Part 7: 对比两个模型的 Train/Val Loss 差距
# ============================================================

print("\n========== Final Epoch Gap (Train Loss - Val Loss) ==========")
gap_a = history_a["train_loss"][-1] - history_a["val_loss"][-1]
gap_b = history_b["train_loss"][-1] - history_b["val_loss"][-1]
print(f"Model A gap: {gap_a:.4f}")
print(f"Model B gap: {gap_b:.4f}")
print("（gap 越大，说明过拟合越严重）")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# Dropout 在训练时和推理时的行为不同。
# 如果推理时忘记写 model.eval()，Dropout 仍然在随机关掉神经元，
# 会导致什么现象？

# 你的回答：会导致推理具有随机性，同一输入每次推理结果可能不同，且整体性能下降。

# 问题 2：
# Model A 和 Model B 的 Train Loss 谁更低？为什么？

# 你的回答：Model A 的 Train Loss 更低，因为 Model A 没有使用 Dropout，训练时所有神经元都在工作，模型更容易拟合训练数据。而 Model B 使用了 Dropout，在训练时随机关闭部分神经元，导致模型在训练数据上的表现不如 Model A。

# 问题 3：
# Dropout 的 p 值设得越大越好吗？如果 p=0.9 会发生什么？

# 你的回答：p 值设得过大并不一定更好。如果 p=0.9，意味着在训练时有 90% 的神经元会被随机关闭，这会导致模型几乎无法学习到有效的特征，训练过程会非常困难，模型性能可能大幅下降。
