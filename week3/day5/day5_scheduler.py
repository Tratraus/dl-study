# day5_scheduler.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1~3: 数据准备（直接复用）
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
# Part 4: 模型（复用 Day 3 的简单版本，避免 Dropout 干扰观察）
# ============================================================

def build_model():
    return nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )

# ============================================================
# Part 5: 训练与评估函数
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
# Part 6: 实验函数（你来补全 scheduler 的调用位置）
# ============================================================

def run_experiment(model_name, scheduler_type, num_epochs=100):
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 根据 scheduler_type 创建不同的 scheduler
    if scheduler_type == "none":
        scheduler = None

    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )

    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    print(f"\n{'='*65}")
    print(f"  {model_name}")
    print(f"{'='*65}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
    print("-" * 65)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # ---- 你来补全：在正确的位置调用 scheduler.step() ----
        # 提示：
        # - scheduler_type == "none"：不需要调用
        # - scheduler_type == "step"：调用 scheduler.step()
        # - scheduler_type == "plateau"：调用 scheduler.step(val_loss)
        if scheduler is not None:
            if scheduler_type == "step":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if epoch % 10 == 0:
            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>10.6f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return history

# ============================================================
# Part 7: 运行三组实验
# ============================================================

history_none    = run_experiment("Experiment 1: No Scheduler",          "none")
history_step    = run_experiment("Experiment 2: StepLR(step=20, γ=0.5)","step")
history_plateau = run_experiment("Experiment 3: ReduceLROnPlateau",     "plateau")

# ============================================================
# Part 8: 对比最终 Val Loss
# ============================================================

print("\n========== Final Val Loss Comparison ==========")
print(f"No Scheduler:        {history_none['val_loss'][-1]:.6f}")
print(f"StepLR:              {history_step['val_loss'][-1]:.6f}")
print(f"ReduceLROnPlateau:   {history_plateau['val_loss'][-1]:.6f}")

print("\n========== LR at Final Epoch ==========")
print(f"No Scheduler:        {history_none['lr'][-1]:.6f}")
print(f"StepLR:              {history_step['lr'][-1]:.6f}")
print(f"ReduceLROnPlateau:   {history_plateau['lr'][-1]:.6f}")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# StepLR 和 ReduceLROnPlateau 最本质的区别是什么？
# 各自适合什么场景？

# 你的回答：StepLR是固定的衰减系数，而ReduceLROnPlateau是根据验证集性能动态调整学习率。
# StepLR适合训练过程比较稳定的情况，而ReduceLROnPlateau适合训练过程中可能出现震荡或者需要根据验证集表现调整学习率的情况。

# 问题 2：
# 如果把 scheduler.step() 写在了 batch 循环里（而不是 epoch 循环里），
# 会发生什么？

# 你的回答：会导致学习率在每个 batch 后都进行调整，这通常会导致学习率过快地衰减，训练过程可能会变得非常慢，甚至无法收敛。

# 问题 3：
# ReduceLROnPlateau 的 patience=10 是什么意思？
# 如果把 patience 设得很小（比如 patience=1），有什么风险？

# 你的回答：如果 patience 设得很小，学习率可能会过于频繁地下降，导致模型在尚未充分学习的情况下就降低学习率，
# 从而影响训练效果，可能导致训练不稳定或收敛速度变慢。
