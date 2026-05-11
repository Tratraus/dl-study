# day6_visualization.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
# Part 4: 模型 + 训练评估函数
# ============================================================

def build_model():
    return nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 3)
    )

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
# Part 5: 训练并收集 history
# ============================================================

NUM_EPOCHS = 100

model     = build_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "lr": []
}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(optimizer.param_groups[0]['lr'])

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Training complete. Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ============================================================
# Part 6: 可视化（你来补全）
# ============================================================

epochs = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'Training Curves (Test Acc: {test_acc:.4f})', fontsize=14)

# ------ 图 1：Loss 曲线 ------
ax1 = axes[0]
# 你来写：
# 1. 画 train_loss（蓝色实线，label='Train Loss'）
# 2. 画 val_loss（橙色实线，label='Val Loss'）
# 3. 设置 xlabel='Epoch', ylabel='Loss', title='Loss Curve'
# 4. 加 legend 和 grid
ax1.plot(epochs, history["train_loss"], label='Train Loss', color='blue')
ax1.plot(epochs, history["val_loss"], label='Val Loss', color='orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid()

# ------ 图 2：Accuracy 曲线 ------
ax2 = axes[1]
# 你来写：
# 1. 画 train_acc（蓝色实线，label='Train Acc'）
# 2. 画 val_acc（橙色实线，label='Val Acc'）
# 3. 设置 xlabel, ylabel='Accuracy', title='Accuracy Curve'
# 4. 设置 y 轴范围：ax2.set_ylim([0.7, 1.05])
# 5. 加 legend 和 grid
ax2.plot(epochs, history["train_acc"], label='Train Acc', color='blue')
ax2.plot(epochs, history["val_acc"], label='Val Acc', color='orange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curve')
ax2.set_ylim([0.7, 1.05])
ax2.legend()
ax2.grid()

# ------ 图 3：学习率曲线 ------
ax3 = axes[2]
# 你来写：
# 1. 画 lr（绿色实线，label='Learning Rate'）
# 2. 设置 xlabel, ylabel='Learning Rate', title='LR Schedule'
# 3. 加 legend 和 grid
# 4. （可选）加 ax3.set_yscale('log') 让 y 轴用对数坐标，更清晰
ax3.plot(epochs, history["lr"], label='Learning Rate', color='green')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('LR Schedule')
ax3.legend()
ax3.grid()
ax3.set_yscale('log')

plt.tight_layout()
plt.savefig('week3/day6/training_curves.png', dpi=150, bbox_inches='tight')
print("图片已保存至 week3/day6/training_curves.png")

# ============================================================
# Questions
# ============================================================

# 问题 1：
# matplotlib.use('Agg') 是什么意思？如果不写这行，在没有图形界面的
# 终端环境下运行会发生什么？

# 你的回答：matplotlib.use('Agg') 是用来指定使用 'Agg' 后端，这个后端不需要图形界面，适合在没有显示器的服务器或终端环境下生成图片。
# 如果不写这行，在没有图形界面的终端环境下运行会报错，因为默认后端需要图形界面支持。

# 问题 2：
# ax.set_yscale('log') 把 y 轴改成对数坐标。
# 什么情况下对数坐标比线性坐标更适合展示 loss 曲线？

# 你的回答：当 loss 曲线的值变化范围很大，尤其是在训练初期 loss 下降非常快，而后期下降缓慢时，使用对数坐标可以更清晰地展示整个训练过程的变化趋势。

# 问题 3：
# 训练结束后，你观察到 Train Loss 和 Val Loss 的曲线形态是什么样的？
# 有没有出现过拟合的信号？

# 你的回答：（运行后根据图片回答）
# 趋于稳定，有部分loss曲线出现震荡，可能存在轻微过拟合的信号。