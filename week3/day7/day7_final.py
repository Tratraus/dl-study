# day7_final.py
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
from sklearn.datasets import load_wine

# ============================================================
# Part 1~3: 数据准备
# ===========================================================
wine = load_wine()
X_np = wine.data.astype(np.float32)   # shape: (178, 13)
Y_np = wine.target.astype(np.int64)   # shape: (178,)

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_np, Y_np, test_size=0.15, random_state=42, stratify=Y_np
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.1765, random_state=42, stratify=Y_temp
)
# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class WineDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

train_loader = DataLoader(WineDataset(X_train, Y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(WineDataset(X_val,   Y_val),   batch_size=16, shuffle=False)
test_loader  = DataLoader(WineDataset(X_test,  Y_test),  batch_size=16, shuffle=False)
# ============================================================
# Part 4: 模型定义
# ===========================================================
def build_model():
    return nn.Sequential(
        nn.Linear(13, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 3)
    )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (preds.argmax(dim=1) == yb).sum().item()
        total_samples += xb.size(0)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)

            total_loss += loss.item() * xb.size(0)
            total_correct += (preds.argmax(dim=1) == yb).sum().item()
            total_samples += xb.size(0)
    return total_loss / total_samples, total_correct / total_samples


# ============================================================
# Part 5: 训练并收集 history
# ============================================================
NUM_EPOCHS = 100
model = build_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
    )


history = {'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []}

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# ============================================================
# Part 6: 可视化 history
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
plt.savefig('week3/day7/training_curves.png', dpi=150, bbox_inches='tight')
print("图片已保存至 week3/day7/training_curves.png")

# ============================================================
# Part 7: 混淆矩阵
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 收集测试集的所有预测结果
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# 分类报告
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
      target_names=wine.target_names))

# 可视化混淆矩阵（保存为第四张图）
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names,
            ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
ax_cm.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('week3/day7/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("混淆矩阵已保存")
