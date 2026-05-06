# day4_train_eval_mode.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# ============================================================
# Part A: Dropout 在 train/eval 下的行为实验
# ============================================================

print("========== Part A: Dropout behavior ==========")

dropout_model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4, 2)
)

x_demo = torch.ones(1, 4)

print("\nInput x_demo:")
print(x_demo)

# ------------------------------------------------------------
# A1. train mode 下多次 forward
# ------------------------------------------------------------

dropout_model.train()

print("\nTrain mode outputs:")
for i in range(5):
    out = dropout_model(x_demo)
    print(f"Output {i}:", out)

# ------------------------------------------------------------
# A2. eval mode 下多次 forward
# ------------------------------------------------------------

dropout_model.eval()

print("\nEval mode outputs:")
for i in range(5):
    out = dropout_model(x_demo)
    print(f"Output {i}:", out)

# ============================================================
# Part B: 在分类模型中使用 train/eval
# ============================================================

print("\n========== Part B: Training with train/eval ==========")

# 构造数据
X = torch.randn(100, 2)
Y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义一个带 Dropout 的 MLP
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(16, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30

for epoch in range(num_epochs):
    # --------------------------------------------------------
    # Training phase
    # --------------------------------------------------------
    model.train()

    total_train_loss = 0.0

    for xb, yb in train_loader:
        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        logits = model(xb)

        # 3. 计算 loss
        loss = criterion(logits, yb)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --------------------------------------------------------
    # Validation phase
    # --------------------------------------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            pred = logits.argmax(dim=1)

            correct += (pred == yb).sum().item()
            total += yb.size(0)

    val_acc = correct / total

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

# ============================================================
# Part C: 观察同一个样本在 train/eval 下预测是否稳定
# ============================================================

print("\n========== Part C: Prediction stability ==========")

x_one = X[:1]

print("\nOne sample:")
print(x_one)

# C1. train mode 下，同一个样本多次预测
model.train()

print("\nPredictions in train mode:")
for i in range(5):
    logits = model(x_one)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    print(f"Run {i}: logits={logits}, probs={probs}, pred={pred}")

# C2. eval mode 下，同一个样本多次预测
model.eval()

print("\nPredictions in eval mode:")
with torch.no_grad():
    for i in range(5):
        logits = model(x_one)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        print(f"Run {i}: logits={logits}, probs={probs}, pred={pred}")

# ============================================================
# Part D: 回答问题
# ============================================================

# 问题 1：
# model.train() 和 model.eval() 的作用是什么？

# 你的回答：分别用于开始模型训练的步骤和模型评估的步骤

# 问题 2：
# model.eval() 和 torch.no_grad() 是一回事吗？为什么？

# 你的回答：不一样，model.eval本质是开启模型的评估模式
# 而torch.no_grad是作用只是禁止梯度计算，节省内存和计算资源。

# 问题 3：
# 为什么带 Dropout 的模型在 train mode 下，同一个输入多次 forward 可能输出不同？

# 你的回答：因为在训练模式下，Dropout 会随机丢弃一部分神经元的输出，从而导致同一个输入在不同的 forward 过程中可能得到不同的输出。

# 问题 4：
# 评估/验证阶段通常应该怎么写？请写出最小代码结构。

# 你的回答：
# model.eval()
#   with torch.no_grad():
#     for xb, yb in val_loader:
#         logits = model(xb)
#         pred = logits.argmax(dim=1)