# day2_eval_loop.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# ============================================================
# Part 1: 构造数据
# ============================================================

# 构造 100 个二维样本
X = torch.randn(100, 2)

# 标签规则：
# 如果两个特征之和 > 0，标签为 1，否则为 0
Y = (X.sum(dim=1) > 0).long()

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y dtype:", Y.dtype)

# ============================================================
# Part 2: Dataset 与 train/val split
# ============================================================

dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("train dataset length:", len(train_dataset))
print("val dataset length:", len(val_dataset))
print("number of train batches:", len(train_loader))
print("number of val batches:", len(val_loader))

# ============================================================
# Part 3: 定义模型
# ============================================================

# 输入维度：2
# 隐藏层维度：8
# 输出维度：2，因为是二分类，使用 CrossEntropyLoss
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

print("\nmodel:")
print(model)

# ============================================================
# Part 4: 定义 loss 和 optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ============================================================
# Part 5: 训练 + 验证
# ============================================================

num_epochs = 30

for epoch in range(num_epochs):
    # ----------------------------
    # Training phase
    # ----------------------------
    model.train()

    total_train_loss = 0.0

    for xb, yb in train_loader:
        # 1. 清空旧梯度
        # 你来写
        optimizer.zero_grad()
        # 2. 前向传播
        logits = model(xb)

        # 3. 计算 loss
        loss = criterion(logits, yb)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # 6. 累加 loss
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ----------------------------
    # Validation phase
    # ----------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            # 1. 前向传播
            logits = model(xb)

            # 2. 预测类别
            pred = logits.argmax(dim=1)

            # 3. 统计预测正确数量
            correct += (pred == yb).sum().item()

            # 4. 统计总样本数
            total += yb.size(0)

    val_acc = correct / total

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

# ============================================================
# Part 6: 最后查看一批验证集预测
# ============================================================

model.eval()

with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)

        print("\nOne validation batch prediction:")
        print("logits shape:", logits.shape)
        print("pred:", pred)
        print("true:", yb)
        print("correct:", pred == yb)
        break

# ============================================================
# Part 7: 回答问题
# ============================================================

# 问题 1：
# training loop 和 evaluation loop 最大区别是什么？

# 你的回答：training loop 会计算loss并更新模型参数，而 evaluation loop 只进行前向传播，不计算 loss，也不更新参数。
# evaluation loop 还会统计预测正确的数量来计算准确率等指标。
# Answer: eval loop可以有loss计算，但是不进行反向传播和参数更新

# 问题 2：
# 为什么验证阶段要用 torch.no_grad()？

# 你的回答：使用 torch.no_grad() 可以在验证阶段关闭梯度计算，从而节省显存和计算资源，提高推理速度。

# 问题 3：
# pred = logits.argmax(dim=1) 的作用是什么？

# 对每一行取最大值所在的类别编号，即预测的类别标签。
