# day3_logits_softmax_pred.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# ============================================================
# Part 1: 构造数据
# ============================================================

X = torch.randn(100, 2)
Y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ============================================================
# Part 2: 定义模型、loss、optimizer
# ============================================================

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ============================================================
# Part 3: 简单训练模型
# ============================================================

num_epochs = 20

for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:
        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播，得到 logits
        logits = model(xb)

        # 3. 计算 loss
        loss = criterion(logits, yb)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}")

# ============================================================
# Part 4: 取一个 validation batch，观察 logits / softmax / pred
# ============================================================

model.eval()

with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb)

        # softmax 后得到概率
        probs = torch.softmax(logits, dim=1)

        # 从 logits 直接得到预测类别
        pred_from_logits = torch.argmax(logits, dim=1)

        # 从 probs 得到预测类别
        pred_from_probs = torch.argmax(probs, dim=1)

        print("\n=== One validation batch ===")
        print("xb shape:", xb.shape)
        print("logits shape:", logits.shape)
        print("probs shape:", probs.shape)
        print("yb shape:", yb.shape)

        print("\nlogits:")
        print(logits)

        print("\nprobs:")
        print(probs)

        print("\nsum of probs for each sample:")
        print(probs.sum(dim=1))

        print("\npred_from_logits:")
        print(pred_from_logits)

        print("\npred_from_probs:")
        print(pred_from_probs)

        print("\ntrue labels:")
        print(yb)

        print("\nAre predictions from logits and probs the same?")
        print(pred_from_logits == pred_from_probs)

        print("\nCorrect prediction?")
        print(pred_from_logits == yb)

        break

# ============================================================
# Part 5: 手动构造 logits，观察 softmax
# ============================================================

manual_logits = torch.tensor([
    [0.2, 1.5],
    [2.1, -0.3],
    [0.0, 0.0],
    [-1.0, 1.0],
    [5.0, 1.0]
])

manual_probs = torch.softmax(manual_logits, dim=1)
manual_pred_logits = torch.argmax(manual_logits, dim=1)
manual_pred_probs = torch.argmax(manual_probs, dim=1)

print("\n=== Manual logits experiment ===")
print("manual logits:")
print(manual_logits)

print("\nmanual probs:")
print(manual_probs)

print("\nsum of manual probs:")
print(manual_probs.sum(dim=1))

print("\npred from manual logits:")
print(manual_pred_logits)

print("\npred from manual probs:")
print(manual_pred_probs)

print("\nAre manual predictions the same?")
print(manual_pred_logits == manual_pred_probs)

# ============================================================
# Part 6: 回答问题
# ============================================================

# 问题 1：
# logits 是概率吗？为什么？

# 你的回答：不是，其本质上是模型对标签的评分，“模型更倾向于这个对象是这个标签”

# 问题 2：
# softmax 的作用是什么？

# 你的回答：将 logits 转换为概率分布，使得每个类别的概率在 0 到 1 之间，并且所有类别的概率和为 1

# 问题 3：
# 为什么 pred_from_logits 和 pred_from_probs 通常是一样的？

# 你的回答：因为sofrmax仅是将logits的值进行了转换缩放，并不会改变原有的排序，所以最大值所在的索引通常不变

# 问题 4：
# 使用 nn.CrossEntropyLoss() 时，应该把 logits 还是 softmax 后的 probs 传进去？

# 你的回答：应该传入 logits，因为 nn.CrossEntropyLoss() 内部会对 logits 进行 softmax 操作，所以不需要手动对 logits 进行 softmax
