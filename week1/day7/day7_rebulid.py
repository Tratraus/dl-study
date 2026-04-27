import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ============================================================
# Part 1: 数据创建
# ============================================================

X = torch.randn(10, 2)
Y = (X[:, 0] * X[:, 1] > 0).long()

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y dtype:", Y.dtype)

# ============================================================
# Part 2: Dataset 和 DataLoader
# ============================================================

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ============================================================
# Part 3: 模型、损失函数、优化器
# ============================================================

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)

# ============================================================
# Part 4: 训练循环
# ============================================================

num_epochs = 100

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_X, batch_Y in loader:
        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_Y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Avg Loss: {avg_loss:.4f}")

# ============================================================
# Part 5: 训练后预测
# ============================================================

with torch.no_grad():
    logits = model(X)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == Y).float().mean()

print("\nlogits:")
print(logits)
print("predictions:", predictions)
print("true labels:", Y)
print(f"Accuracy: {accuracy.item() * 100:.2f}%")
