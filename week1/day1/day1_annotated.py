# 包导入
import torch
import torch.nn as nn
import torch.optim as optim

# 种子设定
torch.manual_seed(42)

# 数据创建
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).long()

# 模型定义
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器，使用的是Adam（不过并不清楚是什么）
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20): # 20次迭代
    logits = model(X) # 选择模型
    loss = criterion(logits, y) # 选择顺势函数


    optimizer.zero_grad()
    # 反向传播loss
    loss.backward()
    # 更新参数
    optimizer.step()

    # 输出训练信息
    if (epoch + 1) % 5 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        print(f"Epoch {epoch+1:02d} | Loss = {loss.item():.4f} | Acc = {acc.item():.4f}")
