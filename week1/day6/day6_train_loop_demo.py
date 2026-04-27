# day6_train_loop_demo.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
torch.manual_seed(42)

# ============================================================
# Part 1: 构造一个简单的二分类数据集
# ============================================================

# 任务 1-1：
# 手动构造 8 个二维样本
# 前 4 个靠近 (0,0)，标签为 0
# 后 4 个靠近 (1,1)，标签为 1
x = torch.tensor([
    [0.0, 0.1],
    [0.2, -0.1],
    [-0.1, 0.0],
    [0.1, 0.2],
    [1.0, 0.9],
    [0.8, 1.1],
    [1.2, 1.0],
    [0.9, 0.8]
], dtype=torch.float32)

# 任务 1-2：
# 创建对应标签 y，shape 为 (8,)
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

print("x shape:", x.shape)
print("y shape:", y.shape)

# ============================================================
# Part 2: Dataset 和 DataLoader
# ============================================================

dataset = TensorDataset(x, y)

# 任务 2-1：
# 创建 DataLoader：
# batch_size = 4
# shuffle = True
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ============================================================
# Part 3: 定义模型、损失函数、优化器
# ============================================================

# 任务 3-1：
# 定义一个最小 MLP：
# Linear(2, 4) -> ReLU -> Linear(4, 2)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

# 任务 3-2：
# 定义交叉熵损失 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 任务 3-3：
# 定义 SGD 优化器，学习率 lr=0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

print(model)

# ============================================================
# Part 4: 训练循环
# ============================================================

# 任务 4-1：
# 训练 20 个 epoch
num_epochs = 20

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for xb, yb in loader:

        # 任务 4-2：
        # 按标准训练顺序补全下面几步
        # 1. 清空梯度
        # 2. 前向传播
        # 3. 计算 loss
        # 4. backward
        # 5. optimizer.step()
        # 你来写
        optimizer.zero_grad() # 1. 清空梯度
        logits = model(xb) # 2. 前向传播
        loss = criterion(logits, yb) # 3. 计算 loss
        loss.backward() # 4. backward
        optimizer.step() # 5. optimizer.step()

        epoch_loss += loss.item()

    print(f"epoch {epoch+1:02d}, loss = {epoch_loss:.4f}")

# ============================================================
# Part 5: 训练后看预测结果
# ============================================================

# 任务 5-1：
# 在训练完成后，重新把整个 x 输入模型
# 得到输出 logits
with torch.no_grad():
    logits = model(x)
    pred_class = torch.argmax(logits, dim=1)

print("\nlogits:")
print(logits)
print("pred_class:", pred_class)
print("true y    :", y)

# ============================================================
# Part 6: 回答问题（写在注释里）
# ============================================================

# 问题 1：
# optimizer.step() 做了什么？

# 你的回答：optimizer.step() 根据之前计算的梯度更新模型参数，以使得损失函数的值下降。

# 问题 2：
# 为什么训练循环的顺序通常是：
# zero_grad -> forward -> loss -> backward -> step ？

# 你的回答：因为我们需要先清空之前的梯度（zero_grad），
# 然后进行前向传播（forward）计算预测值，接着计算损失（loss），
# 再进行反向传播（backward）计算梯度，最后更新参数（step）。
# 这个顺序确保了每次迭代都使用正确的梯度来更新模型参数。

# 问题 3：
# epoch 和 batch 有什么关系？

# 你的回答：batch是数据被分成的小块，每个batch包含一定数量的样本。
# epoch是指整个训练数据被完整地送入模型一次的过程。
# 一个epoch可以包含多个batch，具体数量取决于数据集的大小和batch_size的设置。
