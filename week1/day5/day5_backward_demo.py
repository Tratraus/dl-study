# day5_backward_demo.py

import torch
import torch.nn as nn

torch.manual_seed(42)

# ============================================================
# Part 1: 构造数据
# ============================================================

# 任务 1-1：
# 创建输入 x，shape 为 (4, 3)
# 表示 4 个样本，每个样本 3 个特征
x = torch.randn(4, 3)

# 任务 1-2：
# 创建目标 y，shape 为 (4, 1)
# 表示一个简单回归任务的目标值
y = torch.randn(4, 1)

print("x shape:", x.shape)
print("y shape:", y.shape)

# ============================================================
# Part 2: 定义模型和损失函数
# ============================================================

# 任务 2-1：
# 定义一个线性层：Linear(3, 1)
model = nn.Linear(3, 1)

# 任务 2-2：
# 定义均方误差损失 MSELoss
criterion = nn.MSELoss()

print("initial weight:", model.weight)
print("initial bias:", model.bias)

# ============================================================
# Part 3: 前向传播 + loss
# ============================================================

# 任务 3-1：
# 做前向传播，得到预测 pred
pred = model(x)

# 任务 3-2：
# 计算 loss
loss = criterion(pred, y)

print("pred shape:", pred.shape)
print("pred:", pred)
print("loss:", loss)

# ============================================================
# Part 4: 反向传播
# ============================================================

# 任务 4-1：
# 先确认 backward 前梯度是什么
print("before backward:")
print("weight.grad =", model.weight.grad)
print("bias.grad =", model.bias.grad)

# 任务 4-2：
# 调用 backward
# 让 PyTorch 自动计算梯度
loss.backward()

print("\nafter backward:")
print("weight.grad =", model.weight.grad)
print("bias.grad =", model.bias.grad)

# ============================================================
# Part 5: 观察梯度累加
# ============================================================

# 任务 5-1：
# 再做一次 forward 和 backward，但不要清空梯度
pred2 = model(x)
loss2 = criterion(pred2, y)
loss2.backward()

print("\nafter second backward (without zero_grad):")
print("weight.grad =", model.weight.grad)
print("bias.grad =", model.bias.grad)

# ============================================================
# Part 6: 手动清空梯度
# ============================================================

# 任务 6-1：
# 把梯度清空
model.zero_grad()

print("\nafter zero_grad:")
print("weight.grad =", model.weight.grad)
print("bias.grad =", model.bias.grad)

# ============================================================
# Part 7: 再做一次 backward，确认梯度重新计算
# ============================================================

pred3 = model(x)
loss3 = criterion(pred3, y)
loss3.backward()

print("\nafter third backward (after zero_grad):")
print("weight.grad =", model.weight.grad)
print("bias.grad =", model.bias.grad)

# ============================================================
# Part 8: 回答问题（写在注释里）
# ============================================================

# 问题 1：
# backward() 之后，梯度被存到哪里了？

# 你的回答：梯度被存储在每个参数的 .grad 属性中。

# 问题 2：
# 为什么第二次 backward 后梯度变大了？

# 你的回答：因为第一次的梯度没有被清空，第二次的梯度又被计算出来了，所以两次的梯度叠加在一起了。

# 问题 3：
# 为什么训练时通常每一步都要先 zero_grad()？

# 你的回答：避免上一轮的梯度影响当前轮的梯度计算，确保每轮的梯度都是当前轮的计算结果。
