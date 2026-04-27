# day1_train_val_split.py

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# ============================================================
# Part 1: 构造一个小型二维二分类数据集
# ============================================================

# 任务 1-1：
# 构造 20 个二维样本
# X shape 应该是 (20, 2)
X = torch.randn(20, 2)

# 任务 1-2：
# 构造二分类标签
# 规则：如果两个特征之和 > 0，标签为 1，否则为 0
# Y shape 应该是 (20,)
# Y dtype 应该是 torch.long
Y = (X.sum(dim=1) > 0).long()

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y dtype:", Y.dtype)
print("Y:", Y)

# ============================================================
# Part 2: 创建完整 Dataset
# ============================================================

# 任务 2-1：
# 使用 TensorDataset 封装 X 和 Y
dataset = TensorDataset(X, Y)

print("\nfull dataset length:", len(dataset))

# ============================================================
# Part 3: 划分 train / validation
# ============================================================

# 任务 3-1：
# 按照 80% / 20% 划分数据
# 总共 20 个样本：
# train_size = 16
# val_size = 4
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 任务 3-2：
# 使用 random_split 划分 dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("train dataset length:", len(train_dataset))
print("val dataset length:", len(val_dataset))

# ============================================================
# Part 4: 创建 train_loader 和 val_loader
# ============================================================

# 任务 4-1：
# train_loader:
# batch_size = 4
# shuffle = True
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 任务 4-2：
# val_loader:
# batch_size = 4
# shuffle = False
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("\nnumber of train batches:", len(train_loader))
print("number of val batches:", len(val_loader))

# ============================================================
# Part 5: 查看一个 train batch
# ============================================================

# 任务 5-1：
# 从 train_loader 中取出一个 batch
for xb, yb in train_loader:
    print("\none train batch:")
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("xb:", xb)
    print("yb:", yb)
    break

# ============================================================
# Part 6: 查看一个 validation batch
# ============================================================

# 任务 6-1：
# 从 val_loader 中取出一个 batch
for xb, yb in val_loader:
    print("\none validation batch:")
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("xb:", xb)
    print("yb:", yb)
    break

# ============================================================
# Part 7: 回答问题
# ============================================================

# 问题 1：
# train_dataset 和 val_dataset 分别用来做什么？

# 你的回答：train用于模型训练，而val用于模型验证，评估其在未见过的数据上的表现，以检测过拟合和调整超参数。

# 问题 2：
# 为什么 train_loader 通常 shuffle=True，而 val_loader 通常 shuffle=False？

# 你的回答：前者是为了打乱顺序，避免因顺序导致的过拟合，而后者为了复现性和一致性，保持数据顺序不变，便于评估模型性能。同时验证集不需要打乱，因为它不参与训练过程。

# Answer：训练集 shuffle=True 是为了打乱样本顺序，让每个 batch 更随机，减少数据排列顺序对训练过程的影响。验证集不参与训练，只用于评估，为了结果稳定和可复现，通常 shuffle=False。

# 问题 3：
# 验证集上的数据会不会用于 optimizer.step() 更新参数？为什么？

# 你的回答：不会，因为验证集数据不参与训练过程，不会被用来计算梯度和更新模型参数。验证集主要用于评估模型性能，而不是训练模型。
