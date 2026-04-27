# day4_dataloader_demo.py

import torch
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# Part 1: 构造数据
# ============================================================

# 任务 1-1：
# 创建输入 x，shape 为 (10, 4)
# 表示 10 个样本，每个样本 4 个特征
x = torch.randn(10,4)

# 任务 1-2：
# 创建标签 y，shape 为 (10,)
# 标签内容你可以手写一个二分类序列，比如 0 和 1 交替
y = torch.tensor([0, 0, 1, 0, 1, 0, 0, 0, 1, 1])

print("x shape:", x.shape)
print("y shape:", y.shape)

# ============================================================
# Part 2: 创建 Dataset
# ============================================================

# 任务 2-1：
# 用 TensorDataset 把 x 和 y 绑在一起
dataset = TensorDataset(x, y)

# 任务 2-2：
# 打印 dataset 的长度
print("dataset length:", len(dataset))

# 任务 2-3：
# 打印第 0 个样本和第 0 个标签
sample_x, sample_y = dataset[0]
print("sample 0 x:", sample_x)
print("sample 0 y:", sample_y)

# ============================================================
# Part 3: 创建 DataLoader
# ============================================================

# 任务 3-1：
# 创建 DataLoader，要求：
# - batch_size = 4
# - shuffle = False
loader = DataLoader(dataset, batch_size = 4, shuffle = False)

# ============================================================
# Part 4: 遍历 DataLoader
# ============================================================

# 任务 4-1：
# 遍历 loader，打印每个 batch 的 shape 和内容
for batch_idx, (xb, yb) in enumerate(loader):
    print(f"batch {batch_idx}")
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("xb:", xb)
    print("yb:", yb)
    print("-" * 30)

# ============================================================
# Part 5: 对比 shuffle=False 和 shuffle=True
# ============================================================

# 任务 5-1：
# 再创建一个 DataLoader，要求：
# - batch_size = 4
# - shuffle = True
loader_shuffle = DataLoader(dataset, batch_size = 4, shuffle = True)

print("\nCompare shuffle=True")
for batch_idx, (xb, yb) in enumerate(loader_shuffle):
    print(f"batch {batch_idx}")
    print("yb:", yb)

# ============================================================
# Part 6: 回答问题（写在注释里）
# ============================================================

# 问题 1：
# DataLoader 做了什么？

# 你的回答：将dataset按照指定的batch_size进行分批，并且可以选择是否打乱数据顺序。

# 问题 2：
# 为什么最后一个 batch 可能不是完整的 batch_size？

# 你的回答：因为数据集的大小可能不是 batch_size 的整数倍，最后一个 batch 的样本数可能少于 batch_size。

# 问题 3：
# 为什么训练时通常用 shuffle=True？

# 你的回答：避免模型过拟合到数据的特定顺序，提高训练效果。
