import torch

# ============================================================
# Part 1: 创建 Tensor
# ============================================================

# 任务 1-1：创建一个 shape 为 (3, 4) 的全零 Tensor
t1 = torch.zeros(3,4)

# 任务 1-2：创建一个 shape 为 (3, 4) 的随机 Tensor（标准正态分布）
t2 = torch.randn(3,4)

# 任务 1-3：创建一个 1 维 Tensor，内容是 [1.0, 2.0, 3.0, 4.0]
t3 = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 打印三个 Tensor 的 shape
print("t1 shape:", t1.shape)
print("t2 shape:", t2.shape)
print("t3 shape:", t3.shape)


# ============================================================
# Part 2: reshape 与 view
# ============================================================

# 任务 2-1：把 t2（shape 3x4）reshape 成 (12,)
t2_flat = t2.reshape(12,)
# or t2_flat = t2.reshape(-1)

# 任务 2-2：把 t2_flat reshape 成 (2, 6)
t2_reshaped = t2_flat.reshape(2, 6)

print("t2_flat shape:", t2_flat.shape)
print("t2_reshaped shape:", t2_reshaped.shape)


# ============================================================
# Part 3: 基本统计操作
# ============================================================

# 任务 3-1：计算 t2 所有元素的均值
mean_all = t2.mean()

# 任务 3-2：计算 t2 每一列的均值（沿 dim=0 方向）
mean_col = t2.mean(dim=0)

# 任务 3-3：计算 t2 每一行的均值（沿 dim=1 方向）
mean_row = t2.mean(dim=1)

print("mean_all:", mean_all)
print("mean_col shape:", mean_col.shape, "values:", mean_col)
print("mean_row shape:", mean_row.shape, "values:", mean_row)


# ============================================================
# Part 4: 索引
# ============================================================

# 任务 4-1：取出 t2 的第 0 行
row0 = t2[0]

# 任务 4-2：取出 t2 的第 2 列
col2 = t2[:, 2]

# 任务 4-3：取出 t2 的第 1 行第 3 列的那个元素
elem = t2[1, 3]

print("row0:", row0)
print("col2:", col2)
print("elem:", elem)


# ============================================================
# Part 5: 矩阵乘法与 shape 变化
# ============================================================

import torch.nn as nn

# 任务 5-1：创建一个 batch，shape 为 (8, 4)，模拟 8 个样本、每个样本 4 个特征
x_batch = torch.randn(8, 4)

# 任务 5-2：创建一个 nn.Linear(4, 2)，模拟把 4 维输入映射到 2 维输出
linear = nn.Linear(4, 2)

# 任务 5-3：把 x_batch 送入 linear，得到输出
out = linear(x_batch)

print("x_batch shape:", x_batch.shape)
print("out shape:", out.shape)

# 任务 5-4：回答这个问题（写在注释里）
# 为什么 out 的 shape 是 (8, 2)，而不是 (8, 4) 或其他？
# 你的回答：
# nn.Linear(4,2)的处理是针对每个样本的4个特征进行的，并映射到2维的输出了