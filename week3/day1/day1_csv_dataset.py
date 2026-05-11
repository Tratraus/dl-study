# day1_csv_dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# Part 1: 读取 CSV 文件
# ============================================================

# 注意路径：假设你从 week3/ 目录运行脚本
# 如果从 day1/ 目录运行，路径需要改成 "../data/iris.csv"
csv_path = "week3/data/iris.csv"

df = pd.read_csv(csv_path)

print("========== DataFrame Info ==========")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Label distribution:\n", df["species"].value_counts().sort_index())

# ============================================================
# Part 2: 提取特征和标签，转换为 Tensor
# ============================================================

# 特征列：前 4 列
# 你来写
X_np = df.iloc[:,:4].values

# 标签列：species 列
# 你来写
Y_np = df["species"].values

# 转换为 Tensor
# 注意：
# X 是浮点特征，用 torch.float32
# Y 是整数标签，用 torch.long
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.long)


print("\n========== Tensor Info ==========")
print("X shape:", X.shape)
print("X dtype:", X.dtype)
print("Y shape:", Y.shape)
print("Y dtype:", Y.dtype)

# ============================================================
# Part 3: 自定义 Dataset 类
# ============================================================

class IrisDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================
# Part 4: 创建 Dataset 实例，验证
# ============================================================

dataset = IrisDataset(X, Y)

print("\n========== Dataset Info ==========")
print("Dataset length:", len(dataset))

# 取第 0 个样本
x0, y0 = dataset[0]
print("Sample 0 - x:", x0)
print("Sample 0 - y:", y0)

# 取第 5 个样本
x5, y5 = dataset[5]
print("Sample 5 - x:", x5)
print("Sample 5 - y:", y5)

# ============================================================
# Part 5: 划分 train / val / test
# ============================================================

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print("\n========== Split Info ==========")
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))

# ============================================================
# Part 6: 创建 DataLoader
# ============================================================

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("\n========== DataLoader Info ==========")
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

# ============================================================
# Part 7: 验证一个 batch 的内容
# ============================================================

for xb, yb in train_loader:
    print("\n========== First Batch ==========")
    print("xb shape:", xb.shape)
    print("xb dtype:", xb.dtype)
    print("yb shape:", yb.shape)
    print("yb dtype:", yb.dtype)
    print("yb values:", yb)
    break

# ============================================================
# Questions
# ============================================================

# 问题 1：
# 自定义 Dataset 类需要实现哪三个方法？分别是什么作用？

# 你的回答：需要实现 __init__、__len__ 和 __getitem__ 三个方法。
# __init__：初始化方法，接收数据并进行必要的预处理。
# __len__：返回数据集的样本数量。
# __getitem__：根据索引返回对应的特征和标签。

# 问题 2：
# 为什么 X 要用 torch.float32，Y 要用 torch.long？

# 你的回答：因为X是连续的数值特征，使用 torch.float32 可以更好地表示小数和进行计算，也是PyTorch的默认设置；
# 而Y是离散的类别标签，使用 torch.long 可以表示整数类型，更适合分类任务。

# 问题 3：
# TensorDataset 和自定义 Dataset 有什么区别？什么时候需要自定义？

# 你的回答：TensorDataset 是一个简单的封装，适用于特征和标签已经是 Tensor 的情况；
# 自定义 Dataset 可以处理更复杂的数据预处理逻辑，例如从 CSV 文件读取数据、数据增强等。
