# day2_normalization.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 读取数据
# ============================================================

df = pd.read_csv("week3/data/iris.csv")

X_np = df.iloc[:, :4].values.astype(np.float32)
Y_np = df["species"].values.astype(np.int64)

print("========== Raw Data ==========")
print("X mean (per feature):", X_np.mean(axis=0).round(3))
print("X std  (per feature):", X_np.std(axis=0).round(3))
print("X min  (per feature):", X_np.min(axis=0).round(3))
print("X max  (per feature):", X_np.max(axis=0).round(3))

# ============================================================
# Part 2: 划分 train / val / test（用 sklearn 的 train_test_split）
# ============================================================

# 先划分出 test（15%），再从剩余中划分 val（约 15%）
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_np, Y_np, test_size=0.15, random_state=42, stratify=Y_np
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.176, random_state=42, stratify=Y_temp
    # 0.176 ≈ 0.15 / 0.85，使得 val 约占总数据 15%
)

print("\n========== Split Info ==========")
print("Train size:", len(X_train))
print("Val size:  ", len(X_val))
print("Test size: ", len(X_test))

# ============================================================
# Part 3: 标准化
# ============================================================

scaler = StandardScaler()

# 你来写：
# 1. 在训练集上 fit_transform
# 2. 在验证集和测试集上只 transform
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 验证标准化结果
print("\n========== After Scaling ==========")
print("Train X mean (per feature):", X_train_scaled.mean(axis=0).round(3))
print("Train X std  (per feature):", X_train_scaled.std(axis=0).round(3))
print("Val   X mean (per feature):", X_val_scaled.mean(axis=0).round(3))
print("Val   X std  (per feature):", X_val_scaled.std(axis=0).round(3))

# ============================================================
# Part 4: 转换为 Tensor，封装 Dataset
# ============================================================

class IrisDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = IrisDataset(X_train_scaled, Y_train)
val_dataset   = IrisDataset(X_val_scaled,   Y_val)
test_dataset  = IrisDataset(X_test_scaled,  Y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# ============================================================
# Part 5: 对比标准化前后的第一个样本
# ============================================================

print("\n========== Sample Comparison ==========")
print("Raw      sample 0:", X_np[0])
print("Scaled   sample 0:", X_train_scaled[0] if 0 < len(X_train) else "sample 0 is in val/test")

# 打印第一个 batch
for xb, yb in train_loader:
    print("\n========== First Batch ==========")
    print("xb shape:", xb.shape)
    print("xb mean: ", xb.mean(dim=0).round(decimals=3))
    print("xb std:  ", xb.std(dim=0).round(decimals=3))
    print("yb:", yb)
    break

# ============================================================
# Questions
# ============================================================

# 问题 1：
# 为什么 scaler 只能在训练集上 fit，不能在验证集/测试集上 fit？

# 你的回答：避免模型提前学到验证集和测试集的信息

# 问题 2：
# 标准化之后，训练集的 mean 应该接近 0，std 应该接近 1。
# 但验证集的 mean 和 std 不一定精确是 0 和 1，为什么？

# 你的回答：因为验证集的数据分布可能与训练集略有不同，标准化是基于训练集的统计量进行的，
# 所以验证集的 mean 和 std 不一定是 0 和 1。

# 问题 3：
# 如果推理时来了一条新的样本数据，应该怎么处理？

# 你的回答：不太清楚
# Answer：需要使用训练时保存的同一个scaler来处理这条新样本数据，
# 确保它与训练数据使用相同的标准化参数进行转换。