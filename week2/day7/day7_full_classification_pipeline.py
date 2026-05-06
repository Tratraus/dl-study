# day7_full_classification_pipeline.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)


# ============================================================
# Function 1: 创建 train / val / test dataloaders
# ============================================================

def create_dataloaders(
    n_samples=300,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
):
    """
    创建非线性二分类数据集：
    radius = x1^2 + x2^2
    radius > 1.0 -> class 1
    radius <= 1.0 -> class 0

    返回：
    train_loader, val_loader, test_loader, X, Y
    """

    # 1. 构造 X
    X = torch.randn(n_samples, 2)

    # 2. 构造 radius
    radius = X.pow(2).sum(dim=1)

    # 3. 构造 Y
    Y = (radius > 1.0).long()

    # 4. 封装 TensorDataset
    dataset = TensorDataset(X, Y)

    # 5. 计算 train / val / test size
    # 注意 test_size = 总长度 - train_size - val_size
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 6. random_split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 7. 创建 train_loader, val_loader, test_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X, Y


# ============================================================
# Function 2: 创建模型
# ============================================================

def create_model():
    """
    创建一个稍微深一点的 MLP。
    输入维度: 2
    输出类别数: 2
    """

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    return model


# ============================================================
# Function 3: 训练一个 epoch
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    训练模型一个 epoch，返回 avg_train_loss。
    """

    # 你来写完整 train loop
    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    return avg_train_loss


# ============================================================
# Function 4: 评估模型
# ============================================================

def evaluate(model, data_loader, criterion):
    """
    评估模型，返回 avg_loss, accuracy。
    """

    # 你来写完整 evaluation loop
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    return avg_loss, accuracy


# ============================================================
# Function 5: 预测函数
# ============================================================

def predict(model, x):
    """
    返回 logits, probs, pred。
    """

    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

    return logits, probs, pred


# ============================================================
# Function 6: 主函数
# ============================================================

def main():
    # -----------------------------
    # 1. 创建数据
    # -----------------------------
    train_loader, val_loader, test_loader, X, Y = create_dataloaders(
        n_samples=300,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15
    )

    print("Dataset created.")
    print("Total samples:", len(X))
    print("Class counts:", torch.bincount(Y))

    # -----------------------------
    # 2. 创建模型 / loss / optimizer
    # -----------------------------
    model = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # -----------------------------
    # 3. 准备保存路径
    # -----------------------------
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "day7_best_model.pt")

    # -----------------------------
    # 4. 训练循环
    # -----------------------------
    num_epochs = 50
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练一轮
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc: {best_val_acc:.4f}")

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    print("\nTraining finished.")
    print("Best Val Acc:", best_val_acc)

    # -----------------------------
    # 5. 加载 best model
    # -----------------------------
    best_model = create_model()

    # 你来加载 best_model_path
    state_dict = torch.load(best_model_path)
    best_model.load_state_dict(state_dict)

    # -----------------------------
    # 6. 在 test set 上评估
    # -----------------------------
    test_loss, test_acc = evaluate(best_model, test_loader, criterion)

    print("\n========== Test Evaluation ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    # -----------------------------
    # 7. 对前 10 个样本做预测
    # -----------------------------
    x_demo = X[:10]
    y_demo = Y[:10]

    logits, probs, pred = predict(best_model, x_demo)

    print("\n========== Demo Prediction ==========")

    print("x_demo:")
    print(x_demo)

    print("\ny_demo:")
    print(y_demo)

    print("\nlogits:")
    print(logits)

    print("\nprobs:")
    print(probs)

    print("\npred:")
    print(pred)

    print("\nCorrect?")
    print(pred == y_demo)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# Questions
# ============================================================

# 问题 1：
# 今天这个任务和前几天相比，完整 pipeline 多了哪些部分？

# 你的回答：首先是数据集创建任务不同，
# 其次是训练集不同于之前，使用的是 train/val/test 划分

# 问题 2：
# 为什么这次要划分 train / val / test 三个集合？

# 你的回答：train集合用于训练模型，而val集合用于在训练过程中评估模型性能并选择最佳模型，test集合则用于在训练完成后评估最终模型的泛化能力。划分三个集合可以帮助我们更好地评估模型的性能，避免过拟合，并确保模型在未见过的数据上表现良好。

# 问题 3：
# validation set 和 test set 的区别是什么？

# 你的回答：val是用于评估模型的性能并选择最佳模型的，而test仅用于评估模型泛化能力

# 问题 4：
# 如果 test accuracy 明显低于 validation accuracy，可能说明什么？

# 你的回答：模型泛化能力不足，有过拟合的可能，或者 test 集合的分布与 train/val 集合不同。
