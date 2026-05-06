# day6_functional_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)


# ============================================================
# Function 1: 创建 dataloaders
# ============================================================

def create_dataloaders(
    n_samples=200,
    batch_size=16,
    train_ratio=0.8
):
    """
    创建一个简单二分类数据集，并返回 train_loader, val_loader, X, Y。
    """

    X = torch.randn(n_samples, 2)
    Y = (X.sum(dim=1) > 0).long()

    dataset = TensorDataset(X, Y)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, X, Y


# ============================================================
# Function 2: 创建模型
# ============================================================

def create_model():
    """
    创建一个简单 MLP 分类模型。
    输入维度: 2
    隐藏层: 16
    输出类别数: 2
    """

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    return model


# ============================================================
# Function 3: 训练一个 epoch
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    训练模型一个 epoch，返回平均训练 loss。
    """

    # 1. 切换到训练模式
    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:
        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. forward，得到 logits
        logits = model(xb)

        # 4. 计算 loss
        loss = criterion(logits, yb)

        # 5. backward
        loss.backward()

        # 6. update
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss


# ============================================================
# Function 4: 评估模型
# ============================================================

def evaluate(model, data_loader, criterion):
    """
    在 data_loader 上评估模型，返回平均 loss 和 accuracy。
    既可以用于 validation，也可以用于 test。
    """

    # 1. 切换到评估模式
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # 2. 关闭梯度
    with torch.no_grad():
        for xb, yb in data_loader:
            # 3. forward
            logits = model(xb)

            # 4. loss
            loss = criterion(logits, yb)

            # 5. prediction
            pred = logits.argmax(dim=1)

            # 6. 累加 loss
            total_loss += loss.item()

            # 7. 累加 correct 和 total
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    avg_loss = total_loss / len(data_loader)
    acc = correct / total

    return avg_loss, acc


# ============================================================
# Function 5: 预测函数
# ============================================================

def predict(model, x):
    """
    对输入 x 做预测，返回 logits, probs, pred。
    """

    # 1. 切换到评估模式
    model.eval()

    with torch.no_grad():
        # 2. forward
        logits = model(x)

        # 3. softmax 得到概率
        probs = torch.softmax(logits, dim=1)

        # 4. argmax 得到预测类别
        pred = logits.argmax(dim=1)

    return logits, probs, pred


# ============================================================
# Function 6: 主函数
# ============================================================

def main():
    # -----------------------------
    # 1. 创建数据
    # -----------------------------
    train_loader, val_loader, X, Y = create_dataloaders(
        n_samples=200,
        batch_size=16,
        train_ratio=0.8
    )

    # -----------------------------
    # 2. 创建模型、loss、optimizer
    # -----------------------------
    model = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # -----------------------------
    # 3. 准备保存路径
    # -----------------------------
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "day6_best_model.pt")

    # -----------------------------
    # 4. 训练循环
    # -----------------------------
    num_epochs = 30
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存 best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch}, val_acc={val_acc:.4f}")

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
    state_dict = torch.load(best_model_path)
    best_model.load_state_dict(state_dict)

    # -----------------------------
    # 6. 使用 predict 函数预测前 5 个样本
    # -----------------------------
    x_test = X[:5]
    y_test = Y[:5]

    logits, probs, pred = predict(best_model, x_test)

    print("\n========== Prediction with loaded best model ==========")
    print("x_test:")
    print(x_test)

    print("\ny_test:")
    print(y_test)

    print("\nlogits:")
    print(logits)

    print("\nprobs:")
    print(probs)

    print("\npred:")
    print(pred)

    print("\nCorrect?")
    print(pred == y_test)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# Questions
# ============================================================

# 问题 1：
# 为什么要把训练代码拆成函数？

# 你的回答：便于维护和调试
# Answer：把训练代码拆成函数可以让代码结构更清晰，便于维护、调试和复用。不同功能如训练、评估、预测可以分别检查，也方便以后替换数据或模型。

# 问题 2：
# train_one_epoch() 里面为什么要写 model.train()？

# 你的回答：启用训练模式
# Answer：model.train() 用于把模型切换到训练模式，使 Dropout、BatchNorm 等层按照训练阶段的方式工作，保证训练行为正确。

# 问题 3：
# evaluate() 里面为什么要写 model.eval() 和 torch.no_grad()？

# 你的回答：评估模式需要启用 model.eval() 来关闭训练行为，torch.no_grad() 用于关闭梯度计算，节省内存和计算资源。
# Answer：补充：因为验证阶段不更新参数，不应该执行 backward 和 optimizer.step。

# 问题 4：
# predict() 函数通常应该返回哪些内容？为什么？

# 你的回答：通常应该返回 logits, probs, pred。logits 是模型的原始输出，probs 是经过 softmax 转换后的概率分布，
# pred 是最终的预测类别。这些信息可以帮助我们更好地理解模型的预测结果。
