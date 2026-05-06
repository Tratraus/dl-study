
# day5_save_load_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# ============================================================
# Part 1: 构造数据
# ============================================================

X = torch.randn(200, 2)
Y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ============================================================
# Part 2: 定义模型工厂函数
# ============================================================

def create_model():
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    return model

model = create_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ============================================================
# Part 3: 查看 state_dict
# ============================================================

print("========== State dict keys ==========")

for name, param in model.state_dict().items():
    print(name, param.shape)

# ============================================================
# Part 4: 训练模型，同时保存 best model
# ============================================================

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

last_model_path = os.path.join(save_dir, "last_model.pt")
best_model_path = os.path.join(save_dir, "best_model.pt")

num_epochs = 30
best_val_acc = 0.0

for epoch in range(num_epochs):
    # -----------------------------
    # Training
    # -----------------------------
    model.train()

    total_train_loss = 0.0

    for xb, yb in train_loader:
        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. forward
        logits = model(xb)

        # 3. loss
        loss = criterion(logits, yb)

        # 4. backward
        loss.backward()

        # 5. update
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            pred = logits.argmax(dim=1)

            correct += (pred == yb).sum().item()
            total += yb.size(0)

    val_acc = correct / total

    # 保存当前 epoch 的最后模型
    torch.save(model.state_dict(), last_model_path)

    # 如果当前验证集准确率更好，保存 best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch}, val_acc={val_acc:.4f}")

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

print("\nTraining finished.")
print("Best Val Acc:", best_val_acc)
print("Last model path:", last_model_path)
print("Best model path:", best_model_path)

# ============================================================
# Part 5: 保存前模型的预测
# ============================================================

model.eval()

x_test = X[:5]

with torch.no_grad():
    logits_before = model(x_test)
    probs_before = torch.softmax(logits_before, dim=1)
    pred_before = logits_before.argmax(dim=1)

print("\n========== Prediction before loading ==========")
print("logits_before:")
print(logits_before)
print("probs_before:")
print(probs_before)
print("pred_before:")
print(pred_before)

# ============================================================
# Part 6: 加载 last_model 到新模型
# ============================================================

loaded_model = create_model()

state_dict = torch.load(last_model_path)

loaded_model.load_state_dict(state_dict)

loaded_model.eval()

with torch.no_grad():
    logits_after = loaded_model(x_test)
    probs_after = torch.softmax(logits_after, dim=1)
    pred_after = logits_after.argmax(dim=1)

print("\n========== Prediction after loading last_model ==========")
print("logits_after:")
print(logits_after)
print("probs_after:")
print(probs_after)
print("pred_after:")
print(pred_after)

print("\nAre logits_before and logits_after close?")
print(torch.allclose(logits_before, logits_after))

print("\nAre pred_before and pred_after the same?")
print(torch.equal(pred_before, pred_after))

# ============================================================
# Part 7: 加载 best_model 到新模型，并评估验证集
# ============================================================

best_model = create_model()

best_state_dict = torch.load(best_model_path)

best_model.load_state_dict(best_state_dict)

best_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for xb, yb in val_loader:
        logits = best_model(xb)
        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += yb.size(0)

loaded_best_val_acc = correct / total

print("\n========== Loaded best model evaluation ==========")
print("Loaded Best Val Acc:", loaded_best_val_acc)

# ============================================================
# Part 8: 回答问题
# ============================================================

# 问题 1：
# state_dict 是什么？

# 你的回答：是一个存储了模型参数信息的对象

# 问题 2：
# 为什么通常推荐保存 model.state_dict() 而不是整个 model？

# 你的回答：因为 state_dict 只包含模型的参数信息，保存和加载更灵活，且不依赖于模型的类定义。

# 问题 3：
# 加载 state_dict 时，为什么需要先重新创建同样结构的模型？

# 你的回答：因为 state_dict 只包含参数信息，不包含模型的结构信息，所以需要先创建一个相同结构的模型，然后再加载参数。

# 问题 4：
# 为什么要保存 best_model，而不仅仅是 last_model？

# 你的回答：因为 last_model 只是最后一个 epoch 的模型，可能不是性能最好的模型。而 best_model 是在验证集上表现最好的模型，通常用于实际部署或进一步评估。