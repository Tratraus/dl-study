import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
from esm2_embed import load_esm2
from protein_dataset import (
    load_localization_data, split_dataset,
    make_collate_fn, mean_pooling_with_mask,
    NUM_CLASSES, LOCALIZATION_CLASSES
)


# ── TODO 1：分类器网络 ────────────────────────────────────────
class ProteinClassifier(nn.Module):
    """
    两层 MLP 分类器。

    结构：
      Linear(320 → 128) → ReLU → Dropout(0.3)
      → Linear(128 → NUM_CLASSES)

    注意：
      - 不要在最后加 Softmax（CrossEntropyLoss 内部已包含）
      - ESM-2 的参数在这里不更新（冻结）
    """
    def __init__(self, input_dim: int = 320,
                 hidden_dim: int = 128,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        # 在这里定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 320)
        # 返回 logits: (B, NUM_CLASSES)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ── TODO 2：类别权重（处理不均衡）────────────────────────────
def compute_class_weights(labels: list[int],
                          num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    计算每个类别的权重：weight[c] = total / (num_classes * count[c])

    这样稀有类别（如 Peroxisome 93 条）会得到更高的权重。
    """
    from collections import Counter
    counts = Counter(labels)
    total  = len(labels)
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * counts[c])
    return weights


# ── TODO 3：单个 epoch 的训练 ─────────────────────────────────
def train_one_epoch(
    esm_model,
    classifier:  nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch，返回 (avg_loss, accuracy)。

    关键步骤：
      1. esm_model.eval() + torch.no_grad() 提取嵌入（冻结 ESM-2）
      2. classifier.train()
      3. 对每个 batch：
         a. 提取 ESM-2 嵌入
         b. mean_pooling_with_mask
         c. classifier 前向传播
         d. 计算 loss
         e. 反向传播 + optimizer.step()
    """
    esm_model.eval()
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pooling_with_mask(outputs.last_hidden_state, attention_mask)

        logits = classifier(embeddings)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── TODO 4：验证 ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    esm_model,
    classifier: nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
) -> tuple[float, float]:
    """
    在验证集上评估，返回 (avg_loss, accuracy)。
    """
    esm_model.eval()
    classifier.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pooling_with_mask(outputs.last_hidden_state, attention_mask)

        logits = classifier(embeddings)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── TODO 5：主训练循环 ────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS    = 10
    BATCH     = 16
    LR        = 1e-3

    # 1. 数据
    sequences, labels = load_localization_data()
    train_ds, val_ds, test_ds = split_dataset(sequences, labels)

    # 2. ESM-2（冻结）
    tokenizer, esm_model = load_esm2(device=device)
    esm_model.eval()
    for p in esm_model.parameters():
        p.requires_grad = False

    # 3. 分类器
    classifier = ProteinClassifier().to(device)
    print(f"分类器参数量：{sum(p.numel() for p in classifier.parameters()):,}")

    # 4. 损失函数（带类别权重）
    class_weights = compute_class_weights(
        train_ds.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. 优化器
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    # 6. DataLoader
    collate_fn   = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH,
                              shuffle=False, collate_fn=collate_fn)

    # 7. 训练循环
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            esm_model, classifier, train_loader,
            optimizer, criterion, device)
        val_loss, val_acc = evaluate(
            esm_model, classifier, val_loader,
            criterion, device)

        print(f"Epoch {epoch:2d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), "week11/day3/best_classifier.pt")
            print(f"  ✓ 保存最佳模型（val acc = {best_val_acc:.3f}）")

    print(f"\n训练完成，最佳验证集准确率：{best_val_acc:.3f}")


if __name__ == "__main__":
    main()

# 输出
# 加载 ProtST-SubcellularLocalization 数据集...
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# 过滤后数据集大小：8388
#   类别 0（Cell membrane            ）： 800 条
#   类别 1（Cytoplasm                ）：1635 条
#   类别 2（Endoplasmic reticulum    ）： 516 条
#   类别 3（Golgi apparatus          ）： 214 条
#   类别 4（Lysosome/Vacuole         ）： 192 条
#   类别 5（Mitochondria             ）： 906 条
#   类别 6（Nucleus                  ）：2424 条
#   类别 7（Peroxisome               ）：  93 条
#   类别 8（Plastid                  ）： 453 条
#   类别 9（Extracellular            ）：1155 条
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1714.18it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.bias              | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# pooler.dense.bias         | MISSING    |
# pooler.dense.weight       | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 分类器参数量：42,378
# Epoch  1 | train loss 1.8422 acc 0.461 | val loss 1.5582 acc 0.567
#   ✓ 保存最佳模型（val acc = 0.567）
# Epoch  2 | train loss 1.4395 acc 0.557 | val loss 1.4455 acc 0.599
#   ✓ 保存最佳模型（val acc = 0.599）
# Epoch  3 | train loss 1.3346 acc 0.581 | val loss 1.3766 acc 0.608
#   ✓ 保存最佳模型（val acc = 0.608）
# Epoch  4 | train loss 1.2512 acc 0.604 | val loss 1.3458 acc 0.622
#   ✓ 保存最佳模型（val acc = 0.622）
# Epoch  5 | train loss 1.2002 acc 0.613 | val loss 1.3018 acc 0.615
# Epoch  6 | train loss 1.1502 acc 0.622 | val loss 1.2774 acc 0.610
# Epoch  7 | train loss 1.1112 acc 0.638 | val loss 1.2654 acc 0.618
# Epoch  8 | train loss 1.0910 acc 0.644 | val loss 1.2489 acc 0.634
#   ✓ 保存最佳模型（val acc = 0.634）
# Epoch  9 | train loss 1.0655 acc 0.649 | val loss 1.2506 acc 0.614
# Epoch 10 | train loss 1.0235 acc 0.656 | val loss 1.2573 acc 0.632

# Q1：为什么训练时 esm_model.eval() + torch.no_grad()，而 classifier.train()？如果把 ESM-2 也设为 train() 会发生什么？
# 答：ESM-2 是预训练模型，我们在这里冻结它的参数，不希望它在训练过程中更新。
# 使用 eval() 和 torch.no_grad() 可以确保 ESM-2 的参数不参与梯度计算和更新，从而节省内存和计算资源。
# 如果把 ESM-2 设为 train()，虽然它的参数仍然被冻结，但会增加不必要的计算开销，因为会计算梯度但不更新参数，这可能导致训练效率降低。

# Q2：compute_class_weights 中，Peroxisome（93条）和 Nucleus（2424条）的权重比是多少？这对训练有什么影响？
# 答：Peroxisome 的权重约为 2424 / 93 ≈ 26.05 倍于 Nucleus 的权重。
# 这意味着在计算损失时，模型会更重视 Peroxisome 类别的错误，从而帮助模型更好地学习这个稀有类别，避免被多数类别（如 Nucleus）主导训练过程。

# Q3：如果去掉 CrossEntropyLoss 的 weight 参数，模型可能会学到什么"捷径"？
# 答：如果去掉 weight 参数，模型可能会倾向于预测多数类别（如 Nucleus），因为这样可以在整体上获得较高的准确率。
# 这会导致模型在稀有类别（如 Peroxisome）上的性能非常差，因为模型没有足够的激励去正确分类这些样本，从而形成一个不平衡的分类器。