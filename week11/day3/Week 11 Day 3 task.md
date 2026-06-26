# Week 11 Day 3：分类器训练

数据管道已经就绪，今天开始训练真正的分类器。

## 任务目标

用 ESM-2 提取的嵌入向量，训练一个**亚细胞定位分类器**：

```
蛋白质序列 → ESM-2 → mean pooling → (320,) → MLP → 10类
```

---

## 代码任务

新建 `week11/day3/train_classifier.py`：

```python
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
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 320)
        # 返回 logits: (B, NUM_CLASSES)
        ...


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
    ...


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
    ...


# ── TODO 5：主训练循环 ────────────────────────────────────────
def main():
    device = torch.device('cpu')
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
            torch.save(classifier.state_dict(), "best_classifier.pt")
            print(f"  ✓ 保存最佳模型（val acc = {best_val_acc:.3f}）")

    print(f"\n训练完成，最佳验证集准确率：{best_val_acc:.3f}")


if __name__ == "__main__":
    main()
```

---

## 完成标准

1. `ProteinClassifier` 参数量约为 **41,994**（320×128 + 128 + 128×10 + 10）
2. 10 个 epoch 后验证集准确率 > **50%**（随机猜测为 10%）
3. `best_classifier.pt` 成功保存
4. 训练 loss 呈**下降趋势**

---

## 输出问题

**Q1**：为什么训练时 `esm_model.eval()` + `torch.no_grad()`，而 `classifier.train()`？如果把 ESM-2 也设为 `train()` 会发生什么？

**Q2**：`compute_class_weights` 中，Peroxisome（93条）和 Nucleus（2424条）的权重比是多少？这对训练有什么影响？

**Q3**：如果去掉 `CrossEntropyLoss` 的 `weight` 参数，模型可能会学到什么"捷径"？

准备好后提交代码和训练日志。