import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day3'))
from esm2_embed import load_esm2
from protein_dataset import (
    load_localization_data, split_dataset,
    make_collate_fn, mean_pooling_with_mask,
    NUM_CLASSES, LOCALIZATION_CLASSES
)
from train_classifier import ProteinClassifier, compute_class_weights


# ── TODO 1：选择性解冻 ESM-2 ──────────────────────────────────
def unfreeze_last_n_layers(model, n: int = 2) -> None:
    """
    只解冻 ESM-2 最后 n 个 Transformer 层。
    其余层保持冻结。

    ESM-2 的层结构路径：
      model.encoder.layer[i]  ← 第 i 个 Transformer block

    步骤：
      1. 先冻结所有参数
      2. 再解冻 encoder.layer[-n:] 的参数
      3. 打印每层的 requires_grad 状态
    """
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 解冻最后 n 层
    total_layers = len(model.encoder.layer)
    for i in range(total_layers - n, total_layers):
        for p in model.encoder.layer[i].parameters():
            p.requires_grad = True

    # 打印状态
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"ESM-2 解冻最后 {n} 层")
    print(f"  可训练参数：{trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


# ── TODO 2：差分学习率优化器 ──────────────────────────────────
def make_optimizer(
    esm_model,
    classifier: nn.Module,
    esm_lr:  float = 1e-5,   # ESM-2 解冻层用小学习率
    head_lr: float = 1e-3,   # 分类头用大学习率
) -> torch.optim.Optimizer:
    """
    为不同部分设置不同学习率。

    为什么需要差分学习率？
      - ESM-2 已经预训练好，用大 LR 会破坏预训练的表示
      - 分类头是随机初始化的，需要大 LR 快速收敛

    返回一个 Adam 优化器，包含两个参数组：
      [
        {"params": ESM-2解冻层参数, "lr": esm_lr},
        {"params": 分类头参数,       "lr": head_lr},
      ]
    """
    esm_params  = [p for p in esm_model.parameters() if p.requires_grad]
    head_params = list(classifier.parameters())

    return torch.optim.Adam([
        {"params": esm_params,  "lr": esm_lr},
        {"params": head_params, "lr": head_lr},
    ])


# ── TODO 3：训练一个 epoch（ESM-2 参与梯度）──────────────────
def train_one_epoch_finetune(
    esm_model,
    classifier:  nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
) -> tuple[float, float]:
    """
    与 Day3 的 train_one_epoch 的关键区别：
      - esm_model.train()（不再是 eval）
      - 不用 torch.no_grad() 包裹 ESM-2 前向传播
      - 梯度可以流回 ESM-2 的解冻层
    """
    esm_model.train()
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        # ← 注意：这里不加 torch.no_grad()
        outputs    = esm_model(input_ids=input_ids,
                               attention_mask=attention_mask)
        embeddings = mean_pooling_with_mask(
            outputs.last_hidden_state, attention_mask)
        logits = classifier(embeddings)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


# ── TODO 4：验证（复用 Day3 的 evaluate）────────────────────
@torch.no_grad()
def evaluate(esm_model, classifier, loader, criterion, device):
    esm_model.eval()
    classifier.eval()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)
        outputs    = esm_model(input_ids=input_ids,
                               attention_mask=attention_mask)
        embeddings = mean_pooling_with_mask(
            outputs.last_hidden_state, attention_mask)
        logits = classifier(embeddings)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ── TODO 5：主程序 ────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 10
    BATCH  = 16

    # 1. 数据
    sequences, labels = load_localization_data()
    train_ds, val_ds, _ = split_dataset(sequences, labels, seed=42)

    # 2. 模型
    tokenizer, esm_model = load_esm2(device=device)
    unfreeze_last_n_layers(esm_model, n=2)

    classifier = ProteinClassifier().to(device)

    # 3. 从 Day3 的最佳权重初始化分类头（迁移学习）
    ckpt = os.path.join(
        os.path.dirname(__file__), '..', 'day3', 'best_classifier.pt')
    classifier.load_state_dict(torch.load(ckpt, map_location=device))
    print("分类头已从 Day3 权重初始化")

    # 4. 差分学习率优化器
    optimizer = make_optimizer(esm_model, classifier)

    # 5. 损失函数
    class_weights = compute_class_weights(train_ds.labels).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # 6. DataLoader
    collate_fn   = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH,
                              shuffle=False, collate_fn=collate_fn)

    # 7. 训练循环
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_finetune(
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
            torch.save({
                'esm_state': esm_model.state_dict(),
                'clf_state': classifier.state_dict(),
            }, "week11/day5/best_finetune.pt")
            print(f"  ✓ 保存最佳模型（val acc = {best_val_acc:.3f}）")

    print(f"\nFine-tuning 完成，最佳验证集准确率：{best_val_acc:.3f}")
    print(f"Day3 冻结训练最佳：0.634")
    print(f"Day5 Fine-tuning 最佳：{best_val_acc:.3f}")
    delta = best_val_acc - 0.634
    print(f"提升：{delta:+.3f}")


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
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 2489.95it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# ESM-2 解冻最后 2 层
#   可训练参数：2,465,920 / 7,511,801 (32.8%)
# 分类头已从 Day3 权重初始化
# Epoch  1 | train loss 1.0323 acc 0.649 | val loss 1.2422 acc 0.627
#   ✓ 保存最佳模型（val acc = 0.627）
# Epoch  2 | train loss 0.9597 acc 0.674 | val loss 1.2898 acc 0.638
#   ✓ 保存最佳模型（val acc = 0.638）
# Epoch  3 | train loss 0.9098 acc 0.678 | val loss 1.1971 acc 0.653
#   ✓ 保存最佳模型（val acc = 0.653）
# Epoch  4 | train loss 0.8634 acc 0.700 | val loss 1.2526 acc 0.675
#   ✓ 保存最佳模型（val acc = 0.675）
# Epoch  5 | train loss 0.8282 acc 0.708 | val loss 1.1876 acc 0.654
# Epoch  6 | train loss 0.7934 acc 0.708 | val loss 1.2116 acc 0.672
# Epoch  7 | train loss 0.7451 acc 0.722 | val loss 1.1928 acc 0.668
# Epoch  8 | train loss 0.7247 acc 0.724 | val loss 1.1888 acc 0.677
#   ✓ 保存最佳模型（val acc = 0.677）
# Epoch  9 | train loss 0.6864 acc 0.735 | val loss 1.2016 acc 0.674
# Epoch 10 | train loss 0.6598 acc 0.737 | val loss 1.2255 acc 0.672

# Fine-tuning 完成，最佳验证集准确率：0.677
# Day3 冻结训练最佳：0.634
# Day5 Fine-tuning 最佳：0.677
# 提升：+0.043


# Q1：为什么 Fine-tuning 时 ESM-2 用 1e-5 而分类头用 1e-3，相差 100 倍？如果两者都用 1e-3 会发生什么（这个现象叫什么）？
# ESM-2 是预训练模型，已经学到了丰富的蛋白质表示，如果使用过大的学习率（如 1e-3）来更新 ESM-2 的参数，可能会破坏其已有的知识，导致模型性能下降，
# 这种现象称为“灾难性遗忘”（catastrophic forgetting）。
# 因此，Fine-tuning 时对预训练模型使用较小的学习率（如 1e-5），
# 而对随机初始化的分类头使用较大的学习率（如 1e-3）以便快速收敛。

# Q2：为什么从 Day3 的权重初始化分类头，而不是随机初始化？这个策略叫什么？
# 从 Day3 的权重初始化分类头，而不是随机初始化，是为了利用之前训练好的模型参数，这样可以加速收敛并提高模型性能。
# 这种策略称为“迁移学习”（transfer learning），通过将预训练模型的知识迁移到新的任务中，可以减少训练时间并提高准确率。

# Q3：Fine-tuning 只解冻最后 2 层，而不是全部解冻，原因是什么？
# Fine-tuning 只解冻最后 2 层，而不是全部解冻，是为了在保持预训练模型已有知识的同时，适应新的任务。
# 如果全部解冻，可能会导致训练不稳定或过拟合，尤其是在数据量较小的情况下。
