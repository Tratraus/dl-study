import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Week 12 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))
from transformer_encoder import TransformerEncoder

# Week 11 模块（复用数据集）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
from protein_dataset import (
    load_localization_data, split_dataset, make_collate_fn,
    NUM_CLASSES, LOCALIZATION_CLASSES
)
from esm2_embed import load_esm2


# ── 1. 分类模型 ───────────────────────────────────────────
class ProteinClassifier(nn.Module):
    def __init__(self, num_classes: int, **encoder_kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_kwargs)
        self.classifier = nn.Linear(encoder_kwargs["d_model"], num_classes)

    def forward(self, input_ids, mask=None):
        # mask: Week 11 collate_fn 输出 (B, L)，1=真实 0=padding

        # 1. 生成 attention mask（True=屏蔽，用于 masked_fill）
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = ~mask.unsqueeze(1).unsqueeze(1).bool()  # (B,L)→(B,1,1,L)
            else:
                attn_mask = ~mask.bool()

        # 2. Encoder 前向
        hidden, _ = self.encoder(input_ids, attn_mask)  # (B, L, d_model)

        # 3. Mean pooling（用原始 mask，1=真实 0=padding）
        if mask is not None:
            if mask.dim() == 2:
                m = mask.unsqueeze(-1).float()  # (B, L) → (B, L, 1)
            else:
                m = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()  # (B,1,1,L)→(B,L,1)
            hidden = (hidden * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            hidden = hidden.mean(dim=1)

        return self.classifier(hidden)   # (B, num_classes)


# ── 2. 训练循环 ───────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask      = mask.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask      = mask.to(device)
        labels    = labels.to(device)

        logits = model(input_ids, mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


# ── 3. 类别权重（处理不均衡）────────────────────────────────
def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """weight[c] = total / (num_classes * count[c])"""
    from collections import Counter
    counts = Counter(labels)
    total  = len(labels)
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * counts.get(c, 1))
    return weights


# ── 4. 主程序 ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- 数据加载（复用 Week 11）----------
    print("加载 ProtST-SubcellularLocalization 数据集...")
    sequences, labels = load_localization_data()

    train_ds, val_ds, test_ds = split_dataset(sequences, labels)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---------- Tokenizer（复用 Week 11 的 ESM-2 tokenizer）----------
    print("\n加载 ESM-2 tokenizer...")
    tokenizer, _ = load_esm2(device='cpu')  # 只要 tokenizer，模型加载后丢弃
    collate_fn = make_collate_fn(tokenizer)

    VOCAB_SIZE = len(tokenizer)
    print(f"词表大小: {VOCAB_SIZE}")

    # ---------- DataLoader ----------
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              collate_fn=collate_fn)

    # ---------- 模型 ----------
    model = ProteinClassifier(
        num_classes = NUM_CLASSES,
        vocab_size  = VOCAB_SIZE,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 3,
        d_ff        = 512,
        max_len     = 512,
        dropout     = 0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # ---------- 类别权重 ----------
    class_weights = compute_class_weights(train_ds.labels, NUM_CLASSES).to(device)

    # ---------- 训练 ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_acc = 0
    for epoch in range(1, 31):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1 = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "week12/day5/best_model.pt")

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
                  f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f} F1 {vl_f1:.4f}")

    # ---------- 测试 ----------
    model.load_state_dict(torch.load("week12/day5/best_model.pt"))
    _, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*50}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")
    print(f"{'='*50}")
    print(f"\n对比基准：")
    print(f"  ESM-2 Frozen    : 63.4%")
    print(f"  ESM-2 Fine-tuned: 67.7%")
    print(f"  自实现 Encoder  : {test_acc*100:.1f}%")

    # ---------- 训练曲线 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.axhline(0.634, color="orange", linestyle="--",
                label="ESM-2 Frozen (63.4%)")
    ax2.axhline(0.677, color="red",    linestyle="--",
                label="ESM-2 Fine-tuned (67.7%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("week12/day5/training_curve.png", dpi=150)
    plt.show()
    print("训练曲线已保存")

# 记录
# 修改了 multihead_attention.py 的 forward，
# 增加了 attn_mask 的处理逻辑，
# 避免在 scaled_dot_product_attention 里直接使用 mask 导致的维度不匹配问题。

# 输出
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
# Train: 5871 | Val: 1258 | Test: 1259

# 加载 ESM-2 tokenizer...
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1955.61it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 词表大小: 33
# 总参数量: 600,586
# Epoch 05 | Train Loss 1.5970 Acc 0.4824 | Val Loss 1.5533 Acc 0.5016 F1 0.4033
# Epoch 10 | Train Loss 1.3712 Acc 0.5435 | Val Loss 1.4897 Acc 0.5612 F1 0.4622
# Epoch 15 | Train Loss 1.2176 Acc 0.5829 | Val Loss 1.5111 Acc 0.5771 F1 0.4582
# Epoch 20 | Train Loss 1.0151 Acc 0.6137 | Val Loss 1.5015 Acc 0.5819 F1 0.4998
# Epoch 25 | Train Loss 0.8152 Acc 0.6558 | Val Loss 1.5993 Acc 0.5811 F1 0.5146
# Epoch 30 | Train Loss 0.7377 Acc 0.6794 | Val Loss 1.6046 Acc 0.5715 F1 0.4990

# ==================================================
# Test Accuracy : 0.6100
# Test Macro F1 : 0.4853
# ==================================================

# 对比基准：
#   ESM-2 Frozen    : 63.4%
#   ESM-2 Fine-tuned: 67.7%
#   自实现 Encoder  : 61.0%


# Q1：代码里用了 clip_grad_norm_(model.parameters(), 1.0)，梯度裁剪的作用是什么？什么情况下梯度会"爆炸"？
# 梯度裁剪的作用是限制梯度的最大范数，防止梯度过大导致模型参数更新过大，从而引发训练不稳定或发散的问题。
# 梯度爆炸通常发生在深层神经网络中，尤其是在循环神经网络（RNN）或深度前馈网络中，
# 当网络层数较多时，反向传播过程中梯度可能会累积变得非常大，导致权重更新过大，从而使模型无法收敛。

# Q2：CosineAnnealingLR 的学习率曲线是什么形状？和固定学习率、StepLR 相比有什么优势？
# CosineAnnealingLR 的学习率曲线呈现余弦下降的形状，从初始学习率逐渐下降到最小学习率，然后再周期性地回升。

# Q3：训练完成后，你的自实现模型和 ESM-2 的差距是多少？从模型设计角度分析：差距主要来自哪里？
# 自实现模型的测试准确率为 61.0%，而 ESM-2 Fine-tuned 的准确率为 67.7%，差距约为 6.7%。
# 差距主要来自以下几个方面：
# 1. 模型规模：ESM-2 是一个大规模预训练模型，具有更多的参数和更深的网络结构，而自实现模型相对较小，可能无法捕捉到复杂的蛋白质序列特征。
# 2. 预训练：ESM-2 在大规模蛋白质数据上进行了预训练，
#    使其能够学习到丰富的蛋白质序列表示，而自实现模型是从头开始训练，缺乏预训练的知识迁移。
# 3. 特征表示能力：ESM-2 可能使用了更复杂的特征提取和注意力机制，而自实现模型可能在特征表示能力上有所欠缺，
#    导致在分类任务中表现不如 ESM-2。