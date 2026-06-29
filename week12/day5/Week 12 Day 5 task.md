# Week 12 · Day 5：自实现 Encoder 做蛋白质分类

## 任务设计

用 Day 4 的 `TransformerEncoder` 从头训练，在**蛋白质亚细胞定位**数据集上分类，和 Week 11 的 ESM-2 基准对比。

```
输入：氨基酸序列（字符串）
  ↓ tokenize（字符级）
  ↓ TransformerEncoder（自实现，从头训练）
  ↓ 取 [CLS] token 或做 mean pooling
  ↓ 线性分类头
输出：定位类别（0~9）
```

---

## 关键设计决策

### Pooling 策略：mean pooling vs [CLS] token

| | Mean Pooling | [CLS] Token |
|--|-------------|-------------|
| 做法 | 对所有位置取平均 | 在序列头部加特殊 token，取其输出 |
| 优点 | 简单，利用全序列信息 | ESM-2 的做法，可直接对比 |
| 缺点 | 对序列长度敏感 | 需要模型学会把信息聚合到 [CLS] |
| 今天用 | ✅ **mean pooling**（从头训练，[CLS] 需要更多数据才能收敛） | |

### 超参数设置

```python
d_model    = 128    # 比验证时大一倍，增加表达能力
num_heads  = 4
num_layers = 3
d_ff       = 512    # 4 × d_model
max_len    = 512
dropout    = 0.1
lr         = 1e-3
epochs     = 30
batch_size = 32
```

---

## 代码任务

新建 `week12/day5/train_classifier.py`：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))
from transformer_encoder import TransformerEncoder

# ── 1. 氨基酸词表 ─────────────────────────────────────────
# 20 种标准氨基酸 + PAD + UNK + CLS
AA_VOCAB = {aa: i+3 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB["<PAD>"] = 0
AA_VOCAB["<UNK>"] = 1
AA_VOCAB["<CLS>"] = 2
VOCAB_SIZE = len(AA_VOCAB)   # 23

def tokenize(seq: str, max_len: int = 512) -> list[int]:
    """字符级 tokenize，截断到 max_len"""
    tokens = [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in seq[:max_len]]
    return tokens

def pad_collate(batch):
    """动态 padding 到 batch 内最长序列"""
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    mask   = torch.zeros(len(seqs), 1, 1, max_len)  # (B,1,1,L) 广播用
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s)
        mask[i, 0, 0, :len(s)] = 1.0
    return padded, mask, torch.tensor(labels)


# ── 2. Dataset ────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len=512):
        self.data = [(tokenize(s, max_len), l)
                     for s, l in zip(sequences, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── 3. 分类模型 ───────────────────────────────────────────
class ProteinClassifier(nn.Module):
    def __init__(self, num_classes: int, **encoder_kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_kwargs)
        self.classifier = nn.Linear(encoder_kwargs["d_model"], num_classes)

    def forward(self, input_ids, mask=None):
        # (B, L, d_model)
        hidden, _ = self.encoder(input_ids, mask)

        # Mean pooling：对非 PAD 位置取平均
        if mask is not None:
            # mask: (B, 1, 1, L) → (B, L, 1)
            m = mask.squeeze(1).squeeze(1).unsqueeze(-1)  # (B, L, 1)
            hidden = (hidden * m).sum(dim=1) / m.sum(dim=1)
        else:
            hidden = hidden.mean(dim=1)   # (B, d_model)

        return self.classifier(hidden)   # (B, num_classes)


# ── 4. 训练循环 ───────────────────────────────────────────
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
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


# ── 5. 主程序 ─────────────────────────────────────────────
if __name__ == "__main__":
    # ---------- 数据加载（复用 Week 11 的数据集路径）----------
    import pandas as pd
    # ⚠️ 修改为你的实际路径
    df = pd.read_csv("data/deeploc_sequences.csv")
    sequences = df["sequence"].tolist()
    labels    = df["label"].tolist()

    # ---------- 划分数据集 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ---------- DataLoader ----------
    train_ds = ProteinDataset(X_train, y_train)
    val_ds   = ProteinDataset(X_val,   y_val)
    test_ds  = ProteinDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              collate_fn=pad_collate)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              collate_fn=pad_collate)

    # ---------- 模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    NUM_CLASSES = len(set(labels))
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

    # ---------- 训练 ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

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
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| 训练正常收敛 | Train Loss 持续下降 |
| 无 device 报错 | mask 和 input_ids 都 `.to(device)` |
| 测试准确率 | 记录实际数值，填入对比表 |
| 训练曲线图 | 保存 `training_curve.png` |

---

## 输出问题

**Q1**：代码里用了 `clip_grad_norm_(model.parameters(), 1.0)`，梯度裁剪的作用是什么？什么情况下梯度会"爆炸"？

**Q2**：`CosineAnnealingLR` 的学习率曲线是什么形状？和固定学习率、StepLR 相比有什么优势？

**Q3**：训练完成后，你的自实现模型和 ESM-2 的差距是多少？从模型设计角度分析：差距主要来自哪里？