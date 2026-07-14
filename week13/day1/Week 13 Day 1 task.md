收到，文件路径和函数名已确认：

| 原假设 | 实际命名 |
|--------|---------|
| `from protein_dataset import ProteinDataset, collate_fn` | `from protein_dataset import ProteinDataset, make_collate_fn` |
| `from esm2_utils import load_esm2` | `from esm2_embed import load_esm2` |
| `from protein_classifier import ProteinClassifier` | `from train_classifier import ProteinClassifier` |

路径结构也已记录：`week11/day1`、`week11/day2`、`week12/day5`。

---

# Week 13 Day 1：迁移学习策略框架 + 固定 Baseline

## 今日目标

1. **理论**：建立 Frozen / Fine-tune / Probing / LoRA / Adapter 的统一"参数更新范围"框架
2. **代码**：搭好三策略脚本骨架，跑通（loss 能下降即可）
3. **固定 Baseline**：自实现模型跑完整评估，记录 Test Acc + Macro F1，保存 checkpoint

---

## Part 1：理论框架（先读，5 分钟）

### 统一视角：按"可训练参数占比"排序

```
Probing < Frozen < Adapter ≈ LoRA < Full Fine-tune
  ~1%      ~5%      ~5-10%    ~1-5%      100%
```

| 策略 | 可训练部分 | 核心机制 | 适用数据量 |
|------|-----------|---------|-----------|
| **Probing** | 仅线性分类头（无 BN 更新） | 测试预训练特征的线性可分性 | 极少（<500条） |
| **Frozen** | 分类头（可含 BN） | backbone 完全冻结 | 少（<1000条） |
| **Adapter** | 插入层 + 分类头 | Frozen backbone 中插入小模块 | 少~中 |
| **LoRA** | 低秩矩阵 + 分类头 | 在权重矩阵旁加 $$\Delta W = BA$$（W8 已学） | 少~中 |
| **Fine-tune** | 全部参数（backbone 用小 lr） | 预训练知识 + 任务适配 | 中~多 |

> **关键直觉**：LoRA 不是独立技术，它是 Fine-tune 的参数高效变体——只更新权重矩阵的低秩近似，效果接近 Full Fine-tune，但可训练参数减少 10~100 倍。

---

## Part 2：代码任务

### 文件结构

```
week13/day1/
├── strategy_runner.py     ← 今天的主文件
└── baseline_checkpoint.pt ← Day 1 结束时保存
```

---

### 完整代码：`strategy_runner.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import sys, os

# ── 路径 ──────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from protein_dataset import ProteinDataset, make_collate_fn
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier

# ── 设备 ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════
# 1. 数据加载（复用 W11）
# ══════════════════════════════════════════════════════
def get_dataloaders(batch_size=32):
    """复用 W11 的数据 pipeline，返回 train/val/test loader"""
    # 根据你 W11 的实际用法调整参数
    train_dataset = ProteinDataset(split='train')
    val_dataset   = ProteinDataset(split='val')
    test_dataset  = ProteinDataset(split='test')

    collate_fn = make_collate_fn()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 三策略模型构建
# ══════════════════════════════════════════════════════
def build_model_frozen(num_classes):
    """
    策略 A：Frozen
    - backbone 权重全部冻结（requires_grad=False）
    - 只训练分类头
    """
    esm2_model, tokenizer = load_esm2()
    esm2_model = esm2_model.to(device)

    # 冻结 backbone 所有参数
    for param in esm2_model.parameters():
        param.requires_grad_(False)

    # 分类头（可训练）
    hidden_dim = esm2_model.config.hidden_size   # 调整为你的实际属性名
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)

    return esm2_model, classifier_head, tokenizer


def build_model_finetune(num_classes):
    """
    策略 B：Fine-tune
    - backbone + head 全部可训练
    - backbone 用更小的 lr（Day 2 深入实现，今天只搭骨架）
    """
    esm2_model, tokenizer = load_esm2()
    esm2_model = esm2_model.to(device)

    # 全部参数可训练（默认就是，无需额外操作）
    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)

    return esm2_model, classifier_head, tokenizer


def build_model_probing(num_classes):
    """
    策略 C：Probing（比 Frozen 更严格）
    - backbone 冻结，且设为 eval() 模式（BN/Dropout 不更新）
    - 只训练线性分类头
    """
    esm2_model, tokenizer = load_esm2()
    esm2_model = esm2_model.to(device)

    for param in esm2_model.parameters():
        param.requires_grad_(False)
    esm2_model.eval()   # ← Probing 与 Frozen 的关键区别

    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)

    return esm2_model, classifier_head, tokenizer


# ══════════════════════════════════════════════════════
# 3. 通用训练/评估函数（骨架）
# ══════════════════════════════════════════════════════
def get_esm2_embedding(esm2_model, tokenizer, sequences, strategy):
    """
    从 ESM-2 提取序列嵌入（mean pooling）
    strategy: 'frozen'/'probing' 时用 torch.no_grad()，'finetune' 时正常前向
    """
    inputs = tokenizer(sequences, return_tensors='pt',
                       padding=True, truncation=True,
                       max_length=512).to(device)
    if strategy in ('frozen', 'probing'):
        with torch.no_grad():
            outputs = esm2_model(**inputs)
    else:
        outputs = esm2_model(**inputs)

    # Mean pooling（忽略 padding token）
    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
    embeddings = (outputs.last_hidden_state * attention_mask).sum(1) \
                 / attention_mask.sum(1)
    return embeddings   # (batch, hidden_dim)


def train_one_epoch(esm2_model, classifier_head, optimizer,
                    criterion, train_loader, strategy):
    classifier_head.train()
    if strategy == 'finetune':
        esm2_model.train()
    else:
        esm2_model.eval()   # frozen/probing 时 backbone 保持 eval

    total_loss = 0
    for batch in train_loader:
        sequences = batch['sequence']          # 根据你的 dataset 实际字段名调整
        labels    = batch['label'].to(device)

        optimizer.zero_grad()
        embeddings = get_esm2_embedding(esm2_model, tokenizer,
                                        sequences, strategy)
        logits = classifier_head(embeddings)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def quick_evaluate(esm2_model, classifier_head, tokenizer,
                   loader, strategy):
    """快速评估：返回 Accuracy 和 Macro F1（用 sklearn 临时计算）"""
    esm2_model.eval()
    classifier_head.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            sequences = batch['sequence']
            labels    = batch['label'].to(device)
            embeddings = get_esm2_embedding(esm2_model, tokenizer,
                                            sequences, strategy)
            logits = classifier_head(embeddings)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1


# ══════════════════════════════════════════════════════
# 4. 骨架跑通验证（只跑 3 epoch，确认 loss 下降）
# ══════════════════════════════════════════════════════
def run_strategy_skeleton(strategy_name, esm2_model, classifier_head,
                          tokenizer, train_loader, val_loader,
                          num_classes, epochs=3):
    """
    骨架验证：只跑 3 epoch，确认三种策略均可正常训练
    Day 2~6 会在此基础上深化各策略
    """
    criterion = nn.CrossEntropyLoss()

    # 只优化可训练参数
    trainable_params = [p for p in list(esm2_model.parameters())
                        + list(classifier_head.parameters())
                        if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    print(f"\n{'='*50}")
    print(f"策略：{strategy_name}")
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count     = sum(p.numel() for p in list(esm2_model.parameters())
                          + list(classifier_head.parameters()))
    print(f"可训练参数：{trainable_count:,} / 总参数：{total_count:,}"
          f"（{100*trainable_count/total_count:.1f}%）")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(esm2_model, classifier_head, optimizer,
                                     criterion, train_loader, strategy_name)
        val_acc, val_f1 = quick_evaluate(esm2_model, classifier_head,
                                         tokenizer, val_loader, strategy_name)
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    return esm2_model, classifier_head


# ══════════════════════════════════════════════════════
# 5. Baseline 固定（自实现模型，完整训练）
# ══════════════════════════════════════════════════════
def run_baseline(train_loader, val_loader, test_loader,
                 num_classes, epochs=20):
    """
    Baseline：自实现 ProteinClassifier（W12）
    - 无数据增强
    - 标准 CrossEntropy
    - dropout=0.1
    - 无 Early Stopping
    - 固定 epoch 数
    """
    print("\n" + "="*50)
    print("【Baseline】自实现 ProteinClassifier")
    print("="*50)

    # 根据你 W12 的实际构造参数调整
    model = ProteinClassifier(num_classes=num_classes, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        total_loss = 0
        for batch in train_loader:
            # 根据你的 dataset 字段名调整
            x      = batch['input'].to(device)   # 或 'embedding' 等
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x      = batch['input'].to(device)
                labels = batch['label'].to(device)
                preds  = model(x).argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}"
                  f" | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    # ── 测试集最终评估 ──────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x      = batch['input'].to(device)
            labels = batch['label'].to(device)
            preds  = model(x).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n{'='*50}")
    print(f"【Baseline 最终结果】")
    print(f"  Test Acc    : {test_acc:.4f}")
    print(f"  Macro F1    : {test_f1:.4f}")
    print(f"{'='*50}")

    # ── 保存 checkpoint（Day 4、Day 7 复用）──────────
    save_path = os.path.join(os.path.dirname(__file__), 'baseline_checkpoint.pt')
    torch.save({
        'model_state_dict' : model.state_dict(),
        'test_acc'         : test_acc,
        'macro_f1'         : test_f1,
        'config': {
            'num_classes' : num_classes,
            'dropout'     : 0.1,
            'epochs'      : epochs,
            'lr'          : 1e-3,
            'augmentation': False,
            'loss'        : 'CrossEntropy',
            'early_stop'  : False,
        }
    }, save_path)
    print(f"Baseline checkpoint 已保存：{save_path}")

    return model, test_acc, test_f1


# ══════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10   # 根据你的数据集调整

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # ── Step 1：三策略骨架验证（各跑 3 epoch）──────
    strategies = {
        'frozen'   : build_model_frozen(NUM_CLASSES),
        'finetune' : build_model_finetune(NUM_CLASSES),
        'probing'  : build_model_probing(NUM_CLASSES),
    }

    for name, (esm2_model, classifier_head, tokenizer) in strategies.items():
        run_strategy_skeleton(name, esm2_model, classifier_head,
                              tokenizer, train_loader, val_loader,
                              NUM_CLASSES, epochs=3)

    # ── Step 2：固定 Baseline（自实现模型完整训练）──
    baseline_model, test_acc, test_f1 = run_baseline(
        train_loader, val_loader, test_loader,
        num_classes=NUM_CLASSES, epochs=20
    )

    print("\n【Day 1 完成】")
    print(f"  三策略骨架：✅ 均可正常训练")
    print(f"  Baseline Test Acc : {test_acc:.4f}")
    print(f"  Baseline Macro F1 : {test_f1:.4f}")
    print(f"  Checkpoint 已保存，供 Day 4 / Day 7 使用")
```

---

## 需要你根据实际情况调整的地方

运行前请确认以下 **4 处**，其余不用动：

| 位置 | 需确认的内容 | 常见值 |
|------|------------|--------|
| `ProteinDataset(split='train')` | 你的 dataset 构造参数是否一致 | 可能是传文件路径 |
| `batch['sequence']` / `batch['label']` | 你的 batch 字段名 | 看 W11 的 `make_collate_fn` 返回什么 |
| `batch['input']` | 自实现模型的输入字段名 | 可能是 `'embedding'` 或 `'x'` |
| `esm2_model.config.hidden_size` | ESM-2 隐层维度的属性名 | 通常是 `320`（esm2_t6）或 `480`（esm2_t12） |

---

## 今日完成标准

- [ ] 三策略均可正常训练，loss 下降（不要求收敛）
- [ ] 控制台打印出三策略的"可训练参数 / 总参数"比例
- [ ] Baseline Test Acc 和 Macro F1 已记录
- [ ] `baseline_checkpoint.pt` 已保存

---

## 输出问题（完成后回答）

1. **Frozen 和 Probing 有什么区别？** 什么情况下用 Probing 更合适？
2. **用"可训练参数比例"把五种策略排个序**，并说明各自适用的数据量场景。
3. **你固定的 Baseline 配置是什么？** Test Acc 和 Macro F1 分别是多少？