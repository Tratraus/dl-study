# Week 13 Day 2：差异化学习率

## 今日目标

1. **理解** `param_groups` 的结构，以及为什么 Fine-tune 需要差异化学习率
2. **实现** `get_param_groups()` 函数，支持 backbone/head 独立设置 lr
3. **实验** 对比三组 lr 比例，画出训练曲线，找到最优比例

---

## Part 1：理论（5 分钟）

### 为什么需要差异化学习率？

Fine-tune 时，backbone 和 head 的"起点"完全不同：

```
backbone：已经在大规模数据上训练好，权重已接近最优
   → 只需要小幅调整，用小 lr（防止"遗忘"预训练知识）

head：随机初始化，权重离最优很远
   → 需要大幅更新，用大 lr
```

如果 backbone 和 head 用同一个大 lr，backbone 的权重会被"冲走"，预训练知识丢失，这就是 **Catastrophic Forgetting（灾难性遗忘）**。

### `param_groups` 的结构

```python
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},   # group 0
    {'params': head_params,     'lr': 1e-3},   # group 1
])

# 验证方式
print(optimizer.param_groups[0]['lr'])   # → 1e-5
print(optimizer.param_groups[1]['lr'])   # → 1e-3
```

> **本质**：`param_groups` 是一个列表，每个元素是一个字典，包含 `params`（参数列表）和 `lr`（学习率）等超参数。不同 group 可以有不同的 lr、weight_decay 等。

### 今天的实验设计

| 组别 | backbone_lr | head_lr | 比例 |
|------|------------|---------|------|
| 组 A | 1e-3 | 1e-3 | 1:1（对照，等同 Day 1 骨架） |
| 组 B | 1e-4 | 1e-3 | 1:10 |
| 组 C | 1e-5 | 1e-3 | 1:100 |

---

## Part 2：代码任务

### 文件结构

```
week13/day2/
└── differential_lr.py
```

---

### 完整代码：`differential_lr.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys, os

# ── 路径 ──────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))

from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════
# 1. 数据加载（复用 Day 1）
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, batch_size=32):
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 核心：差异化学习率设置
# ══════════════════════════════════════════════════════
def get_param_groups(esm2_model, classifier_head, backbone_lr, head_lr):
    """
    将 backbone 和 head 的参数分成两个 group，分别设置学习率。

    Args:
        esm2_model:      ESM-2 backbone
        classifier_head: 线性分类头
        backbone_lr:     backbone 的学习率
        head_lr:         分类头的学习率

    Returns:
        param_groups: list of dict，可直接传给 optimizer
    """
    backbone_params = [p for p in esm2_model.parameters() if p.requires_grad]
    head_params     = list(classifier_head.parameters())

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head_params,     'lr': head_lr,     'name': 'head'},
    ]

    # ── 验证输出（每次调用都打印，方便确认）──────────
    backbone_count = sum(p.numel() for p in backbone_params)
    head_count     = sum(p.numel() for p in head_params)
    print(f"  backbone: {backbone_count:,} params, lr={backbone_lr}")
    print(f"  head    : {head_count:,} params, lr={head_lr}")

    return param_groups


def build_finetune_model(num_classes):
    """构建 Fine-tune 模型（backbone 全部可训练）"""
    tokenizer, esm2_model = load_esm2()
    esm2_model = esm2_model.to(device)
    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)
    return esm2_model, classifier_head, tokenizer


# ══════════════════════════════════════════════════════
# 3. 训练 / 评估函数
# ══════════════════════════════════════════════════════
def get_esm2_embedding(esm2_model, input_ids, attention_mask):
    """Fine-tune 模式：backbone 参与梯度计算"""
    from protein_dataset import mean_pooling_with_mask
    outputs = esm2_model(input_ids=input_ids, attention_mask=attention_mask)
    return mean_pooling_with_mask(outputs.last_hidden_state, attention_mask)


def train_one_epoch(esm2_model, classifier_head, optimizer, criterion, train_loader):
    esm2_model.train()
    classifier_head.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        embeddings = get_esm2_embedding(esm2_model, input_ids, attention_mask)
        logits     = classifier_head(embeddings)
        loss       = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(esm2_model, classifier_head, loader):
    esm2_model.eval()
    classifier_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)
            embeddings = get_esm2_embedding(esm2_model, input_ids, attention_mask)
            preds      = classifier_head(embeddings).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1


# ══════════════════════════════════════════════════════
# 4. 单组实验
# ══════════════════════════════════════════════════════
def run_experiment(group_name, backbone_lr, head_lr,
                   train_loader, val_loader, test_loader,
                   num_classes, epochs=10):
    """
    跑一组差异化学习率实验，返回训练曲线数据
    """
    print(f"\n{'='*55}")
    print(f"实验组：{group_name}  "
          f"(backbone_lr={backbone_lr}, head_lr={head_lr})")
    print(f"{'='*55}")

    esm2_model, classifier_head, _ = build_finetune_model(num_classes)
    param_groups = get_param_groups(esm2_model, classifier_head,
                                    backbone_lr, head_lr)

    # ── 验证 lr 设置是否正确 ──────────────────────────
    optimizer = torch.optim.Adam(param_groups)
    for g in optimizer.param_groups:
        print(f"  [验证] group '{g['name']}': lr = {g['lr']}")

    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs, val_f1s = [], [], []

    for epoch in range(1, epochs + 1):
        train_loss       = train_one_epoch(esm2_model, classifier_head,
                                           optimizer, criterion, train_loader)
        val_acc, val_f1  = evaluate(esm2_model, classifier_head, val_loader)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    # ── 测试集最终结果 ────────────────────────────────
    test_acc, test_f1 = evaluate(esm2_model, classifier_head, test_loader)
    print(f"\n  【{group_name} 最终结果】"
          f" Test Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    return {
        'name'        : group_name,
        'backbone_lr' : backbone_lr,
        'head_lr'     : head_lr,
        'train_losses': train_losses,
        'val_accs'    : val_accs,
        'val_f1s'     : val_f1s,
        'test_acc'    : test_acc,
        'test_f1'     : test_f1,
    }


# ══════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════
def plot_results(results, save_dir):
    """
    画三张图：
    1. 训练 Loss 曲线对比
    2. Val Acc 曲线对比
    3. Val Macro F1 曲线对比
    """
    epochs = range(1, len(results[0]['train_losses']) + 1)
    colors = ['#e74c3c', '#2ecc71', '#3498db']   # 红/绿/蓝

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('差异化学习率实验对比（ESM-2 Fine-tune）', fontsize=13)

    metrics = [
        ('train_losses', 'Train Loss',     axes[0]),
        ('val_accs',     'Val Accuracy',   axes[1]),
        ('val_f1s',      'Val Macro F1',   axes[2]),
    ]

    for key, ylabel, ax in metrics:
        for res, color in zip(results, colors):
            ax.plot(epochs, res[key], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'differential_lr_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n训练曲线已保存：{save_path}")


def print_summary_table(results):
    """打印汇总表"""
    print(f"\n{'='*65}")
    print(f"{'实验组':<20} {'backbone_lr':<14} {'head_lr':<10} "
          f"{'Test Acc':<12} {'Macro F1'}")
    print(f"{'-'*65}")
    for res in results:
        print(f"{res['name']:<20} {res['backbone_lr']:<14} "
              f"{res['head_lr']:<10} {res['test_acc']:<12.4f} {res['test_f1']:.4f}")
    print(f"{'='*65}")

    # 找最优组
    best = max(results, key=lambda x: x['val_f1s'][-1])
    print(f"\n最优组（按最终 Val Macro F1）：{best['name']}")
    print(f"  backbone_lr / head_lr = {best['backbone_lr']} / {best['head_lr']}"
          f"  比例 = 1:{int(best['head_lr']/best['backbone_lr'])}")


# ══════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS      = 10   # 10 epoch 足够看出差异，不需要跑太久
    SAVE_DIR    = os.path.dirname(__file__)

    # 先拿 tokenizer 构建 DataLoader
    tokenizer, _ = load_esm2()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, batch_size=32
    )

    # ── 三组实验 ──────────────────────────────────────
    experiment_configs = [
        ('A: 1:1   (lr=1e-3)',  1e-3, 1e-3),
        ('B: 1:10  (lr=1e-4)',  1e-4, 1e-3),
        ('C: 1:100 (lr=1e-5)',  1e-5, 1e-3),
    ]

    all_results = []
    for name, backbone_lr, head_lr in experiment_configs:
        result = run_experiment(
            name, backbone_lr, head_lr,
            train_loader, val_loader, test_loader,
            num_classes=NUM_CLASSES, epochs=EPOCHS
        )
        all_results.append(result)

    # ── 汇总输出 ──────────────────────────────────────
    print_summary_table(all_results)
    plot_results(all_results, SAVE_DIR)

    print("\n【Day 2 完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 2 行。")
```

---

## 今日完成标准

- [ ] 三组实验均跑完，控制台打印 `[验证] group 'backbone': lr = ...` 确认 lr 设置正确
- [ ] 训练曲线图已生成（三条曲线对比）
- [ ] 汇总表已打印，能看出最优 lr 比例

---

## 输出问题（完成后回答）

1. **什么是 Catastrophic Forgetting？** 在你的实验中，组 A（1:1）相比组 C（1:100）有没有 Forgetting 的迹象？怎么判断？
2. **最优 backbone_lr / head_lr 比例是多少？** 你是怎么判断的（用 Val Acc 还是 Val Macro F1，为什么）？
3. **（观察题）** 组 A 的 Loss 下降最快，但最终 Val Acc 不一定最高——如果出现这种情况，说明了什么？