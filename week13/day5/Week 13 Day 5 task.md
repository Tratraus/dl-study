# Week 13 Day 5：类别不平衡处理 + 评估指标

## 今日目标

1. **修复工程债务**：训练脚本加入 `torch.manual_seed`，保证可复现
2. 实现**加权 CrossEntropy Loss**，用类别频率的倒数作为权重
3. 封装 `evaluate()` 函数，验证其输出与 sklearn 一致
4. 对比**加权前后**的 Macro F1、混淆矩阵，重点关注 Day 4 诊断出的两个最差类别（Lysosome/Vacuole、Peroxisome）

---

## Part 1：理论（5 分钟）

### 加权 Loss 的原理

标准 CrossEntropy 对每个样本一视同仁地计算损失，这意味着模型看到 2424 条 Nucleus 样本和 93 条 Peroxisome 样本时，**梳理"少犯错就能拿高分"的最优策略就是把所有模糊样本都判给 Nucleus**——这正是 Day 4 混淆矩阵里 Cytoplasm/Cell membrane 大量样本被误判进 Nucleus 的根源。

**加权 CE 的做法**：给每个类别的 loss 乘一个权重 $$w_c$$，让模型在少数类上犯错的代价变大：

$$\text{Loss} = -\sum_{c} w_c \cdot y_c \log(\hat{y}_c)$$

最常用的权重公式是**倒频率权重**：

$$w_c = \frac{N}{K \cdot n_c}$$

其中 $$N$$ 是总样本数，$$K$$ 是类别数，$$n_c$$ 是类别 $$c$$ 的样本数。这样权重的均值恰好是 1，不会整体改变 loss 的量级。

### 一个常见陷阱：倒频率权重可能"过度纠正"

按 Day 4 的数据，Peroxisome（93条）和 Nucleus（2424条）的倒频率权重比是 **26:1**——如果直接套用，模型可能又走向另一个极端：为了讨好 Peroxisome，牺牲掉 Nucleus 的表现，导致整体 Acc 大跌。

工程上更稳健的做法是用**平方根倒频率**：

$$w_c = \sqrt{\frac{N}{K \cdot n_c}}$$

这样极端类别间的权重差距被压缩（26:1 变成约 5:1），既缓解不平衡又不会矫枉过正。**我们今天两种都实现，对比效果**。

---

## Part 2：代码任务

### 文件结构

```
week13/day5/
├── evaluate.py
└── weighted_loss_experiment.py
```

---

### 文件 1：`evaluate.py`（复用于 Day 6、Day 7）

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import torch


def evaluate(model, loader, device, class_names=None):
    """
    统一的评估函数：跑一遍 loader，返回一个字典，包含常用指标。

    Returns:
        dict，包含：
            - accuracy: float
            - macro_f1: float
            - weighted_f1: float
            - per_class_f1: np.ndarray，shape (num_classes,)
            - per_class_recall: np.ndarray
            - confusion_matrix: np.ndarray
            - y_true, y_pred: np.ndarray（供后续画图/诊断用）
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            logits = model(input_ids, mask)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_f1     = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        'accuracy':          acc,
        'macro_f1':          macro_f1,
        'weighted_f1':       weighted_f1,
        'per_class_f1':      per_class_f1,
        'per_class_recall':  per_class_recall,
        'confusion_matrix':  cm,
        'y_true':            y_true,
        'y_pred':            y_pred,
    }

    if class_names is not None:
        result['report'] = classification_report(
            y_true, y_pred, target_names=class_names,
            zero_division=0, digits=3
        )

    return result


def verify_against_sklearn(result, y_true, y_pred):
    """
    自测函数：验证 evaluate() 内部计算的指标与直接调用 sklearn 是否一致。
    用于 Day 5 的正确性检查，正式训练流程中不需要调用。
    """
    ref_acc      = accuracy_score(y_true, y_pred)
    ref_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    assert np.isclose(result['accuracy'], ref_acc), \
        f"Accuracy 不一致：{result['accuracy']} vs {ref_acc}"
    assert np.isclose(result['macro_f1'], ref_macro_f1), \
        f"Macro F1 不一致：{result['macro_f1']} vs {ref_macro_f1}"

    print("✅ evaluate() 输出与 sklearn 参考值一致")
```

---

### 文件 2：`weighted_loss_experiment.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier
from evaluate import evaluate, verify_against_sklearn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CLASS_NAMES = [
    'Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
    'Golgi apparatus', 'Lysosome/Vacuole', 'Mitochondria',
    'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular'
]

# ⚠️ 重点关注这两个 Day4 诊断出的最差类别
WATCH_CLASSES = {'Lysosome/Vacuole': 4, 'Peroxisome': 7}


# ══════════════════════════════════════════════════════
# 0. 固定所有随机种子（修复 Day4 发现的工程债务）
# ══════════════════════════════════════════════════════
def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cudnn 卷积等操作的确定性（会略微牺牲速度，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════
# 1. 计算类别权重
# ══════════════════════════════════════════════════════
def compute_class_weights(labels, num_classes, mode='inverse'):
    """
    mode:
        'inverse'      -> w_c = N / (K * n_c)          （标准倒频率）
        'sqrt_inverse' -> w_c = sqrt(N / (K * n_c))     （压缩版，更稳健）
    """
    labels = np.array(labels)
    counts = np.array([np.sum(labels == i) for i in range(num_classes)])
    N, K = len(labels), num_classes

    if mode == 'inverse':
        weights = N / (K * counts)
    elif mode == 'sqrt_inverse':
        weights = np.sqrt(N / (K * counts))
    else:
        raise ValueError(f"未知模式：{mode}")

    return torch.tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════
# 2. 训练函数（支持传入 loss 权重）
# ══════════════════════════════════════════════════════
def train_model(train_loader, val_loader, vocab_size, num_classes,
                class_weights=None, epochs=20, tag=""):
    set_all_seeds(42)   # 每组实验前重置种子，保证模型初始化一致

    model = ProteinClassifier(
        num_classes=num_classes, vocab_size=vocab_size,
        d_model=128, num_heads=4, num_layers=3,
        d_ff=512, max_len=512, dropout=0.1
    ).to(device)

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n{'='*55}\n训练：{tag}\n{'='*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for input_ids, mask, labels in train_loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, mask), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            val_result = evaluate(model, val_loader, device)
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}"
                  f" | Val Acc: {val_result['accuracy']:.4f}"
                  f" | Val Macro F1: {val_result['macro_f1']:.4f}")

    return model


# ══════════════════════════════════════════════════════
# 3. 混淆矩阵对比画图（加权前后各一张，聚焦少数类）
# ══════════════════════════════════════════════════════
def plot_confusion_comparison(cm_before, cm_after, class_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = ['加权前（Baseline CE）', '加权后（Weighted CE）']

    for ax, cm, title in zip(axes, [cm_before, cm_after], titles):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_title(title)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵对比图已保存：{save_path}")


# ══════════════════════════════════════════════════════
# 4. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS = 20
    SAVE_DIR = os.path.dirname(__file__)

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 训练集标签，用于计算类别权重（必须用 train，不能用全量，避免信息泄露）
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    # ── 计算两种权重 ─────────────────────────────────
    weights_inverse      = compute_class_weights(train_labels, NUM_CLASSES, mode='inverse')
    weights_sqrt_inverse = compute_class_weights(train_labels, NUM_CLASSES, mode='sqrt_inverse')

    print(f"\n{'='*60}\n类别权重对比\n{'='*60}")
    print(f"{'类别':<25}{'倒频率':<12}{'平方根倒频率'}")
    for name, w1, w2 in zip(CLASS_NAMES, weights_inverse, weights_sqrt_inverse):
        print(f"{name:<25}{w1:<12.3f}{w2:.3f}")

    # ── 组 A：Baseline（无权重）── 复用 Day4 校正后的 checkpoint 逻辑，这里重新训练一次以保证种子固定 ──
    model_baseline = train_model(train_loader, val_loader, vocab_size, NUM_CLASSES,
                                  class_weights=None, epochs=EPOCHS, tag="A: Baseline (无加权)")

    # ── 组 B：倒频率加权 ──────────────────────────────
    model_inverse = train_model(train_loader, val_loader, vocab_size, NUM_CLASSES,
                                 class_weights=weights_inverse, epochs=EPOCHS,
                                 tag="B: 倒频率加权 CE")

    # ── 组 C：平方根倒频率加权 ────────────────────────
    model_sqrt = train_model(train_loader, val_loader, vocab_size, NUM_CLASSES,
                              class_weights=weights_sqrt_inverse, epochs=EPOCHS,
                              tag="C: 平方根倒频率加权 CE")

    # ── 在 Test Set 上评估三组 ────────────────────────
    print(f"\n{'='*70}\nTest Set 最终对比\n{'='*70}")
    results = {}
    for name, model in [('A: Baseline', model_baseline),
                        ('B: 倒频率加权', model_inverse),
                        ('C: 平方根倒频率加权', model_sqrt)]:
        res = evaluate(model, test_loader, device, class_names=CLASS_NAMES)
        results[name] = res
        print(f"\n【{name}】Test Acc: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
        for watch_name, watch_idx in WATCH_CLASSES.items():
            print(f"    {watch_name:<20} F1: {res['per_class_f1'][watch_idx]:.3f}"
                  f" | Recall: {res['per_class_recall'][watch_idx]:.3f}")

    # ── 用 A 组结果自测 evaluate() 正确性 ─────────────
    verify_against_sklearn(results['A: Baseline'],
                           results['A: Baseline']['y_true'],
                           results['A: Baseline']['y_pred'])

    # ── 混淆矩阵对比：Baseline vs 效果更好的加权组 ────
    best_weighted = 'B: 倒频率加权' if results['B: 倒频率加权']['macro_f1'] > results['C: 平方根倒频率加权']['macro_f1'] \
                    else 'C: 平方根倒频率加权'

    plot_confusion_comparison(
        results['A: Baseline']['confusion_matrix'],
        results[best_weighted]['confusion_matrix'],
        CLASS_NAMES,
        os.path.join(SAVE_DIR, 'confusion_matrix_weighted_comparison.png')
    )

    # ── 汇总表 ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'实验组':<25}{'Test Acc':<12}{'Macro F1':<12}"
          f"{'Lysosome F1':<14}{'Peroxisome F1'}")
    print(f"{'-'*70}")
    for name, res in results.items():
        print(f"{name:<25}{res['accuracy']:<12.4f}{res['macro_f1']:<12.4f}"
              f"{res['per_class_f1'][4]:<14.3f}{res['per_class_f1'][7]:.3f}")
    print(f"{'='*70}")

    print("\n【Day 5 完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 5 行。")
```

---

## 今日完成标准

- [ ] `torch.manual_seed` 已加入训练脚本，三组实验用同一初始化种子
- [ ] `evaluate()` 函数输出通过 `verify_against_sklearn` 自测，无报错
- [ ] 倒频率 / 平方根倒频率两种权重都跑完，Test Acc & Macro F1 有记录
- [ ] 混淆矩阵加权前后对比图生成，重点看 Lysosome/Vacuole 和 Peroxisome 两行是否改善

---

## 输出问题（跑完后回答）

1. **倒频率加权 vs 平方根倒频率加权，哪个 Macro F1 更高？** 整体 Test Acc 是否因为加权而下降（对比新 Baseline 0.6331）？下降了多少？
2. **重点看 Lysosome/Vacuole 和 Peroxisome 两个类别的 F1**：加权后是否有提升？如果提升了，是通过"牺牲"了哪个多数类的表现换来的（看混淆矩阵哪一列/哪一行的对角线值下降了）？
3. **（权衡思考）** 如果加权后 Macro F1 提升但 Accuracy 明显下降，在实际的计算生物学场景中（比如药物靶点的亚细胞定位预测），你会更看重哪个指标？为什么？