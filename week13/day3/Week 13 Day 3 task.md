# Week 13 Day 3：蛋白质序列数据增强

## 今日目标

1. **理解**保守氨基酸替换和 Random Cropping 的生物学/工程原理
2. **实现** `conservative_substitution()` 和 `random_crop()` 两个增强函数
3. **对比**增强前后自实现模型的 Val Acc 曲线（与 Day 1 Baseline 对照）

---

## Part 1：理论（5 分钟）

### 为什么蛋白质序列可以做数据增强？

蛋白质的功能主要由**理化性质**决定，而不是精确的氨基酸序列本身。同一理化类别内的氨基酸互换，通常不改变蛋白质的定位/功能：

| 分组 | 氨基酸 | 共同性质 |
|------|--------|---------|
| 脂肪族 | A, V, L, I, M | 疏水、非极性 |
| 芳香族 | F, W, Y | 疏水、大侧链 |
| 极性不带电 | S, T, N, Q | 可形成氢键 |
| 带正电 | K, R, H | 碱性 |
| 带负电 | D, E | 酸性 |
| 特殊 | C, G, P | 二硫键/柔性/刚性 |

> **保守替换**：Val→Ile→Leu 互换，蛋白质折叠几乎不受影响。这是一种**符合生物学先验**的数据增强，比随机替换更合理（随机替换可能破坏关键的疏水核心或活性位点）。

### Random Cropping 的直觉

蛋白质的亚细胞定位往往由**局部信号肽**决定（比如 N 端的信号肽、C 端的定位标签），而不需要看完整序列。随机截取子序列，相当于让模型学会"从局部片段也能判断定位"，减少对序列长度和绝对位置的过拟合。

### 关键原则：增强只在训练集用

> 测试集必须保持原始分布，否则你评估的是"增强后序列的准确率"，而不是"真实数据的泛化能力"。

---

## Part 2：代码任务

### 文件结构

```
week13/day3/
├── augmentation.py      ← 今天的核心文件
└── run_augmentation.py  ← 训练对比脚本
```

---

### 文件 1：`augmentation.py`

```python
import random

# ══════════════════════════════════════════════════════
# 氨基酸保守替换分组
# ══════════════════════════════════════════════════════
CONSERVATIVE_GROUPS = [
    set('AVLIM'),   # 脂肪族疏水
    set('FWY'),     # 芳香族
    set('STNQ'),    # 极性不带电
    set('KRH'),     # 带正电
    set('DE'),      # 带负电
    set('CGP'),     # 特殊结构
]

# 构建 氨基酸 -> 同组其他氨基酸列表 的映射，加速查找
_AA_TO_GROUP_MATES = {}
for group in CONSERVATIVE_GROUPS:
    for aa in group:
        _AA_TO_GROUP_MATES[aa] = list(group - {aa})


def conservative_substitution(seq: str, prob: float = 0.1) -> str:
    """
    保守氨基酸替换：以 prob 概率将每个氨基酸替换为同组的其他氨基酸。

    Args:
        seq:  原始蛋白质序列（大写字母字符串）
        prob: 每个位置被替换的概率

    Returns:
        增强后的序列（长度不变）
    """
    seq_list = list(seq)
    for i, aa in enumerate(seq_list):
        if aa in _AA_TO_GROUP_MATES and random.random() < prob:
            mates = _AA_TO_GROUP_MATES[aa]
            if mates:   # 该组内确实有其他成员可替换
                seq_list[i] = random.choice(mates)
    return ''.join(seq_list)


def random_crop(seq: str, min_len: int = 30) -> str:
    """
    随机截取子序列。若序列本身长度 <= min_len，则不裁剪，原样返回。

    Args:
        seq:     原始蛋白质序列
        min_len: 裁剪后的最小长度

    Returns:
        裁剪后的子序列
    """
    seq_len = len(seq)
    if seq_len <= min_len:
        return seq

    # 随机决定本次裁剪后的长度：[min_len, seq_len] 之间
    crop_len = random.randint(min_len, seq_len)
    # 随机决定起始位置
    max_start = seq_len - crop_len
    start = random.randint(0, max_start)
    return seq[start:start + crop_len]


# ══════════════════════════════════════════════════════
# 数据集包装器：在原有 ProteinDataset 基础上加增强
# ══════════════════════════════════════════════════════
from torch.utils.data import Dataset

class AugmentedProteinDataset(Dataset):
    """
    包装原始 ProteinDataset，训练时开启增强，验证/测试时关闭。

    用法：
        train_ds = AugmentedProteinDataset(base_train_dataset, augment=True)
        val_ds   = AugmentedProteinDataset(base_val_dataset,   augment=False)
    """
    def __init__(self, base_dataset, augment: bool = False,
                sub_prob: float = 0.1, crop_min_len: int = 30):
        self.base_dataset = base_dataset
        self.augment      = augment
        self.sub_prob     = sub_prob
        self.crop_min_len = crop_min_len

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        seq, label = self.base_dataset[idx]

        if self.augment:
            seq = conservative_substitution(seq, prob=self.sub_prob)
            seq = random_crop(seq, min_len=self.crop_min_len)

        return seq, label


if __name__ == '__main__':
    # ── 快速自测 ──────────────────────────────────────
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

    print("原始序列长度：", len(test_seq))
    print("原始序列前50位：", test_seq[:50])

    sub_seq = conservative_substitution(test_seq, prob=0.1)
    print("\n保守替换后前50位：", sub_seq[:50])
    diff_count = sum(1 for a, b in zip(test_seq, sub_seq) if a != b)
    print(f"替换位点数：{diff_count} / {len(test_seq)}"
          f"（约 {100*diff_count/len(test_seq):.1f}%）")

    crop_seq = random_crop(test_seq, min_len=30)
    print(f"\n裁剪后长度：{len(crop_seq)}（原长度 {len(test_seq)}）")
    print("裁剪后序列：", crop_seq[:50], "...")
```

---

### 文件 2：`run_augmentation.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
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
from augmentation import AugmentedProteinDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════
# 1. 数据加载（增强版 vs 原始版）
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, use_augmentation, batch_size=32):
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)

    # ── 只在训练集上包装增强，val/test 保持原始分布 ──
    train_dataset = AugmentedProteinDataset(
        train_dataset, augment=use_augmentation,
        sub_prob=0.1, crop_min_len=30
    )
    val_dataset  = AugmentedProteinDataset(val_dataset,  augment=False)
    test_dataset = AugmentedProteinDataset(test_dataset, augment=False)

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 训练 / 评估（与 Day 1 Baseline 完全一致的超参）
# ══════════════════════════════════════════════════════
def train_and_evaluate(group_name, train_loader, val_loader, test_loader,
                       vocab_size, num_classes, epochs=20):
    print(f"\n{'='*55}")
    print(f"实验组：{group_name}")
    print(f"{'='*55}")

    # ── 与 Day 1 Baseline 完全相同的超参数 ─────────────
    model = ProteinClassifier(
        num_classes = num_classes,
        vocab_size  = vocab_size,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 3,
        d_ff        = 512,
        max_len     = 512,
        dropout     = 0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    val_accs, val_f1s = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for input_ids, mask, labels in train_loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, mask, labels in val_loader:
                input_ids = input_ids.to(device)
                mask      = mask.to(device)
                labels    = labels.to(device)
                preds = model(input_ids, mask).argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}"
                  f" | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    # ── 测试集最终评估 ──────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, mask, labels in test_loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            preds = model(input_ids, mask).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n  【{group_name} 最终结果】"
          f" Test Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    return {
        'name'    : group_name,
        'val_accs': val_accs,
        'val_f1s' : val_f1s,
        'test_acc': test_acc,
        'test_f1' : test_f1,
    }


# ══════════════════════════════════════════════════════
# 3. 可视化对比
# ══════════════════════════════════════════════════════
def plot_comparison(results, save_dir):
    epochs = range(1, len(results[0]['val_accs']) + 1)
    colors = ['#95a5a6', '#e74c3c']   # 灰=无增强，红=增强

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('数据增强前后对比（自实现模型）', fontsize=13)

    for res, color in zip(results, colors):
        axes[0].plot(epochs, res['val_accs'], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs, res['val_f1s'], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)

    axes[0].set_title('Val Accuracy')
    axes[1].set_title('Val Macro F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'augmentation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n对比图已保存：{save_path}")


# ══════════════════════════════════════════════════════
# 4. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS      = 20   # 与 Day 1 Baseline 保持一致
    SAVE_DIR    = os.path.dirname(__file__)

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    all_results = []

    # ── 组 A：无增强（重跑一次，作为本机可比对照） ──
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, use_augmentation=False, batch_size=32
    )
    result_no_aug = train_and_evaluate(
        "无增强（对照）", train_loader, val_loader, test_loader,
        vocab_size, NUM_CLASSES, epochs=EPOCHS
    )
    all_results.append(result_no_aug)

    # ── 组 B：数据增强 ────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, use_augmentation=True, batch_size=32
    )
    result_aug = train_and_evaluate(
        "数据增强", train_loader, val_loader, test_loader,
        vocab_size, NUM_CLASSES, epochs=EPOCHS
    )
    all_results.append(result_aug)

    # ── 汇总 ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'实验组':<15} {'Test Acc':<12} {'Macro F1'}")
    print(f"{'-'*55}")
    for res in all_results:
        print(f"{res['name']:<15} {res['test_acc']:<12.4f} {res['test_f1']:.4f}")
    print(f"{'-'*55}")
    print(f"{'Day1 Baseline(记录)':<15} {0.6124:<12.4f} {0.4527:.4f}")
    print(f"{'='*55}")

    plot_comparison(all_results, SAVE_DIR)

    print("\n【Day 3 完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 3 行。")
```

---

## 需要注意的一点

`AugmentedProteinDataset` 假设 `ProteinDataset.__getitem__` 返回 `(seq_str, label)` 元组（即返回原始字符串序列，而非已 tokenize 的张量），tokenize 在 `collate_fn` 里统一完成。

**如果你的 `ProteinDataset.__getitem__` 返回的格式不同**（比如已经是 tokenized 张量，或返回 dict），运行会报错——把报错信息发给我，我按你的实际结构调整 `AugmentedProteinDataset`。

---

## 今日完成标准

- [ ] `conservative_substitution` 自测通过（替换位点数约等于 `prob × 序列长度`）
- [ ] `random_crop` 自测通过（裁剪后长度 ≥ min_len，且 ≤ 原长度）
- [ ] 增强前后 Val Acc / Macro F1 曲线对比图生成
- [ ] 增强组 Test Acc/F1 相比无增强对照组（或 Day 1 Baseline）有可观察的变化

---

## 输出问题（完成后回答）

1. **数据增强为什么只在训练集上用，不在测试集上用？**
2. **保守替换和随机替换有什么本质区别？** 为什么保守替换更合理？
3. **（观察题）** 增强后的 Test Acc/F1 相比无增强组有提升吗？如果提升有限甚至下降，可能是什么原因（提示：想想 `sub_prob=0.1` 和 `crop_min_len=30` 这两个超参数是否合适，以及模型本身参数量只有 600K）？