# Week 11 · Day 6：Fine-tuned 模型的完整测试评估

## 今日目标

用 `best_finetune.pt` 在测试集上做**最终评估**，与 Day4 的 frozen baseline 进行系统性对比，回答：

> Fine-tuning 是否真的提升了泛化性能？提升在哪些类别上？

---

## 背景回顾

| 阶段 | 方法 | 验证集 Accuracy |
|------|------|---------------:|
| Day3 | 冻结 ESM-2，只训练 MLP 分类头 | 0.634 |
| Day5 | 解冻最后 2 层 Fine-tuning | 0.677 |

但验证集不能作为最终结论，**测试集才是真正的泛化评估**。

---

## 代码

新建 `week11/day6/evaluate_finetune.py`：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day3'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day5'))
from esm2_embed import load_esm2
from protein_dataset import (
    load_localization_data, split_dataset,
    make_collate_fn, mean_pooling_with_mask,
    NUM_CLASSES, LOCALIZATION_CLASSES
)
from train_classifier import ProteinClassifier
from finetune import unfreeze_last_n_layers


# ── 1. 收集预测结果 ───────────────────────────────────────────
@torch.no_grad()
def get_predictions(
    esm_model,
    classifier: nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    遍历 loader，返回 (all_preds, all_labels)，shape 均为 (N,)。
    """
    esm_model.eval()
    classifier.eval()
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs    = esm_model(input_ids=input_ids,
                               attention_mask=attention_mask)
        embeddings = mean_pooling_with_mask(
            outputs.last_hidden_state, attention_mask)
        preds = classifier(embeddings).argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── 2. 每类别 Precision / Recall / F1 ────────────────────────
def per_class_report(
    preds:       np.ndarray,
    labels:      np.ndarray,
    class_names: dict,
    tag:         str = "",
) -> tuple[float, float, dict]:
    """
    打印每类别的 P / R / F1，返回 (accuracy, macro_f1, {c: f1})。
    """
    print(f"\n{'═'*70}")
    print(f"  {tag}")
    print(f"{'类别':>4}  {'名称':<25} {'支持数':>5}  "
          f"{'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("─" * 70)

    macro_f1   = 0.0
    f1_by_class = {}

    for c, name in class_names.items():
        tp      = ((preds == c) & (labels == c)).sum()
        fp      = ((preds == c) & (labels != c)).sum()
        fn      = ((preds != c) & (labels == c)).sum()
        support = (labels == c).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        macro_f1       += f1
        f1_by_class[c]  = float(f1)

        print(f"{c:>4}  {name:<25} {support:>5}  "
              f"{precision:>9.3f}  {recall:>6.3f}  {f1:>6.3f}")

    print("─" * 70)
    mf1 = macro_f1 / len(class_names)
    acc = float((preds == labels).mean())
    print(f"{'Macro F1':>40}  {mf1:>6.3f}")
    print(f"{'Overall Accuracy':>40}  {acc:>6.3f}")

    return acc, mf1, f1_by_class


# ── 3. Day4 vs Day6 对比表 ────────────────────────────────────
def print_comparison(
    acc_ft:      float,
    mf1_ft:      float,
    f1_ft:       dict,
    class_names: dict,
) -> None:
    """
    将 Fine-tuned 结果与 Day4 Frozen baseline 对比打印。
    Day4 结果直接写死（来自 Day4 的实际输出）。
    """
    # Day4 实际测试集结果
    frozen_acc = 0.664
    frozen_mf1 = 0.551
    frozen_f1  = {
        0: 0.683,   # Cell membrane
        1: 0.520,   # Cytoplasm
        2: 0.481,   # Endoplasmic reticulum
        3: 0.338,   # Golgi apparatus
        4: 0.291,   # Lysosome/Vacuole
        5: 0.629,   # Mitochondria
        6: 0.753,   # Nucleus
        7: 0.214,   # Peroxisome
        8: 0.707,   # Plastid
        9: 0.894,   # Extracellular
    }

    print(f"\n{'═'*58}")
    print(f"  Day4 Frozen  vs  Day6 Fine-tuned  （Test Set）")
    print(f"{'─'*58}")
    print(f"{'指标':<22} {'Day4 Frozen':>12} {'Day6 FT':>10} {'Δ':>8}")
    print(f"{'─'*58}")
    print(f"{'Overall Accuracy':<22} {frozen_acc:>12.3f} "
          f"{acc_ft:>10.3f} {acc_ft - frozen_acc:>+8.3f}")
    print(f"{'Macro F1':<22} {frozen_mf1:>12.3f} "
          f"{mf1_ft:>10.3f} {mf1_ft - frozen_mf1:>+8.3f}")
    print(f"{'═'*58}")

    print(f"\n  各类别 F1 对比：")
    print(f"{'类别':<25} {'Frozen':>8} {'FT':>8} {'Δ':>8}  {'趋势':>4}")
    print(f"{'─'*58}")
    for c, name in class_names.items():
        delta  = f1_ft[c] - frozen_f1[c]
        marker = "↑↑" if delta > 0.05 else (
                 "↑"  if delta > 0.02 else (
                 "↓↓" if delta < -0.05 else (
                 "↓"  if delta < -0.02 else "─")))
        print(f"{name:<25} {frozen_f1[c]:>8.3f} {f1_ft[c]:>8.3f} "
              f"{delta:>+8.3f}  {marker:>4}")
    print(f"{'─'*58}")


# ── 4. 混淆矩阵 ───────────────────────────────────────────────
def plot_confusion_matrix(
    preds:       np.ndarray,
    labels:      np.ndarray,
    class_names: dict,
    save_path:   str = "week11/day6/confusion_matrix_finetune.png",
) -> None:
    import matplotlib.pyplot as plt

    n  = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            cm[i][j] = ((labels == i) & (preds == j)).sum()

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    names = [class_names[i] for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – Fine-tuned ESM-2 (row-normalized)")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n混淆矩阵已保存至：{save_path}")


# ── 主程序 ────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据（seed=42 保证 test_ds 与 Day4 完全一致）
    sequences, labels = load_localization_data()
    _, _, test_ds = split_dataset(sequences, labels, seed=42)
    print(f"测试集大小：{len(test_ds)}")

    # 2. 加载模型结构
    tokenizer, esm_model = load_esm2(device=device)
    unfreeze_last_n_layers(esm_model, n=2)   # 结构必须与训练时一致
    classifier = ProteinClassifier().to(device)

    # 3. 加载 Fine-tuned 权重
    ckpt_path = os.path.join(
        os.path.dirname(__file__), '..', 'day5', 'best_finetune.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    esm_model.load_state_dict(ckpt['esm_state'])
    classifier.load_state_dict(ckpt['clf_state'])
    print("已加载 best_finetune.pt ✓")

    # 4. DataLoader
    collate_fn  = make_collate_fn(tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, collate_fn=collate_fn)

    # 5. 预测
    preds, true_labels = get_predictions(
        esm_model, classifier, test_loader, device)
    print(f"预测完成，共 {len(preds)} 条样本")

    # 6. 每类别报告
    acc_ft, mf1_ft, f1_ft = per_class_report(
        preds, true_labels, LOCALIZATION_CLASSES,
        tag="Day6 · Fine-tuned ESM-2（Test Set）")

    # 7. 对比表
    print_comparison(acc_ft, mf1_ft, f1_ft, LOCALIZATION_CLASSES)

    # 8. 混淆矩阵
    plot_confusion_matrix(preds, true_labels, LOCALIZATION_CLASSES)


if __name__ == "__main__":
    main()
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| 测试集大小 | 1259 条（与 Day4 完全一致） |
| Fine-tuned Test Accuracy | > Day4 的 0.664 |
| Fine-tuned Macro F1 | > Day4 的 0.551 |
| 混淆矩阵保存 | `confusion_matrix_finetune.png` |

---

## 输出问题

运行完成后，结合输出回答：

**Q1**：对比表中，哪个类别的 F1 提升最大（`Δ` 最高）？哪个类别反而下降了？你能从生物学角度猜测原因吗？

**Q2**：Fine-tuning 后 val acc 提升了 +0.043，test accuracy 提升了多少？两者是否接近？如果 test 提升远小于 val 提升，说明什么？

**Q3**：`unfreeze_last_n_layers` 在加载权重时也必须调用，为什么？如果不调用直接 `load_state_dict` 会发生什么错误？

---

提交：代码输出 + 混淆矩阵图片 + 三个问题的回答。