# Week 11 Day 4：测试集评估 + 混淆矩阵

训练完成，今天做**系统性评估**：加载最佳模型，在测试集上计算每个类别的 Precision/Recall/F1，并画混淆矩阵。

---

## 代码任务

新建 `week11/day4/evaluate_model.py`：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
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
from train_classifier import ProteinClassifier


# ── TODO 1：收集所有预测结果 ──────────────────────────────────
@torch.no_grad()
def get_predictions(
    esm_model,
    classifier: nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    遍历 loader，收集所有预测标签和真实标签。
    返回 (all_preds, all_labels)，均为 np.ndarray，shape (N,)
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
        logits = classifier(embeddings)
        preds  = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── TODO 2：每类别 Precision / Recall / F1 ───────────────────
def per_class_report(
    preds:  np.ndarray,
    labels: np.ndarray,
    class_names: dict,
) -> None:
    """
    手动计算每个类别的 TP / FP / FN，
    然后计算 Precision / Recall / F1，打印表格。

    公式：
      Precision[c] = TP[c] / (TP[c] + FP[c])
      Recall[c]    = TP[c] / (TP[c] + FN[c])
      F1[c]        = 2 * P * R / (P + R)

    提示：
      TP[c] = ((preds == c) & (labels == c)).sum()
      FP[c] = ((preds == c) & (labels != c)).sum()
      FN[c] = ((preds != c) & (labels == c)).sum()
    """
    print(f"\n{'类别':>4}  {'名称':<25} {'支持数':>5}  "
          f"{'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("─" * 70)

    macro_f1 = 0.0
    for c, name in class_names.items():
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        support = (labels == c).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 += f1

        print(f"{c:>4}  {name:<25} {support:>5}  "
              f"{precision:>9.3f}  {recall:>6.3f}  {f1:>6.3f}")

    print("─" * 70)
    print(f"{'Macro F1':>40}  {macro_f1 / len(class_names):>6.3f}")
    overall_acc = (preds == labels).mean()
    print(f"{'Overall Accuracy':>40}  {overall_acc:>6.3f}")


# ── TODO 3：混淆矩阵 ─────────────────────────────────────────
def plot_confusion_matrix(
    preds:  np.ndarray,
    labels: np.ndarray,
    class_names: dict,
    save_path: str = "confusion_matrix.png",
) -> None:
    """
    用 matplotlib 画混淆矩阵热图。

    步骤：
      1. 构建 confusion matrix：cm[i][j] = (labels==i & preds==j).sum()
      2. 归一化为行百分比（每行除以该行总数）
      3. 用 imshow 画热图，颜色越深代表比例越高
      4. 在每个格子里写数字（保留2位小数）
      5. 保存为 save_path
    """
    import matplotlib.pyplot as plt
    import matplotlib

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            cm[i][j] = ((labels == i) & (preds == j)).sum()

    # 归一化
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
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n混淆矩阵已保存至：{save_path}")


# ── 主程序 ────────────────────────────────────────────────────
def main():
    device = torch.device('cpu')

    # 1. 数据（必须用相同 seed 保证 test_ds 一致）
    sequences, labels = load_localization_data()
    _, _, test_ds = split_dataset(sequences, labels, seed=42)
    print(f"测试集大小：{len(test_ds)}")

    # 2. 加载模型
    tokenizer, esm_model = load_esm2(device=device)
    for p in esm_model.parameters():
        p.requires_grad = False

    classifier = ProteinClassifier().to(device)
    ckpt_path  = os.path.join(
        os.path.dirname(__file__), '..', 'day3', 'best_classifier.pt')
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("已加载 best_classifier.pt")

    # 3. DataLoader
    collate_fn  = make_collate_fn(tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, collate_fn=collate_fn)

    # 4. 获取预测
    preds, true_labels = get_predictions(
        esm_model, classifier, test_loader, device)
    print(f"预测完成，共 {len(preds)} 条样本")

    # 5. 每类别报告
    per_class_report(preds, true_labels, LOCALIZATION_CLASSES)

    # 6. 混淆矩阵
    plot_confusion_matrix(preds, true_labels, LOCALIZATION_CLASSES)


if __name__ == "__main__":
    main()
```

---

## 完成标准

1. 打印出 10 个类别的 Precision / Recall / F1 表格
2. Overall Accuracy 与 Day 3 的 val acc 接近（±5%）
3. `confusion_matrix.png` 成功保存
4. 观察哪些类别 F1 最低（通常是 Peroxisome 或 Lysosome）

---

## 输出问题

**Q1**：混淆矩阵对角线代表什么？哪个类别的对角线值最低？为什么？

**Q2**：Macro F1 和 Overall Accuracy 哪个更能反映模型在不均衡数据上的真实能力？为什么？

**Q3**：如果 Peroxisome（93条）的 Recall 很低但 Precision 很高，说明模型对这个类别的预测策略是什么？

提交代码输出 + 混淆矩阵图片。