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
    save_path: str = "week11/day4/confusion_matrix.png",
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 测试集大小：1259
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1828.82it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 已加载 best_classifier.pt
# 预测完成，共 1259 条样本

#   类别  名称                          支持数  Precision  Recall      F1
# ──────────────────────────────────────────────────────────────────────
#    0  Cell membrane               115      0.778   0.609   0.683
#    1  Cytoplasm                   261      0.574   0.475   0.520
#    2  Endoplasmic reticulum        78      0.582   0.410   0.481
#    3  Golgi apparatus              21      0.250   0.524   0.338
#    4  Lysosome/Vacuole             26      0.276   0.308   0.291
#    5  Mitochondria                140      0.622   0.636   0.629
#    6  Nucleus                     369      0.724   0.783   0.753
#    7  Peroxisome                   13      0.200   0.231   0.214
#    8  Plastid                      76      0.659   0.763   0.707
#    9  Extracellular               160      0.844   0.950   0.894
# ──────────────────────────────────────────────────────────────────────
#                                 Macro F1   0.551
#                         Overall Accuracy   0.664

# 混淆矩阵已保存至：week11/day4/confusion_matrix.png

# Q1：混淆矩阵对角线代表什么？哪个类别的对角线值最低？为什么？
# 代表模型预测正确的样本数。
# Peroxisome 类别的对角线值最低，可能是因为该类别的样本数量较少（仅 93 条），
# 导致模型在训练过程中对该类别的学习不足，从而预测准确率较低。

# Q2：Macro F1 和 Overall Accuracy 哪个更能反映模型在不均衡数据上的真实能力？为什么？
# Macro F1 更能反映模型在不均衡数据上的真实能力，
# 因为它对每个类别的 F1 分数进行平均，能够更好地体现模型在少数类上的表现，
# 而 Overall Accuracy 可能会被多数类的高准确率所掩盖。

# Q3：如果 Peroxisome（93条）的 Recall 很低但 Precision 很高，说明模型对这个类别的预测策略是什么？
# 说明模型对 Peroxisome 类别的预测策略是保守的，即模型只在非常有信心的情况下才会预测为 Peroxisome，
# 导致预测为 Peroxisome 的样本数量较少，从而 Precision 很高，
# 但 Recall 很低，因为很多实际属于 Peroxisome 的样本被预测为其他类别。