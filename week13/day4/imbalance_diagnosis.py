import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 类别名称（顺序需与数据集标签编码一致）
CLASS_NAMES = [
    'Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
    'Golgi apparatus', 'Lysosome/Vacuole', 'Mitochondria',
    'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular'
]

BASELINE_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'day1', 'baseline_checkpoint.pt'
)


# ══════════════════════════════════════════════════════
# 1. 类别分布统计与可视化
# ══════════════════════════════════════════════════════
def analyze_class_distribution(labels, class_names):
    """
    统计每个类别的样本数，画柱状图，返回不平衡比例。
    """
    labels = np.array(labels)
    counts = np.array([np.sum(labels == i) for i in range(len(class_names))])

    max_count, min_count = counts.max(), counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n{'='*50}")
    print("类别分布统计")
    print(f"{'='*50}")
    for name, cnt in zip(class_names, counts):
        bar = '█' * int(cnt / max_count * 40)
        print(f"  {name:<25} {cnt:>5} {bar}")
    print(f"{'-'*50}")
    print(f"  最大类：{class_names[counts.argmax()]}（{max_count} 条）")
    print(f"  最小类：{class_names[counts.argmin()]}（{min_count} 条）")
    print(f"  不平衡比例（max/min）：{imbalance_ratio:.1f} : 1")
    print(f"{'='*50}")

    # ── 画图 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if c == min_count else
              ('#3498db' if c == max_count else '#95a5a6') for c in counts]
    bars = ax.bar(class_names, counts, color=colors)
    ax.set_ylabel('样本数')
    ax.set_title(f'类别分布（不平衡比例 = {imbalance_ratio:.1f} : 1）')
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(cnt), ha='center', fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n类别分布图已保存：{save_path}")

    return imbalance_ratio, counts


# ══════════════════════════════════════════════════════
# 2. 加载 Baseline checkpoint 并推理
# ══════════════════════════════════════════════════════
def load_baseline_model(vocab_size, num_classes, ckpt_path):
    model = ProteinClassifier(
        num_classes=num_classes, vocab_size=vocab_size,
        d_model=128, num_heads=4, num_layers=3,
        d_ff=512, max_len=512, dropout=0.1
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    # 兼容两种保存格式：直接 state_dict 或 {'model_state_dict': ...}
    state_dict = checkpoint.get('model_state_dict', checkpoint) \
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint \
                else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载 Baseline checkpoint：{ckpt_path}")
    return model


def get_predictions(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            preds = model(input_ids, mask).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ══════════════════════════════════════════════════════
# 3. 混淆矩阵可视化
# ══════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, class_names, save_name, normalize='row'):
    """
    normalize='row'：按真实类别归一化（每行和为1），方便看每个类别的误判去向
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    if normalize == 'row':
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
        title_suffix = '（按行归一化）'
    else:
        cm_display = cm
        fmt = 'd'
        title_suffix = '（原始计数）'

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_display, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title(f'Baseline 混淆矩阵{title_suffix}')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_display[i, j]
            color = 'white' if val > cm_display.max()/2 else 'black'
            ax.text(j, i, format(val, fmt), ha='center', va='center',
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存：{save_path}")

    return cm


def diagnose_worst_classes(cm, class_names, counts, top_k=3):
    """
    找出 F1/召回率最差的类别，以及它们最容易被误判为哪个类别。
    """
    print(f"\n{'='*60}")
    print("少数类误判诊断")
    print(f"{'='*60}")

    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    worst_idx = np.argsort(recalls)[:top_k]

    for idx in worst_idx:
        row = cm[idx].copy()
        correct = row[idx]
        row[idx] = -1   # 排除自己，找误判去向
        confused_with = np.argmax(row)
        confused_count = cm[idx, confused_with]
        total = cm[idx].sum() + correct - row[idx] if row[idx] == -1 else cm[idx].sum()
        total = cm[idx].sum()  # 修正：cm[idx] 未被修改，row 是拷贝

        print(f"\n  【{class_names[idx]}】（样本数：{counts[idx]}，Recall：{recalls[idx]:.2%}）")
        print(f"    正确预测：{correct} / {total}")
        print(f"    最常被误判为：{class_names[confused_with]}"
              f"（{confused_count} 次，占该类样本 {confused_count/total:.1%}）")

    print(f"{'='*60}")


# ══════════════════════════════════════════════════════
# 4. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    sequences, labels = load_localization_data()

    # ── Step 1：整体类别分布（用全量数据，反映真实不平衡程度）──
    imbalance_ratio, counts = analyze_class_distribution(labels, CLASS_NAMES)

    # ── Step 2：加载 test set，用 Baseline 模型推理 ──────
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)
    collate_fn = make_collate_fn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, collate_fn=collate_fn)

    model = load_baseline_model(vocab_size, NUM_CLASSES, BASELINE_CKPT_PATH)
    y_true, y_pred = get_predictions(model, test_loader)

    # ── Step 3：sklearn 分类报告（含 per-class F1）──────
    print(f"\n{'='*60}")
    print("Baseline 分类报告（Test Set）")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0, digits=3))

    # ── Step 4：混淆矩阵（归一化 + 原始计数各一张）──────
    cm_norm = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                                    'confusion_matrix_normalized.png',
                                    normalize='row')
    cm_raw  = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                                    'confusion_matrix_raw.png',
                                    normalize=None)

    # ── Step 5：诊断最差的类别 ───────────────────────────
    test_counts = np.array([np.sum(y_true == i) for i in range(NUM_CLASSES)])
    diagnose_worst_classes(cm_raw, CLASS_NAMES, test_counts, top_k=3)

    print("\n【Day 4 完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 4 行。")

#  输出
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1616.00it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 加载 ProtST-SubcellularLocalization 数据集...
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

# ==================================================
# 类别分布统计
# ==================================================
#   Cell membrane               800 █████████████
#   Cytoplasm                  1635 ██████████████████████████
#   Endoplasmic reticulum       516 ████████
#   Golgi apparatus             214 ███
#   Lysosome/Vacuole            192 ███
#   Mitochondria                906 ██████████████
#   Nucleus                    2424 ████████████████████████████████████████
#   Peroxisome                   93 █
#   Plastid                     453 ███████
#   Extracellular              1155 ███████████████████
# --------------------------------------------------
#   最大类：Nucleus（2424 条）
#   最小类：Peroxisome（93 条）
#   不平衡比例（max/min）：26.1 : 1
# ==================================================
# /home/tratr/dl-study/week13/day4/imbalance_diagnosis.py:69: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
#   ax.set_xticklabels(class_names, rotation=45, ha='right')

# 类别分布图已保存：/home/tratr/dl-study/week13/day4/class_distribution.png
# 已加载 Baseline checkpoint：/home/tratr/dl-study/week13/day4/../day1/baseline_checkpoint.pt

# ============================================================
# Baseline 分类报告（Test Set）
# ============================================================
#                        precision    recall  f1-score   support

#         Cell membrane      0.571     0.557     0.564       115
#             Cytoplasm      0.528     0.291     0.375       261
# Endoplasmic reticulum      0.420     0.474     0.446        78
#       Golgi apparatus      0.227     0.238     0.233        21
#      Lysosome/Vacuole      0.000     0.000     0.000        26
#          Mitochondria      0.777     0.721     0.748       140
#               Nucleus      0.600     0.867     0.710       369
#            Peroxisome      0.143     0.077     0.100        13
#               Plastid      0.935     0.763     0.841        76
#         Extracellular      0.839     0.844     0.841       160

#              accuracy                          0.633      1259
#             macro avg      0.504     0.483     0.486      1259
#          weighted avg      0.618     0.633     0.611      1259

# 混淆矩阵已保存：/home/tratr/dl-study/week13/day4/confusion_matrix_normalized.png
# 混淆矩阵已保存：/home/tratr/dl-study/week13/day4/confusion_matrix_raw.png

# ============================================================
# 少数类误判诊断
# ============================================================

#   【Lysosome/Vacuole】（样本数：26，Recall：0.00%）
#     正确预测：0 / 26
#     最常被误判为：Cell membrane（10 次，占该类样本 38.5%）

#   【Peroxisome】（样本数：13，Recall：7.69%）
#     正确预测：1 / 13
#     最常被误判为：Cytoplasm（3 次，占该类样本 23.1%）

#   【Golgi apparatus】（样本数：21，Recall：23.81%）
#     正确预测：5 / 21
#     最常被误判为：Cell membrane（5 次，占该类样本 23.8%）
# ============================================================

# 【Day 4 完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 4 行。

# 1. 不平衡比例是多少？ 最大类和最小类分别是什么？
# 不平衡比例为 26.1 : 1，最大类是 Nucleus（2424 条），最小类是 Peroxisome（93 条）。

# 2。 Peroxisome（或实际跑出来的最差类别）主要被误判为哪个类别？
# 从生物学角度猜一下可能的原因（提示：想想这两个细胞器在序列组成或长度上是否有相似性）。
# Peroxisome 主要被误判为 Cytoplasm（3 次，占该类样本 23.1%）。
# 从生物学角度来看，Peroxisome 和 Cytoplasm 在蛋白质序列组成上可能有一定的相似性，
# 尤其是在信号肽和定位信号方面，这可能导致模型难以区分这两类蛋白质。

# 3. 对比 sklearn 分类报告里的 Macro F1，和 Day 1 记录的 Baseline Macro F1（0.4527）是否一致？
# 如果不一致，可能是什么原因（提示：想想 Day 1 用的是哪个 checkpoint，训练了多少 epoch）。
# sklearn 分类报告里的 Macro F1 为 0.486，与 Day 1 记录的 Baseline Macro F1（0.4527）不一致。
# 可能的原因是 Day 1 使用的 checkpoint 是在训练了 5 个 epoch 后保存的，
# 而当前使用的 checkpoint 可能是训练了更多 epoch 的模型，导致 Macro F1 提高。