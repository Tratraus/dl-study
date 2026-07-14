import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # 必须在其他CUDA操作之前设置

import torch
torch.use_deterministic_algorithms(True)   # 比 cudnn.deterministic 更严格

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import sys

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

WATCH_CLASSES = {'Lysosome/Vacuole': 4, 'Peroxisome': 7}

# 官方基准（Day1 修复种子后重跑的 Baseline v5，请用你自己最新跑出来的数字替换）
BASELINE_V5 = {'accuracy': 0.6481334392374901, 'macro_f1': 0.46426376359159355}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_seed] 全局随机种子已固定为 {seed}")


def get_dataloaders(tokenizer, batch_size=32, seed=42):
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)
    collate_fn = make_collate_fn(tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    # ★ 新增：把 collate_fn 也返回出去，供 train_model 重建 DataLoader 用
    return train_loader, val_loader, test_loader, train_dataset, collate_fn


def compute_class_weights(labels, num_classes, mode='inverse'):
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


def train_model(train_dataset, val_loader, collate_fn, vocab_size, num_classes,
                class_weights=None, epochs=20, tag="", seed=42, save_path=None):
    """
    ★ train_dataset 必须传 Dataset 对象（不是 DataLoader！）
    ★ collate_fn 必须显式传入（不能依赖外部作用域）
    每组训练都会用全新 generator 重建 train_loader，保证 shuffle 顺序从头开始，
    不受其他组训练顺序影响。
    """
    set_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, generator=torch.Generator().manual_seed(seed)
    )

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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  💾 已保存 checkpoint: {save_path}")
    return model


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


if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS = 20
    SAVE_DIR = os.path.dirname(__file__)
    SEED = 42

    set_seed(SEED)

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    # ★ 修改：接收新增的 collate_fn 返回值
    train_loader, val_loader, test_loader, train_dataset, collate_fn = get_dataloaders(
        tokenizer=tokenizer, batch_size=32, seed=SEED)

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    weights_inverse      = compute_class_weights(train_labels, NUM_CLASSES, mode='inverse')
    weights_sqrt_inverse = compute_class_weights(train_labels, NUM_CLASSES, mode='sqrt_inverse')

    print(f"\n{'='*60}\n类别权重对比\n{'='*60}")
    print(f"{'类别':<25}{'倒频率':<12}{'平方根倒频率'}")
    for name, w1, w2 in zip(CLASS_NAMES, weights_inverse, weights_sqrt_inverse):
        print(f"{name:<25}{w1:<12.3f}{w2:.3f}")

    CKPT_DIR = os.path.join(SAVE_DIR, 'checkpoints')
    # ★ 修改：调用处统一传 train_dataset（不是 train_loader）+ collate_fn
    model_baseline = train_model(train_dataset, val_loader, collate_fn, vocab_size, NUM_CLASSES,
                                  class_weights=None, epochs=EPOCHS,
                                  tag="A: Baseline (无加权)", seed=SEED,
                                  save_path=os.path.join(CKPT_DIR, 'baseline.pth'))

    model_inverse = train_model(train_dataset, val_loader, collate_fn, vocab_size, NUM_CLASSES,
                                class_weights=weights_inverse, epochs=EPOCHS,
                                tag="B: 倒频率加权 CE", seed=SEED,
                                save_path=os.path.join(CKPT_DIR, 'inverse.pth'))

    model_sqrt = train_model(train_dataset, val_loader, collate_fn, vocab_size, NUM_CLASSES,
                              class_weights=weights_sqrt_inverse, epochs=EPOCHS,
                              tag="C: 平方根倒频率加权 CE", seed=SEED,
                              save_path=os.path.join(CKPT_DIR, 'sqrt_inverse.pth'))

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

    verify_against_sklearn(results['A: Baseline'],
                           results['A: Baseline']['y_true'],
                           results['A: Baseline']['y_pred'])

    print(f"\n{'='*70}\n★ 种子固定有效性核验 ★\n{'='*70}")
    acc_diff = abs(results['A: Baseline']['accuracy'] - BASELINE_V5['accuracy'])
    f1_diff  = abs(results['A: Baseline']['macro_f1'] - BASELINE_V5['macro_f1'])
    print(f"Baseline v4 (Day1)     : Acc={BASELINE_V5['accuracy']:.4f} | F1={BASELINE_V5['macro_f1']:.4f}")
    print(f"A组本次结果 (Day5)      : Acc={results['A: Baseline']['accuracy']:.4f} | F1={results['A: Baseline']['macro_f1']:.4f}")
    print(f"差异                   : ΔAcc={acc_diff:.4f} | ΔF1={f1_diff:.4f}")
    if acc_diff < 0.001 and f1_diff < 0.001:
        print("✅ 种子固定完全生效，A/B/C 三组现在与 Baseline v4 严格可比")
    else:
        print("⚠️ 仍存在差异，需要进一步排查（检查 load_esm2 内部是否消耗随机数等）")

    best_weighted = 'B: 倒频率加权' if results['B: 倒频率加权']['macro_f1'] > results['C: 平方根倒频率加权']['macro_f1'] \
                    else 'C: 平方根倒频率加权'

    plot_confusion_comparison(
        results['A: Baseline']['confusion_matrix'],
        results[best_weighted]['confusion_matrix'],
        CLASS_NAMES,
        os.path.join(SAVE_DIR, 'confusion_matrix_weighted_comparison.png')
    )

    print(f"\n{'='*70}")
    print(f"{'实验组':<25}{'Test Acc':<12}{'Macro F1':<12}"
          f"{'Lysosome F1':<14}{'Peroxisome F1'}")
    print(f"{'-'*70}")
    print(f"{'Baseline v5 (Day1参照)':<25}{BASELINE_V5['accuracy']:<12.4f}{BASELINE_V5['macro_f1']:<12.4f}"
          f"{'--':<14}{'--'}")
    for name, res in results.items():
        print(f"{name:<25}{res['accuracy']:<12.4f}{res['macro_f1']:<12.4f}"
              f"{res['per_class_f1'][4]:<14.3f}{res['per_class_f1'][7]:.3f}")
    print(f"{'='*70}")

    print("\n【Day 5 完成（已修复 collate_fn/train_dataset 传参 Bug）】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 5 行。")

# 输出
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1503.28it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# pooler.dense.bias         | MISSING    |
# pooler.dense.weight       | MISSING    |

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

# ============================================================
# 类别权重对比
# ============================================================
# 类别                       倒频率         平方根倒频率
# Cell membrane            1.028       1.014
# Cytoplasm                0.509       0.713
# Endoplasmic reticulum    1.658       1.288
# Golgi apparatus          4.106       2.026
# Lysosome/Vacuole         4.224       2.055
# Mitochondria             0.942       0.971
# Nucleus                  0.350       0.592
# Peroxisome               8.269       2.876
# Plastid                  1.925       1.387
# Extracellular            0.704       0.839

# =======================================================
# 训练：A: Baseline (无加权)
# =======================================================
#   Epoch   5/20 | Loss: 1.2194 | Val Acc: 0.5763 | Val Macro F1: 0.3812
#   Epoch  10/20 | Loss: 1.1040 | Val Acc: 0.5994 | Val Macro F1: 0.4348
#   Epoch  15/20 | Loss: 1.0192 | Val Acc: 0.6081 | Val Macro F1: 0.4202
#   Epoch  20/20 | Loss: 0.9378 | Val Acc: 0.6192 | Val Macro F1: 0.4402

# =======================================================
# 训练：B: 倒频率加权 CE
# =======================================================
#   Epoch   5/20 | Loss: 1.6117 | Val Acc: 0.5548 | Val Macro F1: 0.3630
#   Epoch  10/20 | Loss: 1.4380 | Val Acc: 0.5413 | Val Macro F1: 0.4323
#   Epoch  15/20 | Loss: 1.2907 | Val Acc: 0.5803 | Val Macro F1: 0.4402
#   Epoch  20/20 | Loss: 1.2107 | Val Acc: 0.5437 | Val Macro F1: 0.4472

# =======================================================
# 训练：C: 平方根倒频率加权 CE
# =======================================================
#   Epoch   5/20 | Loss: 1.4263 | Val Acc: 0.5723 | Val Macro F1: 0.3729
#   Epoch  10/20 | Loss: 1.2696 | Val Acc: 0.5914 | Val Macro F1: 0.4491
#   Epoch  15/20 | Loss: 1.1536 | Val Acc: 0.6161 | Val Macro F1: 0.4760
#   Epoch  20/20 | Loss: 1.0723 | Val Acc: 0.5946 | Val Macro F1: 0.4604

# ======================================================================
# Test Set 最终对比
# ======================================================================

# 【A: Baseline】Test Acc: 0.6299 | Macro F1: 0.4571
#     Lysosome/Vacuole     F1: 0.000 | Recall: 0.000
#     Peroxisome           F1: 0.000 | Recall: 0.000

# 【B: 倒频率加权】Test Acc: 0.5401 | Macro F1: 0.4306
#     Lysosome/Vacuole     F1: 0.132 | Recall: 0.269
#     Peroxisome           F1: 0.121 | Recall: 0.154

# 【C: 平方根倒频率加权】Test Acc: 0.6052 | Macro F1: 0.4672
#     Lysosome/Vacuole     F1: 0.105 | Recall: 0.077
#     Peroxisome           F1: 0.143 | Recall: 0.077
# ✅ evaluate() 输出与 sklearn 参考值一致
# 混淆矩阵对比图已保存：/home/tratr/dl-study/week13/day5/confusion_matrix_weighted_comparison.png

# ======================================================================
# 实验组                      Test Acc    Macro F1    Lysosome F1   Peroxisome F1
# ----------------------------------------------------------------------
# A: Baseline              0.6299      0.4571      0.000         0.000
# B: 倒频率加权                 0.5401      0.4306      0.132         0.121
# C: 平方根倒频率加权              0.6052      0.4672      0.105         0.143
# ======================================================================

# 【Day 5 完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 5 行。

# 1. 倒频率加权 vs 平方根倒频率加权，哪个 Macro F1 更高？ 整体 Test Acc 是否因为加权而下降（对比新 Baseline 0.6331）？下降了多少？
# 平方根倒频率加权的 Macro F1 更高（0.4672 vs 0.4306），Test Acc 因加权而下降，下降了约 2.4% （0.6052 vs 0.6331）。

# 2. 重点看 Lysosome/Vacuole 和 Peroxisome 两个类别的 F1：加权后是否有提升？
# 如果提升了，是通过"牺牲"了哪个多数类的表现换来的（看混淆矩阵哪一列/哪一行的对角线值下降了）？
# Lysosome/Vacuole 和 Peroxisome 的 F1 有所提升，牺牲了一部分的Nucleus， Cytoplasm 和 Extracellular(最多) 的表现（混淆矩阵中这些类别的对角线值下降）。

# 3. （权衡思考） 如果加权后 Macro F1 提升但 Accuracy 明显下降，
# 在实际的计算生物学场景中（比如药物靶点的亚细胞定位预测），你会更看重哪个指标？为什么？
# 在实际的计算生物学场景中，我会更看重 Macro F1 指标。因为 Macro F1 能够更好地反映模型在少数类上的表现，
# 而在药物靶点的亚细胞定位预测中，少数类（如 Lysosome/Vacuole 和 Peroxisome）可能具有重要的生物学意义。
# 即使整体准确率下降，如果模型能够更准确地识别这些关键类别，对于实际应用来说可能更有价值。
