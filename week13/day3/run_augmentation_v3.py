import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random                          # ← 新增
import matplotlib
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import sys, os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # ← 新增，必须在其他CUDA操作之前
torch.use_deterministic_algorithms(True)             # ← 新增

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier
from augmentation import AugmentedProteinDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 官方基准（Day1 修复种子后重跑的 Baseline v5）
BASELINE_V5 = {'accuracy': 0.6481334392374901, 'macro_f1': 0.46426376359159355}   # ← 修正


# ══════════════════════════════════════════════════════
# 0.【新增】全局种子固定函数
# ══════════════════════════════════════════════════════
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_seed] 全局随机种子已固定为 {seed}")


# ══════════════════════════════════════════════════════
# 1. 数据加载：支持四种增强配置（← 新增 seed 参数）
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, augment, use_sub, use_crop, batch_size=32, seed=42):
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)

    train_dataset = AugmentedProteinDataset(
        train_dataset, augment=augment,
        use_substitution=use_sub, use_crop=use_crop,
        sub_prob=0.1, crop_min_len_ratio=0.7
    )
    val_dataset  = AugmentedProteinDataset(val_dataset,  augment=False)
    test_dataset = AugmentedProteinDataset(test_dataset, augment=False)

    collate_fn = make_collate_fn(tokenizer)

    g = torch.Generator()                        # ← 新增：控制 shuffle 顺序
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              generator=g)                          # ← 新增
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 训练 / 评估（← train_and_evaluate 增加 seed 参数）
# ══════════════════════════════════════════════════════
def train_and_evaluate(group_name, train_loader, val_loader, test_loader,
                       vocab_size, num_classes, epochs=20, seed=42):
    set_seed(seed)   # ★★★ 关键新增：每组独立重置种子，避免A→B→C→D组间随机数状态互相污染 ★★★

    print(f"\n{'='*55}")
    print(f"实验组：{group_name}")
    print(f"{'='*55}")

    model = ProteinClassifier(
        num_classes=num_classes, vocab_size=vocab_size,
        d_model=128, num_heads=4, num_layers=3,
        d_ff=512, max_len=512, dropout=0.1
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

    return {'name': group_name, 'val_accs': val_accs, 'val_f1s': val_f1s,
            'test_acc': test_acc, 'test_f1': test_f1}


def plot_comparison(results, save_dir):
    epochs = range(1, len(results[0]['val_accs']) + 1)
    colors = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('数据增强调优消融实验（自实现模型）', fontsize=13)

    for res, color in zip(results, colors):
        axes[0].plot(epochs, res['val_accs'], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs, res['val_f1s'], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)

    axes[0].set_title('Val Accuracy')
    axes[1].set_title('Val Macro F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'augmentation_ablation_v2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n对比图已保存：{save_path}")



# ══════════════════════════════════════════════════════
# 4. 主程序：四组消融
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS      = 20
    SEED        = 42                          # ← 新增
    SAVE_DIR    = os.path.dirname(__file__)

    set_seed(SEED)                            # ← 新增：主流程最外层也固定一次

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    configs = [
        ("A: 无增强（对照）",        False, False, False),
        ("B: 仅保守替换",           True,  True,  False),
        ("C: 仅比例裁剪(0.7)",      True,  False, True),
        ("D: 替换+比例裁剪",        True,  True,  True),
    ]

    all_results = []
    for name, augment, use_sub, use_crop in configs:
        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer, augment, use_sub, use_crop, batch_size=32, seed=SEED   # ← 传 seed
        )
        result = train_and_evaluate(
            name, train_loader, val_loader, test_loader,
            vocab_size, NUM_CLASSES, epochs=EPOCHS, seed=SEED                 # ← 传 seed
        )
        all_results.append(result)

    # ── 汇总 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'实验组':<25} {'Test Acc':<12} {'Macro F1'}")
    print(f"{'-'*60}")
    for res in all_results:
        print(f"{res['name']:<25} {res['test_acc']:<12.4f} {res['test_f1']:.4f}")
    print(f"{'-'*60}")
    print(f"{'Baseline v5 (Day1参照)':<25} {BASELINE_V5['accuracy']:<12.4f} {BASELINE_V5['macro_f1']:.4f}")  # ← 修正
    print(f"{'='*60}")

    plot_comparison(all_results, SAVE_DIR)

    print("\n【Day 3 数据增强实验（已固定种子重跑）完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 3 行。")

# 输出
# Using device: cuda
# [set_seed] 全局随机种子已固定为 42
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1743.55it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
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
# [set_seed] 全局随机种子已固定为 42

# =======================================================
# 实验组：A: 无增强（对照）
# =======================================================
#   Epoch   5/20 | Loss: 1.2038 | Val Acc: 0.5588 | Val Macro F1: 0.3584
#   Epoch  10/20 | Loss: 1.0768 | Val Acc: 0.6041 | Val Macro F1: 0.4332
#   Epoch  15/20 | Loss: 1.0031 | Val Acc: 0.6049 | Val Macro F1: 0.4012
#   Epoch  20/20 | Loss: 0.9302 | Val Acc: 0.6089 | Val Macro F1: 0.4532

#   【A: 无增强（对照） 最终结果】 Test Acc: 0.6481 | Macro F1: 0.4643
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
# [set_seed] 全局随机种子已固定为 42

# =======================================================
# 实验组：B: 仅保守替换
# =======================================================
#   Epoch   5/20 | Loss: 1.2332 | Val Acc: 0.5676 | Val Macro F1: 0.3742
#   Epoch  10/20 | Loss: 1.1296 | Val Acc: 0.5843 | Val Macro F1: 0.3810
#   Epoch  15/20 | Loss: 1.0637 | Val Acc: 0.5898 | Val Macro F1: 0.3906
#   Epoch  20/20 | Loss: 1.0064 | Val Acc: 0.6145 | Val Macro F1: 0.4626

#   【B: 仅保守替换 最终结果】 Test Acc: 0.6283 | Macro F1: 0.4806
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
# [set_seed] 全局随机种子已固定为 42

# =======================================================
# 实验组：C: 仅比例裁剪(0.7)
# =======================================================
#   Epoch   5/20 | Loss: 1.4501 | Val Acc: 0.5183 | Val Macro F1: 0.3279
#   Epoch  10/20 | Loss: 1.3594 | Val Acc: 0.5405 | Val Macro F1: 0.3110
#   Epoch  15/20 | Loss: 1.3147 | Val Acc: 0.5445 | Val Macro F1: 0.3584
#   Epoch  20/20 | Loss: 1.2815 | Val Acc: 0.5707 | Val Macro F1: 0.4029

#   【C: 仅比例裁剪(0.7) 最终结果】 Test Acc: 0.6021 | Macro F1: 0.4116
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
# [set_seed] 全局随机种子已固定为 42

# =======================================================
# 实验组：D: 替换+比例裁剪
# =======================================================
#   Epoch   5/20 | Loss: 1.4820 | Val Acc: 0.5302 | Val Macro F1: 0.3351
#   Epoch  10/20 | Loss: 1.3874 | Val Acc: 0.5262 | Val Macro F1: 0.3154
#   Epoch  15/20 | Loss: 1.3495 | Val Acc: 0.5453 | Val Macro F1: 0.3595
#   Epoch  20/20 | Loss: 1.2978 | Val Acc: 0.5533 | Val Macro F1: 0.3785

#   【D: 替换+比例裁剪 最终结果】 Test Acc: 0.5894 | Macro F1: 0.3973

# ============================================================
# 实验组                       Test Acc     Macro F1
# ------------------------------------------------------------
# A: 无增强（对照）                0.6481       0.4643
# B: 仅保守替换                  0.6283       0.4806
# C: 仅比例裁剪(0.7)             0.6021       0.4116
# D: 替换+比例裁剪                0.5894       0.3973
# ------------------------------------------------------------
# Baseline v5 (Day1参照)      0.6481       0.4643
# ============================================================

# 对比图已保存：/home/tratr/dl-study/week13/day3/augmentation_ablation_v2.png

# 【Day 3 数据增强实验（已固定种子重跑）完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 3 行。
