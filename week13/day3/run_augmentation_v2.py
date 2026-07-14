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
# 1. 数据加载：支持四种增强配置
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, augment, use_sub, use_crop, batch_size=32):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 训练 / 评估（与之前完全一致）
# ══════════════════════════════════════════════════════
def train_and_evaluate(group_name, train_loader, val_loader, test_loader,
                       vocab_size, num_classes, epochs=20):
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


# ══════════════════════════════════════════════════════
# 3. 可视化：四组曲线对比
# ══════════════════════════════════════════════════════
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
    SAVE_DIR    = os.path.dirname(__file__)

    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    # (组名, augment, use_sub, use_crop)
    configs = [
        ("A: 无增强（对照）",        False, False, False),
        ("B: 仅保守替换",           True,  True,  False),
        ("C: 仅比例裁剪(0.7)",      True,  False, True),
        ("D: 替换+比例裁剪",        True,  True,  True),
    ]

    all_results = []
    for name, augment, use_sub, use_crop in configs:
        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer, augment, use_sub, use_crop, batch_size=32
        )
        result = train_and_evaluate(
            name, train_loader, val_loader, test_loader,
            vocab_size, NUM_CLASSES, epochs=EPOCHS
        )
        all_results.append(result)

    # ── 汇总 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'实验组':<25} {'Test Acc':<12} {'Macro F1'}")
    print(f"{'-'*60}")
    for res in all_results:
        print(f"{res['name']:<25} {res['test_acc']:<12.4f} {res['test_f1']:.4f}")
    print(f"{'-'*60}")
    print(f"{'Day1 Baseline(记录)':<25} {0.6124:<12.4f} {0.4527:.4f}")
    print(f"{'='*60}")

    plot_comparison(all_results, SAVE_DIR)

    print("\n【Day 3 调优实验完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 3 行。")

# 输出
# Using device: cuda

# =======================================================
# 实验组：A: 无增强（对照）
# =======================================================
#   Epoch   5/20 | Loss: 1.2137 | Val Acc: 0.5723 | Val Macro F1: 0.3705
#   Epoch  10/20 | Loss: 1.0917 | Val Acc: 0.5835 | Val Macro F1: 0.3941
#   Epoch  15/20 | Loss: 1.0130 | Val Acc: 0.5803 | Val Macro F1: 0.4586
#   Epoch  20/20 | Loss: 0.9607 | Val Acc: 0.5930 | Val Macro F1: 0.4136

#   【A: 无增强（对照） 最终结果】 Test Acc: 0.6219 | Macro F1: 0.4421


# =======================================================
# 实验组：B: 仅保守替换
# =======================================================
#   Epoch   5/20 | Loss: 1.2417 | Val Acc: 0.5533 | Val Macro F1: 0.3942
#   Epoch  10/20 | Loss: 1.1277 | Val Acc: 0.6017 | Val Macro F1: 0.4313
#   Epoch  15/20 | Loss: 1.0657 | Val Acc: 0.5978 | Val Macro F1: 0.4566
#   Epoch  20/20 | Loss: 1.0071 | Val Acc: 0.6184 | Val Macro F1: 0.4467

#   【B: 仅保守替换 最终结果】 Test Acc: 0.6267 | Macro F1: 0.4509


# =======================================================
# 实验组：C: 仅比例裁剪(0.7)
# =======================================================
#   Epoch   5/20 | Loss: 1.4720 | Val Acc: 0.5286 | Val Macro F1: 0.3430
#   Epoch  10/20 | Loss: 1.3816 | Val Acc: 0.5437 | Val Macro F1: 0.3638
#   Epoch  15/20 | Loss: 1.3197 | Val Acc: 0.5827 | Val Macro F1: 0.3785
#   Epoch  20/20 | Loss: 1.2598 | Val Acc: 0.5771 | Val Macro F1: 0.4025

#   【C: 仅比例裁剪(0.7) 最终结果】 Test Acc: 0.6037 | Macro F1: 0.4210


# =======================================================
# 实验组：D: 替换+比例裁剪
# =======================================================
#   Epoch   5/20 | Loss: 1.4863 | Val Acc: 0.5183 | Val Macro F1: 0.3181
#   Epoch  10/20 | Loss: 1.4095 | Val Acc: 0.5445 | Val Macro F1: 0.3657
#   Epoch  15/20 | Loss: 1.3601 | Val Acc: 0.5278 | Val Macro F1: 0.3753
#   Epoch  20/20 | Loss: 1.2994 | Val Acc: 0.5811 | Val Macro F1: 0.3988

#   【D: 替换+比例裁剪 最终结果】 Test Acc: 0.6037 | Macro F1: 0.4131

# ============================================================
# 实验组                       Test Acc     Macro F1
# ------------------------------------------------------------
# A: 无增强（对照）                0.6219       0.4421
# B: 仅保守替换                  0.6267       0.4509
# C: 仅比例裁剪(0.7)             0.6037       0.4210
# D: 替换+比例裁剪                0.6037       0.4131
# ------------------------------------------------------------
# Day1 Baseline(记录)         0.6124       0.4527
# ============================================================

# 对比图已保存：/home/tratr/dl-study/week13/day3/augmentation_ablation_v2.png

# 【Day 3 调优实验完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 3 行。

# 1. 对比 A/B/C/D 四组，性能排序是怎样的？ 这个排序是否符合你的预期？
# B > A > C = D，不算符合

# 2. 修复后的组 D（替换+比例裁剪）相比组 A（无增强）表现如何？
# 如果仍然没有超过 Baseline，可能还需要调整什么超参数？
# 仍然没有超过baseline，可能需要调整的超参数包括：
# - 替换概率 sub_prob
# - 裁剪最小比例 crop_min_len_ratio

# 3. （迁移思考） 如果组 C（仅裁剪，比例版）明显优于上次的负结果，
# 这说明关于"固定长度裁剪 vs 比例裁剪"的教训，
# 在你未来处理其他生物序列任务（比如 DNA/RNA 序列）时应该怎么应用？
# 应该优先考虑比例裁剪，避免固定长度裁剪对不同长度的序列造成信息丢失。