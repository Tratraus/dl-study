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


# 输出
# Using device: cuda

# =======================================================
# 实验组：无增强（对照）
# =======================================================
#   Epoch   5/20 | Loss: 1.2094 | Val Acc: 0.5843 | Val Macro F1: 0.3957
#   Epoch  10/20 | Loss: 1.1018 | Val Acc: 0.5787 | Val Macro F1: 0.3960
#   Epoch  15/20 | Loss: 1.0188 | Val Acc: 0.6113 | Val Macro F1: 0.4268
#   Epoch  20/20 | Loss: 0.9334 | Val Acc: 0.6200 | Val Macro F1: 0.4788

#   【无增强（对照） 最终结果】 Test Acc: 0.6338 | Macro F1: 0.4708

# =======================================================
# 实验组：数据增强
# =======================================================
#   Epoch   5/20 | Loss: 1.6779 | Val Acc: 0.4300 | Val Macro F1: 0.2341
#   Epoch  10/20 | Loss: 1.6071 | Val Acc: 0.5032 | Val Macro F1: 0.3217
#   Epoch  15/20 | Loss: 1.5707 | Val Acc: 0.5207 | Val Macro F1: 0.3408
#   Epoch  20/20 | Loss: 1.5413 | Val Acc: 0.5445 | Val Macro F1: 0.3312

#   【数据增强 最终结果】 Test Acc: 0.5536 | Macro F1: 0.3504

# =======================================================
# 实验组             Test Acc     Macro F1
# -------------------------------------------------------
# 无增强（对照）         0.6338       0.4708
# 数据增强            0.5536       0.3504
# -------------------------------------------------------
# Day1 Baseline(记录) 0.6124       0.4527
# =======================================================

# 对比图已保存：/home/tratr/dl-study/week13/day3/augmentation_comparison.png

# 【Day 3 完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 3 行。

# 1. 数据增强为什么只在训练集上用，不在测试集上用？
# 因为数据增强本质是构建一套新的训练样本，目的是让模型在训练阶段学习到更多的特征和模式，从而提高模型的泛化能力。
# 而测试集的目的是评估模型在未见过的数据上的表现，如果在测试集上使用数据增强，
# 会改变测试数据的分布，使得评估结果不再具有可比性。
# 因此，数据增强只应用于训练集，而测试集保持原始分布，以便准确评估模型性能。
# 2. 保守替换和随机替换有什么本质区别？ 为什么保守替换更合理？
# 保守替换一般不会改变该氨基酸附近高级结构的稳定性和功能性，而随机替换可能会引入不合理的氨基酸，破坏蛋白质的结构和功能。
# 因此，保守替换更合理，因为它在增强数据的同时，尽量保持蛋白质的生物学特性，从而提高模型在真实数据上的泛化能力。

# 3. （观察题） 增强后的 Test Acc/F1 相比无增强组有提升吗？
# 如果提升有限甚至下降，可能是什么原因
# （提示：想想 sub_prob=0.1 和 crop_min_len=30 这两个超参数是否合适，以及模型本身参数量只有 600K）？
# 增强后的 Test Acc/F1 相比无增强组没有提升，反而下降了。这可能是由于以下原因：
# 1. sub_prob=0.1 的保守替换概率可能过高，导致训练数据中引入了过多的噪声，使模型难以学习到有效的特征。
# 2. crop_min_len=30 的裁剪长度可能过短，导致训练数据中丢失了重要的序列信息，使模型无法捕捉到完整的蛋白质特征。
# 3. 模型本身参数量只有 600K，可能容量不足以有效地学习增强后的复杂数据分布，从而导致性能下降。
# 4. 数据增强的方式可能不适合当前任务，可能需要尝试其他增强方法或调整超参数，以找到更适合的增强策略，从而提升模型性能。