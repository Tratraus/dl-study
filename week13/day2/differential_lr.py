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
import matplotlib.gridspec as gridspec
import numpy as np
import sys, os

# ── 路径 ──────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))

from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════
# 1. 数据加载（复用 Day 1）
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, batch_size=32):
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 核心：差异化学习率设置
# ══════════════════════════════════════════════════════
def get_param_groups(esm2_model, classifier_head, backbone_lr, head_lr):
    """
    将 backbone 和 head 的参数分成两个 group，分别设置学习率。

    Args:
        esm2_model:      ESM-2 backbone
        classifier_head: 线性分类头
        backbone_lr:     backbone 的学习率
        head_lr:         分类头的学习率

    Returns:
        param_groups: list of dict，可直接传给 optimizer
    """
    backbone_params = [p for p in esm2_model.parameters() if p.requires_grad]
    head_params     = list(classifier_head.parameters())

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head_params,     'lr': head_lr,     'name': 'head'},
    ]

    # ── 验证输出（每次调用都打印，方便确认）──────────
    backbone_count = sum(p.numel() for p in backbone_params)
    head_count     = sum(p.numel() for p in head_params)
    print(f"  backbone: {backbone_count:,} params, lr={backbone_lr}")
    print(f"  head    : {head_count:,} params, lr={head_lr}")

    return param_groups


def build_finetune_model(num_classes):
    """构建 Fine-tune 模型（backbone 全部可训练）"""
    tokenizer, esm2_model = load_esm2()
    esm2_model = esm2_model.to(device)
    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)
    return esm2_model, classifier_head, tokenizer


# ══════════════════════════════════════════════════════
# 3. 训练 / 评估函数
# ══════════════════════════════════════════════════════
def get_esm2_embedding(esm2_model, input_ids, attention_mask):
    """Fine-tune 模式：backbone 参与梯度计算"""
    from protein_dataset import mean_pooling_with_mask
    outputs = esm2_model(input_ids=input_ids, attention_mask=attention_mask)
    return mean_pooling_with_mask(outputs.last_hidden_state, attention_mask)


def train_one_epoch(esm2_model, classifier_head, optimizer, criterion, train_loader):
    esm2_model.train()
    classifier_head.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        embeddings = get_esm2_embedding(esm2_model, input_ids, attention_mask)
        logits     = classifier_head(embeddings)
        loss       = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(esm2_model, classifier_head, loader):
    esm2_model.eval()
    classifier_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)
            embeddings = get_esm2_embedding(esm2_model, input_ids, attention_mask)
            preds      = classifier_head(embeddings).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1


# ══════════════════════════════════════════════════════
# 4. 单组实验
# ══════════════════════════════════════════════════════
def run_experiment(group_name, backbone_lr, head_lr,
                   train_loader, val_loader, test_loader,
                   num_classes, epochs=10):
    """
    跑一组差异化学习率实验，返回训练曲线数据
    """
    print(f"\n{'='*55}")
    print(f"实验组：{group_name}  "
          f"(backbone_lr={backbone_lr}, head_lr={head_lr})")
    print(f"{'='*55}")

    esm2_model, classifier_head, _ = build_finetune_model(num_classes)
    param_groups = get_param_groups(esm2_model, classifier_head,
                                    backbone_lr, head_lr)

    # ── 验证 lr 设置是否正确 ──────────────────────────
    optimizer = torch.optim.Adam(param_groups)
    for g in optimizer.param_groups:
        print(f"  [验证] group '{g['name']}': lr = {g['lr']}")

    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs, val_f1s = [], [], []

    for epoch in range(1, epochs + 1):
        train_loss       = train_one_epoch(esm2_model, classifier_head,
                                           optimizer, criterion, train_loader)
        val_acc, val_f1  = evaluate(esm2_model, classifier_head, val_loader)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    # ── 测试集最终结果 ────────────────────────────────
    test_acc, test_f1 = evaluate(esm2_model, classifier_head, test_loader)
    print(f"\n  【{group_name} 最终结果】"
          f" Test Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    return {
        'name'        : group_name,
        'backbone_lr' : backbone_lr,
        'head_lr'     : head_lr,
        'train_losses': train_losses,
        'val_accs'    : val_accs,
        'val_f1s'     : val_f1s,
        'test_acc'    : test_acc,
        'test_f1'     : test_f1,
    }


# ══════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════
def plot_results(results, save_dir):
    """
    画三张图：
    1. 训练 Loss 曲线对比
    2. Val Acc 曲线对比
    3. Val Macro F1 曲线对比
    """
    epochs = range(1, len(results[0]['train_losses']) + 1)
    colors = ['#e74c3c', '#2ecc71', '#3498db']   # 红/绿/蓝

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('差异化学习率实验对比（ESM-2 Fine-tune）', fontsize=13)

    metrics = [
        ('train_losses', 'Train Loss',     axes[0]),
        ('val_accs',     'Val Accuracy',   axes[1]),
        ('val_f1s',      'Val Macro F1',   axes[2]),
    ]

    for key, ylabel, ax in metrics:
        for res, color in zip(results, colors):
            ax.plot(epochs, res[key], label=res['name'],
                    color=color, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'differential_lr_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n训练曲线已保存：{save_path}")


def print_summary_table(results):
    """打印汇总表"""
    print(f"\n{'='*65}")
    print(f"{'实验组':<20} {'backbone_lr':<14} {'head_lr':<10} "
          f"{'Test Acc':<12} {'Macro F1'}")
    print(f"{'-'*65}")
    for res in results:
        print(f"{res['name']:<20} {res['backbone_lr']:<14} "
              f"{res['head_lr']:<10} {res['test_acc']:<12.4f} {res['test_f1']:.4f}")
    print(f"{'='*65}")

    # 找最优组
    best = max(results, key=lambda x: x['val_f1s'][-1])
    print(f"\n最优组（按最终 Val Macro F1）：{best['name']}")
    print(f"  backbone_lr / head_lr = {best['backbone_lr']} / {best['head_lr']}"
          f"  比例 = 1:{int(best['head_lr']/best['backbone_lr'])}")


# ══════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    NUM_CLASSES = 10
    EPOCHS      = 10   # 10 epoch 足够看出差异，不需要跑太久
    SAVE_DIR    = os.path.dirname(__file__)

    # 先拿 tokenizer 构建 DataLoader
    tokenizer, _ = load_esm2()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, batch_size=32
    )

    # ── 三组实验 ──────────────────────────────────────
    experiment_configs = [
        ('A: 1:1   (lr=1e-3)',  1e-3, 1e-3),
        ('B: 1:10  (lr=1e-4)',  1e-4, 1e-3),
        ('C: 1:100 (lr=1e-5)',  1e-5, 1e-3),
    ]

    all_results = []
    for name, backbone_lr, head_lr in experiment_configs:
        result = run_experiment(
            name, backbone_lr, head_lr,
            train_loader, val_loader, test_loader,
            num_classes=NUM_CLASSES, epochs=EPOCHS
        )
        all_results.append(result)

    # ── 汇总输出 ──────────────────────────────────────
    print_summary_table(all_results)
    plot_results(all_results, SAVE_DIR)

    print("\n【Day 2 完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 2 行。")


# 输出
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 2292.68it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
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

# =======================================================
# 实验组：A: 1:1   (lr=1e-3)  (backbone_lr=0.001, head_lr=0.001)
# =======================================================
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 2415.04it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
#   backbone: 7,511,801 params, lr=0.001
#   head    : 3,210 params, lr=0.001
#   [验证] group 'backbone': lr = 0.001
#   [验证] group 'head': lr = 0.001
#   Epoch  1/10 | Loss: 1.5113 | Val Acc: 0.5477 | Val Macro F1: 0.3725
#   Epoch  2/10 | Loss: 1.2084 | Val Acc: 0.5588 | Val Macro F1: 0.3359
#   Epoch  3/10 | Loss: 1.0739 | Val Acc: 0.6010 | Val Macro F1: 0.4063
#   Epoch  4/10 | Loss: 1.0325 | Val Acc: 0.6169 | Val Macro F1: 0.4468
#   Epoch  5/10 | Loss: 0.9707 | Val Acc: 0.6288 | Val Macro F1: 0.4607
#   Epoch  6/10 | Loss: 0.9371 | Val Acc: 0.6280 | Val Macro F1: 0.4304
#   Epoch  7/10 | Loss: 0.9257 | Val Acc: 0.6264 | Val Macro F1: 0.4299
#   Epoch  8/10 | Loss: 0.8774 | Val Acc: 0.6431 | Val Macro F1: 0.4572
#   Epoch  9/10 | Loss: 0.8428 | Val Acc: 0.6455 | Val Macro F1: 0.4986
#   Epoch 10/10 | Loss: 0.8187 | Val Acc: 0.6566 | Val Macro F1: 0.5080

#   【A: 1:1   (lr=1e-3) 最终结果】 Test Acc: 0.6728 | Macro F1: 0.5374

# =======================================================
# 实验组：B: 1:10  (lr=1e-4)  (backbone_lr=0.0001, head_lr=0.001)
# =======================================================
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 2298.41it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
#   backbone: 7,511,801 params, lr=0.0001
#   head    : 3,210 params, lr=0.001
#   [验证] group 'backbone': lr = 0.0001
#   [验证] group 'head': lr = 0.001
#   Epoch  1/10 | Loss: 1.1375 | Val Acc: 0.6940 | Val Macro F1: 0.5105
#   Epoch  2/10 | Loss: 0.7904 | Val Acc: 0.7075 | Val Macro F1: 0.5463
#   Epoch  3/10 | Loss: 0.6298 | Val Acc: 0.7281 | Val Macro F1: 0.5865
#   Epoch  4/10 | Loss: 0.5003 | Val Acc: 0.7321 | Val Macro F1: 0.5892
#   Epoch  5/10 | Loss: 0.4077 | Val Acc: 0.7480 | Val Macro F1: 0.6513
#   Epoch  6/10 | Loss: 0.2771 | Val Acc: 0.7170 | Val Macro F1: 0.6170
#   Epoch  7/10 | Loss: 0.2034 | Val Acc: 0.7480 | Val Macro F1: 0.6592
#   Epoch  8/10 | Loss: 0.1477 | Val Acc: 0.7289 | Val Macro F1: 0.6464
#   Epoch  9/10 | Loss: 0.1009 | Val Acc: 0.7496 | Val Macro F1: 0.6206
#   Epoch 10/10 | Loss: 0.0860 | Val Acc: 0.7258 | Val Macro F1: 0.6313

#   【B: 1:10  (lr=1e-4) 最终结果】 Test Acc: 0.7562 | Macro F1: 0.6649

# =======================================================
# 实验组：C: 1:100 (lr=1e-5)  (backbone_lr=1e-05, head_lr=0.001)
# =======================================================
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 2293.26it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
#   backbone: 7,511,801 params, lr=1e-05
#   head    : 3,210 params, lr=0.001
#   [验证] group 'backbone': lr = 1e-05
#   [验证] group 'head': lr = 0.001
#   Epoch  1/10 | Loss: 1.3832 | Val Acc: 0.6383 | Val Macro F1: 0.4039
#   Epoch  2/10 | Loss: 0.8941 | Val Acc: 0.6741 | Val Macro F1: 0.4657
#   Epoch  3/10 | Loss: 0.7808 | Val Acc: 0.6987 | Val Macro F1: 0.5054
#   Epoch  4/10 | Loss: 0.7002 | Val Acc: 0.7178 | Val Macro F1: 0.5296
#   Epoch  5/10 | Loss: 0.6278 | Val Acc: 0.7162 | Val Macro F1: 0.5366
#   Epoch  6/10 | Loss: 0.5598 | Val Acc: 0.7345 | Val Macro F1: 0.5602
#   Epoch  7/10 | Loss: 0.4964 | Val Acc: 0.7281 | Val Macro F1: 0.5615
#   Epoch  8/10 | Loss: 0.4294 | Val Acc: 0.7369 | Val Macro F1: 0.5894
#   Epoch  9/10 | Loss: 0.3595 | Val Acc: 0.7321 | Val Macro F1: 0.6338
#   Epoch 10/10 | Loss: 0.2991 | Val Acc: 0.7258 | Val Macro F1: 0.6131

#   【C: 1:100 (lr=1e-5) 最终结果】 Test Acc: 0.7601 | Macro F1: 0.6759

# =================================================================
# 实验组                  backbone_lr    head_lr    Test Acc     Macro F1
# -----------------------------------------------------------------
# A: 1:1   (lr=1e-3)   0.001          0.001      0.6728       0.5374
# B: 1:10  (lr=1e-4)   0.0001         0.001      0.7562       0.6649
# C: 1:100 (lr=1e-5)   1e-05          0.001      0.7601       0.6759
# =================================================================

# 最优组（按最终 Val Macro F1）：B: 1:10  (lr=1e-4)
#   backbone_lr / head_lr = 0.0001 / 0.001  比例 = 1:10

# 训练曲线已保存：/home/tratr/dl-study/week13/day2/differential_lr_curves.png

# 【Day 2 完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 2 行。

# 什么是 Catastrophic Forgetting？ 在你的实验中，组 A（1:1）相比组 C（1:100）有没有 Forgetting 的迹象？怎么判断？
# Catastrophic Forgetting 是指在连续学习过程中，模型在学习新任务时遗忘了之前学习的任务的能力。
# 在你的实验中，组 A（1:1）相比组 C（1:100）有 Forgetting 的迹象。可以从以下几个方面判断：
# 1. A 组Loss持续下降，但 Val Acc 和 Val Macro F1 在中后期出现波动甚至下降，
# 说明模型在训练过程中可能过度拟合训练数据，而在验证集上的表现不稳定。
# 2. C 组在训练过程中，虽然 Loss 下降较慢，但 Val Acc 和 Val Macro F1 稳定上升，最终表现优于 A 组。

# 最优 backbone_lr / head_lr 比例是多少？ 你是怎么判断的（用 Val Acc 还是 Val Macro F1，为什么）？
# 最优的 backbone_lr / head_lr 比例是 1:10（组 B）。判断依据是最终的 Val Macro F1 值，
# 因为 Macro F1 能够更好地反映模型在各类别上的综合表现，尤其是在类别不平衡的情况下，而不仅仅依赖于准确率。
# （但根据表中数据来看，似乎是C组的Macro F1更大？）

# （观察题） 组 A 的 Loss 下降最快，但最终 Val Acc 不一定最高——如果出现这种情况，说明了什么？
# 这种情况说明了模型可能在训练集上过拟合，导致在验证集上的泛化能力不足。
# 虽然 Loss 下降快，但模型可能没有学到对验证集有用的特征，从而导致 Val Acc 不高。
# 这也强调了在模型评估中，除了关注训练 Loss，还需要关注验证集的性能指标。
