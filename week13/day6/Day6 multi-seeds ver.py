"""
Day 6 补充实验：多种子稳健性验证
------------------------------------------------------------
验证 Day6 两个候选配置的结论是否稳健（而非单一种子的噪声）：
  A. Dropout=0.1（无LS）      —— Day6中Macro F1最高
  B. Dropout=0.1+LabelSmoothing0.1 —— Day6中Test Acc最高

方法论要点：
  - train/val/test 数据切分固定不变（get_dataloaders 只调用一次，seed=42）
  - 仅"模型初始化"与"训练时DataLoader shuffle顺序"随种子变化
  - 每个配置独立跑 3 个种子，报告 mean ± std
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week13', 'day5'))
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier
from weighted_loss_experiment import get_dataloaders

torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# 基础工具函数（与 Day6 完全一致，直接复用）
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_seed] 全局随机种子已固定为 {seed}")


def build_fresh_train_loader(train_dataset, collate_fn, batch_size, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=g
    )


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            preds = model(input_ids, mask).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.best_epoch = 0
        self.early_stop = False

    def step(self, score, model, epoch):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def restore_best(self, model):
        model.load_state_dict(self.best_state)
        return model


def train_experiment(tag, dropout, train_dataset, collate_fn, batch_size,
                      val_loader, test_loader, vocab_size, num_classes,
                      max_epochs=30, patience=5, seed=42,
                      label_smoothing=0.0, verbose=True):

    set_seed(seed)
    train_loader = build_fresh_train_loader(train_dataset, collate_fn, batch_size, seed=seed)

    model = ProteinClassifier(
        num_classes=num_classes, vocab_size=vocab_size,
        d_model=128, num_heads=4, num_layers=3, d_ff=512,
        max_len=512, dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=patience)

    if verbose:
        print(f"\n{'='*60}\n训练组：{tag} (seed={seed})\n{'='*60}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        for input_ids, mask, labels in train_loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, mask), labels)
            loss.backward()
            optimizer.step()

        _, val_f1 = evaluate_model(model, val_loader, device)
        is_best = early_stopper.step(val_f1, model, epoch)

        if verbose and (is_best or early_stopper.early_stop):
            marker = " ← 新最优" if is_best else ""
            print(f"  Epoch {epoch:3d} | Val Macro F1:{val_f1:.4f}{marker}")

        if early_stopper.early_stop:
            if verbose:
                print(f"  🛑 Early Stop @ Epoch {epoch}（最优 Epoch {early_stopper.best_epoch}）")
            break

    model = early_stopper.restore_best(model)
    test_acc, test_f1 = evaluate_model(model, test_loader, device)

    if verbose:
        print(f"  【结果】Best Epoch={early_stopper.best_epoch} | Test Acc={test_acc:.4f} | Macro F1={test_f1:.4f}")

    return {
        'tag': tag, 'seed': seed, 'best_epoch': early_stopper.best_epoch,
        'test_acc': test_acc, 'test_f1': test_f1
    }


# ============================================================
# 主流程：多种子验证
# ============================================================

SEEDS = [42, 114514, 1919810]
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
NUM_CLASSES = 10

tokenizer, _ = load_esm2()
vocab_size = len(tokenizer)

# ⚠️ 关键：数据切分固定用 seed=42，与训练种子循环无关，
# 确保三次实验的 train/val/test 划分完全一致
train_loader, val_loader, test_loader, train_dataset, collate_fn = get_dataloaders(
    tokenizer, batch_size=BATCH_SIZE, seed=42
)

CONFIGS = [
    {'name': 'A. Dropout=0.1（无LS）',        'dropout': 0.1, 'label_smoothing': 0.0},
    {'name': 'B. Dropout=0.1+LabelSmoothing0.1', 'dropout': 0.1, 'label_smoothing': 0.1},
]

all_results = []

for cfg in CONFIGS:
    print(f"\n{'#'*70}\n# 配置：{cfg['name']}\n{'#'*70}")
    for seed in SEEDS:
        r = train_experiment(
            tag=cfg['name'], dropout=cfg['dropout'],
            train_dataset=train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE,
            val_loader=val_loader, test_loader=test_loader,
            vocab_size=vocab_size, num_classes=NUM_CLASSES,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
            label_smoothing=cfg['label_smoothing'], verbose=True
        )
        all_results.append(r)

# ============================================================
# 汇总统计：mean ± std
# ============================================================

df = pd.DataFrame(all_results)
print("\n" + "="*70)
print("原始逐次结果")
print("="*70)
print(df.to_string(index=False))

summary = df.groupby('tag').agg(
    acc_mean=('test_acc', 'mean'),
    acc_std=('test_acc', 'std'),
    f1_mean=('test_f1', 'mean'),
    f1_std=('test_f1', 'std'),
    epoch_mean=('best_epoch', 'mean'),
).reset_index()

print("\n" + "="*70)
print("多种子汇总（mean ± std，n=3）")
print("="*70)
for _, row in summary.iterrows():
    print(f"\n{row['tag']}")
    print(f"  Test Acc : {row['acc_mean']:.4f} ± {row['acc_std']:.4f}")
    print(f"  Macro F1 : {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
    print(f"  平均最优Epoch: {row['epoch_mean']:.1f}")

# 与 Baseline v5 对比
BASELINE_V5 = {'test_acc': 0.6481334392374901, 'test_f1': 0.46426376359159355}
print("\n" + "="*70)
print(f"参照：Baseline v5 (单次, 无EarlyStop) Acc={BASELINE_V5['test_acc']:.4f} | F1={BASELINE_V5['test_f1']:.4f}")
print("="*70)

# 统计显著性提示（简易版：均值差是否大于两组std之和，粗略判断）
a_row = summary[summary['tag'] == CONFIGS[0]['name']].iloc[0]
b_row = summary[summary['tag'] == CONFIGS[1]['name']].iloc[0]
f1_diff = abs(a_row['f1_mean'] - b_row['f1_mean'])
f1_combined_std = a_row['f1_std'] + b_row['f1_std']
print(f"\nA/B两组 Macro F1 均值差 = {f1_diff:.4f}，两组std之和 = {f1_combined_std:.4f}")
if f1_diff < f1_combined_std:
    print("⚠️ 均值差小于std之和 → 差异可能不稳健，不能排除是种子噪声导致的表面差异")
else:
    print("✅ 均值差大于std之和 → 差异相对稳健，大概率是真实效应而非噪声")

df.to_csv('week13/day6/day6_seed_robustness_raw.csv', index=False)
summary.to_csv('week13/day6/day6_seed_robustness_summary.csv', index=False)
print("\n已保存：week13/day6/day6_seed_robustness_raw.csv / week13/day6/day6_seed_robustness_summary.csv")

# 输出
# Using device: cuda
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1662.08it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
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

# ######################################################################
# # 配置：A. Dropout=0.1（无LS）
# ######################################################################

# ============================================================
# 训练组：A. Dropout=0.1（无LS） (seed=42)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2339 ← 新最优
#   Epoch   2 | Val Macro F1:0.3384 ← 新最优
#   Epoch   3 | Val Macro F1:0.3411 ← 新最优
#   Epoch   5 | Val Macro F1:0.3584 ← 新最优
#   Epoch   6 | Val Macro F1:0.4172 ← 新最优
#   Epoch  10 | Val Macro F1:0.4332 ← 新最优
#   Epoch  11 | Val Macro F1:0.4799 ← 新最优
#   Epoch  16 | Val Macro F1:0.4758
#   🛑 Early Stop @ Epoch 16（最优 Epoch 11）
#   【结果】Best Epoch=11 | Test Acc=0.6299 | Macro F1=0.4890

# ============================================================
# 训练组：A. Dropout=0.1（无LS） (seed=114514)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2147 ← 新最优
#   Epoch   2 | Val Macro F1:0.3172 ← 新最优
#   Epoch   4 | Val Macro F1:0.3721 ← 新最优
#   Epoch   5 | Val Macro F1:0.3928 ← 新最优
#   Epoch   6 | Val Macro F1:0.4026 ← 新最优
#   Epoch  10 | Val Macro F1:0.4062 ← 新最优
#   Epoch  11 | Val Macro F1:0.4353 ← 新最优
#   Epoch  14 | Val Macro F1:0.4407 ← 新最优
#   Epoch  17 | Val Macro F1:0.4582 ← 新最优
#   Epoch  22 | Val Macro F1:0.4619 ← 新最优
#   Epoch  24 | Val Macro F1:0.4650 ← 新最优
#   Epoch  25 | Val Macro F1:0.4714 ← 新最优
#   Epoch  27 | Val Macro F1:0.4866 ← 新最优
#   Epoch  30 | Val Macro F1:0.5018 ← 新最优
#   【结果】Best Epoch=30 | Test Acc=0.6442 | Macro F1=0.5071

# ============================================================
# 训练组：A. Dropout=0.1（无LS） (seed=1919810)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2424 ← 新最优
#   Epoch   2 | Val Macro F1:0.2555 ← 新最优
#   Epoch   3 | Val Macro F1:0.3690 ← 新最优
#   Epoch   5 | Val Macro F1:0.3949 ← 新最优
#   Epoch   6 | Val Macro F1:0.4171 ← 新最优
#   Epoch   8 | Val Macro F1:0.4249 ← 新最优
#   Epoch   9 | Val Macro F1:0.4291 ← 新最优
#   Epoch  14 | Val Macro F1:0.4457 ← 新最优
#   Epoch  15 | Val Macro F1:0.4558 ← 新最优
#   Epoch  17 | Val Macro F1:0.4662 ← 新最优
#   Epoch  19 | Val Macro F1:0.4987 ← 新最优
#   Epoch  24 | Val Macro F1:0.4436
#   🛑 Early Stop @ Epoch 24（最优 Epoch 19）
#   【结果】Best Epoch=19 | Test Acc=0.6307 | Macro F1=0.4741

# ######################################################################
# # 配置：B. Dropout=0.1+LabelSmoothing0.1
# ######################################################################

# ============================================================
# 训练组：B. Dropout=0.1+LabelSmoothing0.1 (seed=42)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2270 ← 新最优
#   Epoch   2 | Val Macro F1:0.3287 ← 新最优
#   Epoch   4 | Val Macro F1:0.3324 ← 新最优
#   Epoch   5 | Val Macro F1:0.3631 ← 新最优
#   Epoch   6 | Val Macro F1:0.4200 ← 新最优
#   Epoch  11 | Val Macro F1:0.4374 ← 新最优
#   Epoch  16 | Val Macro F1:0.4458 ← 新最优
#   Epoch  17 | Val Macro F1:0.4607 ← 新最优
#   Epoch  22 | Val Macro F1:0.4598
#   🛑 Early Stop @ Epoch 22（最优 Epoch 17）
#   【结果】Best Epoch=17 | Test Acc=0.6521 | Macro F1=0.4827

# ============================================================
# 训练组：B. Dropout=0.1+LabelSmoothing0.1 (seed=114514)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2059 ← 新最优
#   Epoch   2 | Val Macro F1:0.3307 ← 新最优
#   Epoch   4 | Val Macro F1:0.3669 ← 新最优
#   Epoch   5 | Val Macro F1:0.3886 ← 新最优
#   Epoch   6 | Val Macro F1:0.3952 ← 新最优
#   Epoch   7 | Val Macro F1:0.4170 ← 新最优
#   Epoch  10 | Val Macro F1:0.4540 ← 新最优
#   Epoch  13 | Val Macro F1:0.4630 ← 新最优
#   Epoch  18 | Val Macro F1:0.4550
#   🛑 Early Stop @ Epoch 18（最优 Epoch 13）
#   【结果】Best Epoch=13 | Test Acc=0.6140 | Macro F1=0.4256

# ============================================================
# 训练组：B. Dropout=0.1+LabelSmoothing0.1 (seed=1919810)
# ============================================================
#   Epoch   1 | Val Macro F1:0.2324 ← 新最优
#   Epoch   2 | Val Macro F1:0.2444 ← 新最优
#   Epoch   3 | Val Macro F1:0.3480 ← 新最优
#   Epoch   4 | Val Macro F1:0.3800 ← 新最优
#   Epoch   5 | Val Macro F1:0.3855 ← 新最优
#   Epoch   6 | Val Macro F1:0.4221 ← 新最优
#   Epoch   8 | Val Macro F1:0.4255 ← 新最优
#   Epoch  13 | Val Macro F1:0.4040
#   🛑 Early Stop @ Epoch 13（最优 Epoch 8）
#   【结果】Best Epoch=8 | Test Acc=0.6172 | Macro F1=0.4357

# ======================================================================
# 原始逐次结果
# ======================================================================
#                              tag    seed  best_epoch  test_acc  test_f1
#              A. Dropout=0.1（无LS）      42          11  0.629865 0.489038
#              A. Dropout=0.1（无LS）  114514          30  0.644162 0.507123
#              A. Dropout=0.1（无LS） 1919810          19  0.630659 0.474118
# B. Dropout=0.1+LabelSmoothing0.1      42          17  0.652105 0.482698
# B. Dropout=0.1+LabelSmoothing0.1  114514          13  0.613979 0.425603
# B. Dropout=0.1+LabelSmoothing0.1 1919810           8  0.617156 0.435748

# ======================================================================
# 多种子汇总（mean ± std，n=3）
# ======================================================================

# A. Dropout=0.1（无LS）
#   Test Acc : 0.6349 ± 0.0080
#   Macro F1 : 0.4901 ± 0.0165
#   平均最优Epoch: 20.0

# B. Dropout=0.1+LabelSmoothing0.1
#   Test Acc : 0.6277 ± 0.0212
#   Macro F1 : 0.4480 ± 0.0305
#   平均最优Epoch: 12.7

# ======================================================================
# 参照：Baseline v5 (单次, 无EarlyStop) Acc=0.6481 | F1=0.4643
# ======================================================================

# A/B两组 Macro F1 均值差 = 0.0421，两组std之和 = 0.0470
# ⚠️ 均值差小于std之和 → 差异可能不稳健，不能排除是种子噪声导致的表面差异

# 已保存：week13/day6/day6_seed_robustness_raw.csv / week13/day6/day6_seed_robustness_summary.csv
