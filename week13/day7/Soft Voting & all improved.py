"""
Week13 Day7：集成与收官
------------------------------------------------------------
Phase 1：Soft Voting 集成（Baseline v5 + 加权sqrt模型 + 正则化模型，等权平均）
Phase 2：全部改进综合验证（数据增强[仅替换] + WCE(sqrt) + Dropout0.1 + EarlyStopping，3种子）
Phase 3：最终消融汇总表
------------------------------------------------------------
依赖的历史 checkpoint：
  week13/day1/baseline_checkpoint_v5.pt      (dict, key='model_state_dict')
  week13/day5/checkpoints/sqrt_inverse.pth   (裸 state_dict)
  week13/day6/day6_best_checkpoint.pt        (dict, key='model_state_dict')
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # 必须在其他CUDA操作之前

import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

torch.use_deterministic_algorithms(True)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week13', 'day3'))

from protein_dataset import make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier
from augmentation import AugmentedProteinDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CLASS_NAMES = [
    'Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
    'Golgi apparatus', 'Lysosome/Vacuole', 'Mitochondria',
    'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular'
]
NUM_CLASSES = 10
BASELINE_V5 = {'test_acc': 0.6481334392374901, 'test_f1': 0.46426376359159355}

# 历史单模型结果（Plan表记录，用于最终对比表）
HISTORY_RESULTS = {
    'Baseline v5':        {'test_acc': 0.6481, 'test_f1': 0.4643},
    '加权模型(sqrt)':      {'test_acc': 0.5997, 'test_f1': 0.4791},
    '正则化模型(单种子)':   {'test_acc': 0.6299, 'test_f1': 0.4890},
    '正则化模型(3种子均值)': {'test_acc': 0.6349, 'test_f1': 0.4901},
}


# ============================================================
# 基础工具函数
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


def build_model(dropout=0.1, vocab_size=None):
    return ProteinClassifier(
        num_classes=NUM_CLASSES, vocab_size=vocab_size,
        d_model=128, num_heads=4, num_layers=3,
        d_ff=512, max_len=512, dropout=dropout
    ).to(device)


def load_checkpoint_flexible(model, ckpt_path):
    """
    差异化加载逻辑：兼容 字典包裹(dict, key='model_state_dict') 与 裸 state_dict 两种格式。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  ✅ [字典包裹格式] 已加载: {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"  ✅ [裸 state_dict 格式] 已加载: {ckpt_path}")
    model.eval()
    return model


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


# ============================================================
# Phase 1：Soft Voting 集成
# ============================================================

def soft_voting_inference(models, loader, device, weights=None):
    """
    对多个模型的 softmax 输出做加权平均（等权 = weights=None）。
    """
    n_models = len(models)
    if weights is None:
        weights = [1.0 / n_models] * n_models

    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)

            probs_sum = None
            for model, w in zip(models, weights):
                logits = model(input_ids, mask)
                probs = F.softmax(logits, dim=-1) * w
                probs_sum = probs if probs_sum is None else probs_sum + probs

            preds = probs_sum.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return acc, f1, per_class_f1


def run_soft_voting(test_loader, vocab_size):
    print(f"\n{'#'*70}\n# Phase 1：Soft Voting 集成\n{'#'*70}")

    set_seed(42)  # 确保模型结构初始化的随机性不影响（load会覆盖权重，但保险起见统一基座）

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    model_a = build_model(dropout=0.1, vocab_size=vocab_size)
    model_a = load_checkpoint_flexible(
        model_a, os.path.join(base_dir, 'week13', 'day1', 'baseline_checkpoint_v5.pt'))

    model_b = build_model(dropout=0.1, vocab_size=vocab_size)
    model_b = load_checkpoint_flexible(
        model_b, os.path.join(base_dir, 'week13', 'day5', 'checkpoints', 'sqrt_inverse.pth'))

    model_c = build_model(dropout=0.1, vocab_size=vocab_size)
    model_c = load_checkpoint_flexible(
        model_c, os.path.join(base_dir, 'week13', 'day6', 'day6_best_checkpoint.pt'))

    models = [model_a, model_b, model_c]
    model_names = ['Baseline v5', '加权模型(sqrt)', '正则化模型']

    # 单模型独立复测（确认checkpoint加载无误，应与历史记录一致）
    print(f"\n{'-'*60}\n单模型复测（验证checkpoint加载正确性）\n{'-'*60}")
    for name, model in zip(model_names, models):
        acc, f1 = evaluate_model(model, test_loader, device)
        print(f"  {name:<15} Test Acc={acc:.4f} | Macro F1={f1:.4f}")

    # Soft Voting 等权集成
    print(f"\n{'-'*60}\nSoft Voting 集成结果（等权 1/3）\n{'-'*60}")
    ens_acc, ens_f1, per_class_f1 = soft_voting_inference(models, test_loader, device)
    print(f"  【集成】Test Acc={ens_acc:.4f} | Macro F1={ens_f1:.4f}")

    best_single_f1 = HISTORY_RESULTS['正则化模型(单种子)']['test_f1']
    gain = ens_f1 - best_single_f1
    print(f"\n  相较最优单模型(正则化模型 F1={best_single_f1:.4f})的增益: {gain:+.4f}")
    if gain > 0:
        print("  ✅ 集成带来正向增益")
    else:
        print("  ⚠️ 集成未超越最优单模型，可能是三个模型相关性过高（同架构+同数据）")

    print(f"\n  逐类别 F1（关注稀有类 Lysosome/Peroxisome）：")
    for idx in [4, 7]:
        print(f"    {CLASS_NAMES[idx]:<20} F1={per_class_f1[idx]:.3f}")

    return {'name': 'Soft Voting 集成', 'test_acc': ens_acc, 'test_f1': ens_f1}


# ============================================================
# Phase 2：全部改进 综合验证（数据增强[仅替换]+WCE(sqrt)+Dropout0.1+EarlyStop）
# ============================================================

def compute_class_weights_sqrt_inverse(labels, num_classes):
    labels = np.array(labels)
    counts = np.array([np.sum(labels == i) for i in range(num_classes)])
    N, K = len(labels), num_classes
    weights = np.sqrt(N / (K * counts))
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders_full(tokenizer, batch_size=32, seed=42):
    """
    Day7 融合版数据管线：
      - WCE(sqrt) 权重：在【未增强】的原始 train_dataset 上计算 label 分布
      - 数据增强：仅用保守替换(use_sub=True)，裁剪已被Day3验证为负增益，排除(use_crop=False)
    """
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)

    # 先在原始（未增强）train_dataset上取label，计算WCE权重
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_weights = compute_class_weights_sqrt_inverse(train_labels, NUM_CLASSES)

    # 再包一层增强（仅替换）
    train_dataset = AugmentedProteinDataset(
        train_dataset, augment=True,
        use_substitution=True, use_crop=False,
        sub_prob=0.1
    )
    val_dataset  = AugmentedProteinDataset(val_dataset,  augment=False)
    test_dataset = AugmentedProteinDataset(test_dataset, augment=False)

    collate_fn = make_collate_fn(tokenizer)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=g)
    val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset, collate_fn, class_weights


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


def train_full_experiment(train_dataset, collate_fn, class_weights, batch_size,
                           val_loader, test_loader, vocab_size,
                           max_epochs=30, patience=5, seed=42, tag="全部改进"):

    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=g)

    model = build_model(dropout=0.1, vocab_size=vocab_size)
    weight_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=patience)

    print(f"\n{'='*60}\n训练组：{tag} (seed={seed})\n{'='*60}")

    for epoch in range(1, max_epochs + 1):
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

        avg_loss = total_loss / len(train_loader)
        _, val_f1 = evaluate_model(model, val_loader, device)
        is_best = early_stopper.step(val_f1, model, epoch)

        if epoch % 2 == 0 or epoch == 1 or is_best or early_stopper.early_stop:
            marker = " ← 新最优" if is_best else ""
            print(f"  Epoch {epoch:3d}/{max_epochs} | Loss:{avg_loss:.4f} | Val Macro F1:{val_f1:.4f}{marker}")

        if early_stopper.early_stop:
            print(f"  🛑 Early Stop @ Epoch {epoch}（最优 Epoch {early_stopper.best_epoch}）")
            break

    model = early_stopper.restore_best(model)
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    print(f"  【结果】Best Epoch={early_stopper.best_epoch} | Test Acc={test_acc:.4f} | Macro F1={test_f1:.4f}")

    return {'tag': tag, 'seed': seed, 'best_epoch': early_stopper.best_epoch,
            'test_acc': test_acc, 'test_f1': test_f1, 'model': model}


def run_full_improvement(tokenizer, vocab_size, seeds=(42, 114514, 1919810)):
    print(f"\n{'#'*70}\n# Phase 2：全部改进 综合验证（3种子）\n{'#'*70}")

    # 数据切分固定 seed=42，只有模型初始化/shuffle顺序随种子变化
    train_loader0, val_loader, test_loader, train_dataset, collate_fn, class_weights = \
        get_dataloaders_full(tokenizer, batch_size=32, seed=42)

    all_results = []
    for seed in seeds:
        r = train_full_experiment(
            train_dataset, collate_fn, class_weights, batch_size=32,
            val_loader=val_loader, test_loader=test_loader, vocab_size=vocab_size,
            max_epochs=30, patience=5, seed=seed, tag="全部改进(增强+WCE+Dropout+EarlyStop)"
        )
        all_results.append(r)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in all_results])
    print(f"\n{'='*60}\n全部改进 - 3种子原始结果\n{'='*60}")
    print(df.to_string(index=False))

    f1_mean, f1_std = df['test_f1'].mean(), df['test_f1'].std()
    acc_mean, acc_std = df['test_acc'].mean(), df['test_acc'].std()
    print(f"\n汇总（mean±std, n={len(seeds)}）：")
    print(f"  Test Acc : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Macro F1 : {f1_mean:.4f} ± {f1_std:.4f}")

    # 与正则化模型(3种子)对比
    reg_f1_mean = HISTORY_RESULTS['正则化模型(3种子均值)']['test_f1']
    diff = f1_mean - reg_f1_mean
    print(f"\n  相较 正则化模型(3种子均值, F1={reg_f1_mean:.4f}) 的差异: {diff:+.4f}")
    if diff > 0:
        print("  ✅ 全部改进组合突破了单一正则化模型的极值")
    else:
        print("  ⚠️ 全部改进组合未能超越单一正则化模型，说明多个改进叠加存在收益冲突/边际递减")

    best_seed_idx = df['test_f1'].idxmax()
    best_model = all_results[best_seed_idx]['model']

    return {'name': '全部改进(3种子均值)', 'test_acc': acc_mean, 'test_f1': f1_mean,
            'test_acc_std': acc_std, 'test_f1_std': f1_std}, best_model, df


# ============================================================
# 主流程
# ============================================================

if __name__ == '__main__':
    tokenizer, _ = load_esm2()
    vocab_size = len(tokenizer)

    # test_loader 用于 Soft Voting（用纯净版split，与三个历史checkpoint训练时切分一致）
    set_seed(42)
    _, _, test_loader_pure, _, _, _ = get_dataloaders_full(tokenizer, batch_size=32, seed=42)

    # ---- Phase 1 ----
    voting_result = run_soft_voting(test_loader_pure, vocab_size)

    # ---- Phase 2 ----
    full_result, full_best_model, full_df = run_full_improvement(tokenizer, vocab_size)

    # ---- Phase 3：最终汇总 ----
    print(f"\n{'#'*70}\n# Phase 3：最终消融汇总\n{'#'*70}")
    print(f"{'实验组':<28}{'Test Acc':<12}{'Macro F1':<12}{'备注'}")
    print(f"{'-'*75}")
    print(f"{'Baseline v5':<28}{HISTORY_RESULTS['Baseline v5']['test_acc']:<12.4f}"
          f"{HISTORY_RESULTS['Baseline v5']['test_f1']:<12.4f}{'单种子':<10}")
    print(f"{'加权模型(sqrt)':<28}{HISTORY_RESULTS['加权模型(sqrt)']['test_acc']:<12.4f}"
          f"{HISTORY_RESULTS['加权模型(sqrt)']['test_f1']:<12.4f}{'单种子':<10}")
    print(f"{'正则化模型(3种子均值)':<28}{HISTORY_RESULTS['正则化模型(3种子均值)']['test_acc']:<12.4f}"
          f"{HISTORY_RESULTS['正则化模型(3种子均值)']['test_f1']:<12.4f}{'F1±0.0165':<10}")
    print(f"{'Soft Voting 集成':<28}{voting_result['test_acc']:<12.4f}"
          f"{voting_result['test_f1']:<12.4f}{'等权3模型':<10}")
    print(f"{'全部改进(3种子均值)':<28}{full_result['test_acc']:<12.4f}"
          f"{full_result['test_f1']:<12.4f}{'F1±'+format(full_result['test_f1_std'],'.4f'):<10}")
    print(f"{'='*75}")

    # 保存最终最优模型
    SAVE_DIR = os.path.dirname(__file__)
    best_overall_f1 = max(voting_result['test_f1'], full_result['test_f1'],
                          HISTORY_RESULTS['正则化模型(3种子均值)']['test_f1'])

    if full_result['test_f1'] == best_overall_f1:
        final_tag = '全部改进组合'
        torch.save({
            'model_state_dict': full_best_model.state_dict(),
            'macro_f1_mean': full_result['test_f1'],
            'macro_f1_std': full_result['test_f1_std'],
            'version': 'day7_final_full_improvement',
        }, os.path.join(SAVE_DIR, 'day7_final_model.pt'))
        print(f"\n✅ 最终推荐配置：{final_tag}（Macro F1={full_result['test_f1']:.4f}）")
    else:
        final_tag = 'Soft Voting 集成' if voting_result['test_f1'] == best_overall_f1 else '正则化模型(单一)'
        print(f"\n✅ 最终推荐配置：{final_tag}（Macro F1={best_overall_f1:.4f}）")
        print("   （该配置无需单独保存新checkpoint，直接复用对应历史产出）")

    full_df.to_csv(os.path.join(SAVE_DIR, 'day7_full_improvement_raw.csv'), index=False)
    print(f"\n已保存：day7_full_improvement_raw.csv")

    print("\n【Day 7 集成与收官完成】")
    print("  请回答输出问题后，将结果更新到 Plan 表 Day 7 行，完成 Week13 收官总结。")


# 输出
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1561.77it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# pooler.dense.bias         | MISSING    |
# pooler.dense.weight       | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# [set_seed] 全局随机种子已固定为 42
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
# # Phase 1：Soft Voting 集成
# ######################################################################
# [set_seed] 全局随机种子已固定为 42
#   ✅ [字典包裹格式] 已加载: /home/tratr/dl-study/week13/day7/../../week13/day1/baseline_checkpoint_v5.pt
#   ✅ [裸 state_dict 格式] 已加载: /home/tratr/dl-study/week13/day7/../../week13/day5/checkpoints/sqrt_inverse.pth
#   ✅ [字典包裹格式] 已加载: /home/tratr/dl-study/week13/day7/../../week13/day6/day6_best_checkpoint.pt

# ------------------------------------------------------------
# 单模型复测（验证checkpoint加载正确性）
# ------------------------------------------------------------
#   Baseline v5     Test Acc=0.6481 | Macro F1=0.4643
#   加权模型(sqrt)      Test Acc=0.5997 | Macro F1=0.4791
#   正则化模型           Test Acc=0.6299 | Macro F1=0.4890

# ------------------------------------------------------------
# Soft Voting 集成结果（等权 1/3）
# ------------------------------------------------------------
#   【集成】Test Acc=0.6481 | Macro F1=0.5117

#   相较最优单模型(正则化模型 F1=0.4890)的增益: +0.0227
#   ✅ 集成带来正向增益

#   逐类别 F1（关注稀有类 Lysosome/Peroxisome）：
#     Lysosome/Vacuole     F1=0.067
#     Peroxisome           F1=0.250

# ######################################################################
# # Phase 2：全部改进 综合验证（3种子）
# ######################################################################
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

# ============================================================
# 训练组：全部改进(增强+WCE+Dropout+EarlyStop) (seed=42)
# ============================================================
#   Epoch   1/30 | Loss:1.9643 | Val Macro F1:0.1886 ← 新最优
#   Epoch   2/30 | Loss:1.7773 | Val Macro F1:0.2746 ← 新最优
#   Epoch   3/30 | Loss:1.5937 | Val Macro F1:0.3402 ← 新最优
#   Epoch   4/30 | Loss:1.5314 | Val Macro F1:0.3387
#   Epoch   5/30 | Loss:1.4577 | Val Macro F1:0.3997 ← 新最优
#   Epoch   6/30 | Loss:1.4265 | Val Macro F1:0.4195 ← 新最优
#   Epoch   8/30 | Loss:1.3555 | Val Macro F1:0.3888
#   Epoch  10/30 | Loss:1.3109 | Val Macro F1:0.3987
#   Epoch  11/30 | Loss:1.2914 | Val Macro F1:0.4685 ← 新最优
#   Epoch  12/30 | Loss:1.2998 | Val Macro F1:0.4486
#   Epoch  13/30 | Loss:1.2574 | Val Macro F1:0.4918 ← 新最优
#   Epoch  14/30 | Loss:1.2383 | Val Macro F1:0.4517
#   Epoch  16/30 | Loss:1.2237 | Val Macro F1:0.4504
#   Epoch  18/30 | Loss:1.1952 | Val Macro F1:0.4764
#   🛑 Early Stop @ Epoch 18（最优 Epoch 13）
#   【结果】Best Epoch=13 | Test Acc=0.6299 | Macro F1=0.4698
# [set_seed] 全局随机种子已固定为 114514

# ============================================================
# 训练组：全部改进(增强+WCE+Dropout+EarlyStop) (seed=114514)
# ============================================================
#   Epoch   1/30 | Loss:1.9632 | Val Macro F1:0.2180 ← 新最优
#   Epoch   2/30 | Loss:1.6656 | Val Macro F1:0.3340 ← 新最优
#   Epoch   3/30 | Loss:1.5577 | Val Macro F1:0.3422 ← 新最优
#   Epoch   4/30 | Loss:1.5056 | Val Macro F1:0.3767 ← 新最优
#   Epoch   5/30 | Loss:1.4471 | Val Macro F1:0.3903 ← 新最优
#   Epoch   6/30 | Loss:1.4089 | Val Macro F1:0.3962 ← 新最优
#   Epoch   7/30 | Loss:1.3736 | Val Macro F1:0.4307 ← 新最优
#   Epoch   8/30 | Loss:1.3351 | Val Macro F1:0.4187
#   Epoch  10/30 | Loss:1.3187 | Val Macro F1:0.4454 ← 新最优
#   Epoch  12/30 | Loss:1.2697 | Val Macro F1:0.4431
#   Epoch  13/30 | Loss:1.2543 | Val Macro F1:0.4725 ← 新最优
#   Epoch  14/30 | Loss:1.2274 | Val Macro F1:0.4814 ← 新最优
#   Epoch  16/30 | Loss:1.2002 | Val Macro F1:0.4247
#   Epoch  17/30 | Loss:1.1985 | Val Macro F1:0.4881 ← 新最优
#   Epoch  18/30 | Loss:1.1641 | Val Macro F1:0.4739
#   Epoch  20/30 | Loss:1.1458 | Val Macro F1:0.4972 ← 新最优
#   Epoch  22/30 | Loss:1.1042 | Val Macro F1:0.5361 ← 新最优
#   Epoch  24/30 | Loss:1.0952 | Val Macro F1:0.4971
#   Epoch  26/30 | Loss:1.0594 | Val Macro F1:0.4840
#   Epoch  27/30 | Loss:1.0463 | Val Macro F1:0.4816
#   🛑 Early Stop @ Epoch 27（最优 Epoch 22）
#   【结果】Best Epoch=22 | Test Acc=0.5997 | Macro F1=0.4940
# [set_seed] 全局随机种子已固定为 1919810

# ============================================================
# 训练组：全部改进(增强+WCE+Dropout+EarlyStop) (seed=1919810)
# ============================================================
#   Epoch   1/30 | Loss:1.9338 | Val Macro F1:0.2526 ← 新最优
#   Epoch   2/30 | Loss:1.7502 | Val Macro F1:0.2653 ← 新最优
#   Epoch   3/30 | Loss:1.5976 | Val Macro F1:0.3519 ← 新最优
#   Epoch   4/30 | Loss:1.5165 | Val Macro F1:0.3753 ← 新最优
#   Epoch   6/30 | Loss:1.4148 | Val Macro F1:0.4233 ← 新最优
#   Epoch   7/30 | Loss:1.3639 | Val Macro F1:0.4249 ← 新最优
#   Epoch   8/30 | Loss:1.3558 | Val Macro F1:0.4122
#   Epoch   9/30 | Loss:1.3137 | Val Macro F1:0.4398 ← 新最优
#   Epoch  10/30 | Loss:1.3125 | Val Macro F1:0.4275
#   Epoch  12/30 | Loss:1.2724 | Val Macro F1:0.4511 ← 新最优
#   Epoch  14/30 | Loss:1.2281 | Val Macro F1:0.4403
#   Epoch  15/30 | Loss:1.2147 | Val Macro F1:0.4520 ← 新最优
#   Epoch  16/30 | Loss:1.1897 | Val Macro F1:0.4418
#   Epoch  17/30 | Loss:1.1823 | Val Macro F1:0.4661 ← 新最优
#   Epoch  18/30 | Loss:1.1754 | Val Macro F1:0.4702 ← 新最优
#   Epoch  19/30 | Loss:1.1467 | Val Macro F1:0.4737 ← 新最优
#   Epoch  20/30 | Loss:1.1336 | Val Macro F1:0.4508
#   Epoch  22/30 | Loss:1.1080 | Val Macro F1:0.4686
#   Epoch  24/30 | Loss:1.0879 | Val Macro F1:0.4672
#   🛑 Early Stop @ Epoch 24（最优 Epoch 19）
#   【结果】Best Epoch=19 | Test Acc=0.5957 | Macro F1=0.4709

# ============================================================
# 全部改进 - 3种子原始结果
# ============================================================
#                            tag    seed  best_epoch  test_acc  test_f1
# 全部改进(增强+WCE+Dropout+EarlyStop)      42          13  0.629865 0.469818
# 全部改进(增强+WCE+Dropout+EarlyStop)  114514          22  0.599682 0.493975
# 全部改进(增强+WCE+Dropout+EarlyStop) 1919810          19  0.595711 0.470889

# 汇总（mean±std, n=3）：
#   Test Acc : 0.6084 ± 0.0187
#   Macro F1 : 0.4782 ± 0.0136

#   相较 正则化模型(3种子均值, F1=0.4901) 的差异: -0.0119
#   ⚠️ 全部改进组合未能超越单一正则化模型，说明多个改进叠加存在收益冲突/边际递减

# ######################################################################
# # Phase 3：最终消融汇总
# ######################################################################
# 实验组                         Test Acc    Macro F1    备注
# ---------------------------------------------------------------------------
# Baseline v5                 0.6481      0.4643      单种子
# 加权模型(sqrt)                  0.5997      0.4791      单种子
# 正则化模型(3种子均值)                0.6349      0.4901      F1±0.0165
# Soft Voting 集成              0.6481      0.5117      等权3模型
# 全部改进(3种子均值)                 0.6084      0.4782      F1±0.0136
# ===========================================================================

# ✅ 最终推荐配置：Soft Voting 集成（Macro F1=0.5117）
#    （该配置无需单独保存新checkpoint，直接复用对应历史产出）

# 已保存：day7_full_improvement_raw.csv

# 【Day 7 集成与收官完成】
#   请回答输出问题后，将结果更新到 Plan 表 Day 7 行，完成 Week13 收官总结。
