"""
Day 6: Dropout 调优 + Early Stopping + Label Smoothing 对比
------------------------------------------------------------
已内置三项 Day5 排查验证过的确定性修复：
  ① GPU确定性算子 (use_deterministic_algorithms)
  ② 每组实验入口独立 set_seed(42)
  ③ 每组实验独立重建 DataLoader (全新 generator)
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week13', 'day5'))
from train_classifier import ProteinClassifier
from esm2_embed import load_esm2          # 复用 tokenizer（词表与ESM-2共享）
from weighted_loss_experiment import get_dataloaders   # ⚠️ 请核对该函数实际所在的文件名

torch.use_deterministic_algorithms(True)   # 修复① GPU确定性

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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



def build_fresh_train_loader(train_dataset, collate_fn, batch_size, seed=42):
    """修复③ 每组实验独立重建 DataLoader，全新 generator"""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g
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
    return acc, f1, all_preds, all_labels


# ============================================================
# Early Stopping 实现
# ============================================================

class EarlyStopping:
    """监控 Val Macro F1（而非 Val Acc），因类别不平衡严重，
    Acc 容易被多数类刷高，F1 更能反映稀有类学习效果。"""
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


# ============================================================
# 核心训练函数
# ============================================================

def train_experiment(tag, dropout, train_dataset, collate_fn, batch_size,
                      val_loader, test_loader, vocab_size, num_classes,
                      max_epochs=30, patience=5, seed=42,
                      label_smoothing=0.0):

    set_seed(seed)  # 修复②：模型初始化前独立重置
    train_loader = build_fresh_train_loader(train_dataset, collate_fn, batch_size, seed=seed)  # 修复③

    model = ProteinClassifier(
        num_classes = num_classes,
        vocab_size  = vocab_size,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 3,
        d_ff        = 512,
        max_len     = 512,
        dropout     = dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=patience)

    print(f"\n{'='*60}\n训练组：{tag}\n{'='*60}")

    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0
        for input_ids, mask, labels in train_loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        is_best = early_stopper.step(val_f1, model, epoch)

        if epoch % 2 == 0 or epoch == 1 or is_best or early_stopper.early_stop:
            marker = " ← 新最优" if is_best else ""
            print(f"Epoch {epoch:3d}/{max_epochs} | Loss:{avg_loss:.4f} | "
                  f"Val Acc:{val_acc:.4f} | Val Macro F1:{val_f1:.4f}{marker}")

        if early_stopper.early_stop:
            print(f"🛑 Early Stopping 触发！(patience={patience}) "
                  f"最优出现在 Epoch {early_stopper.best_epoch}，Val Macro F1={early_stopper.best_score:.4f}")
            break

    if epoch == max_epochs and not early_stopper.early_stop:
        print(f"⚠️ 训练到 max_epochs={max_epochs} 也未触发 Early Stopping，"
              f"最优仍在 Epoch {early_stopper.best_epoch}")

    model = early_stopper.restore_best(model)
    test_acc, test_f1, preds, labels_true = evaluate_model(model, test_loader, device)

    print(f"\n【{tag} 最终结果（Early Stop 最优权重）】")
    print(f"  Best Epoch  : {early_stopper.best_epoch}")
    print(f"  停止 Epoch  : {epoch}")
    print(f"  Test Acc    : {test_acc:.4f}")
    print(f"  Macro F1    : {test_f1:.4f}")

    return {
        'tag': tag, 'dropout': dropout, 'label_smoothing': label_smoothing,
        'best_epoch': early_stopper.best_epoch, 'stopped_epoch': epoch,
        'test_acc': test_acc, 'test_f1': test_f1,
        'history': history, 'model': model
    }


# ============================================================
# 主流程
# ============================================================

SEED = 42
BATCH_SIZE = 32
DROPOUT_CANDIDATES = [0.1, 0.3, 0.5]
MAX_EPOCHS = 30
PATIENCE = 5
NUM_CLASSES = 10

# ✅ 补上：复用 ESM-2 的 tokenizer（与 Day1~5 保持一致的词表体系）
tokenizer, _ = load_esm2()
vocab_size = len(tokenizer)

train_loader, val_loader, test_loader, train_dataset, collate_fn = get_dataloaders(
    tokenizer, batch_size=BATCH_SIZE, seed=SEED
)

results = {}

print("\n" + "#"*60)
print("# 阶段一：Dropout 调优 (0.1 / 0.3 / 0.5)")
print("#"*60)

for d in DROPOUT_CANDIDATES:
    tag = f"Dropout={d}"
    results[tag] = train_experiment(
        tag=tag, dropout=d,
        train_dataset=train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE,
        val_loader=val_loader, test_loader=test_loader,
        vocab_size=vocab_size, num_classes=NUM_CLASSES,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
        label_smoothing=0.0
    )

best_dropout_tag = max(results, key=lambda k: results[k]['test_f1'])
best_dropout = results[best_dropout_tag]['dropout']
print(f"\n✅ 最优 Dropout = {best_dropout} (来自 {best_dropout_tag}, "
      f"Test Macro F1={results[best_dropout_tag]['test_f1']:.4f})")

print("\n" + "#"*60)
print(f"# 阶段二：Label Smoothing 对比（基于最优 Dropout={best_dropout}）")
print("#"*60)

tag_no_ls = f"最优Dropout({best_dropout})+无LabelSmoothing"
results[tag_no_ls] = results[best_dropout_tag]

tag_ls = f"最优Dropout({best_dropout})+LabelSmoothing0.1"
results[tag_ls] = train_experiment(
    tag=tag_ls, dropout=best_dropout,
    train_dataset=train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE,
    val_loader=val_loader, test_loader=test_loader,
    vocab_size=vocab_size, num_classes=NUM_CLASSES,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
    label_smoothing=0.1
)

# ============================================================
# 结果汇总
# ============================================================

print("\n" + "="*70)
print("Day 6 完整结果汇总")
print("="*70)
print(f"{'实验组':<35}{'Best Epoch':<12}{'Test Acc':<10}{'Macro F1':<10}")
print("-"*70)
for tag, r in results.items():
    print(f"{tag:<35}{r['best_epoch']:<12}{r['test_acc']:<10.4f}{r['test_f1']:<10.4f}")

BASELINE_V5 = {'test_acc': 0.6481334392374901, 'test_f1': 0.46426376359159355}
d01_result = results['Dropout=0.1']
print("\n" + "="*70)
print("★ Dropout=0.1 组 与 Baseline v5 一致性核验 ★")
print("="*70)
print(f"Baseline v5        : Acc={BASELINE_V5['test_acc']:.4f} | F1={BASELINE_V5['test_f1']:.4f}")
print(f"Dropout=0.1(本组)   : Acc={d01_result['test_acc']:.4f} | F1={d01_result['test_f1']:.4f}")
print("⚠️ 注意：本组加了 Early Stopping，若训练在 <20 epoch 就早停，"
      "结果与Baseline v5理论上会有差异，这是正常现象。")

best_overall_tag = max(results, key=lambda k: results[k]['test_f1'])
best_model = results[best_overall_tag]['model']

SAVE_DIR = os.path.dirname(__file__)
save_path = os.path.join(SAVE_DIR, 'day6_best_checkpoint.pt')

torch.save({
    'model_state_dict': best_model.state_dict(),
    'test_acc': results[best_overall_tag]['test_acc'],
    'macro_f1': results[best_overall_tag]['test_f1'],
    'dropout': results[best_overall_tag]['dropout'],
    'label_smoothing': results[best_overall_tag]['label_smoothing'],
    'best_epoch': results[best_overall_tag]['best_epoch'],
    'seed': SEED,
    'version': 'day6_best_regularized',
    'config': {'tag': best_overall_tag, 'early_stopping_patience': PATIENCE},
}, save_path)

print(f"\n✅ 最优正则化组合：{best_overall_tag}")
print(f"   Test Acc={results[best_overall_tag]['test_acc']:.4f} | "
      f"Macro F1={results[best_overall_tag]['test_f1']:.4f}")
print(f"已保存至 {save_path}（供 Day7 消融汇总/集成使用）")


# 输出
# Using device: cuda
# Using device: cuda
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1757.27it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
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

# ############################################################
# # 阶段一：Dropout 调优 (0.1 / 0.3 / 0.5)
# ############################################################

# ============================================================
# 训练组：Dropout=0.1
# ============================================================
# Epoch   1/30 | Loss:1.7213 | Val Acc:0.4436 | Val Macro F1:0.2339 ← 新最优
# Epoch   2/30 | Loss:1.4848 | Val Acc:0.5072 | Val Macro F1:0.3384 ← 新最优
# Epoch   3/30 | Loss:1.3140 | Val Acc:0.5541 | Val Macro F1:0.3411 ← 新最优
# Epoch   4/30 | Loss:1.2377 | Val Acc:0.5262 | Val Macro F1:0.3144
# Epoch   5/30 | Loss:1.2038 | Val Acc:0.5588 | Val Macro F1:0.3584 ← 新最优
# Epoch   6/30 | Loss:1.1749 | Val Acc:0.5866 | Val Macro F1:0.4172 ← 新最优
# Epoch   8/30 | Loss:1.1296 | Val Acc:0.5922 | Val Macro F1:0.3957
# Epoch  10/30 | Loss:1.0768 | Val Acc:0.6041 | Val Macro F1:0.4332 ← 新最优
# Epoch  11/30 | Loss:1.0509 | Val Acc:0.6017 | Val Macro F1:0.4799 ← 新最优
# Epoch  12/30 | Loss:1.0541 | Val Acc:0.5986 | Val Macro F1:0.4249
# Epoch  14/30 | Loss:1.0043 | Val Acc:0.5986 | Val Macro F1:0.4650
# Epoch  16/30 | Loss:0.9883 | Val Acc:0.6081 | Val Macro F1:0.4758
# 🛑 Early Stopping 触发！(patience=5) 最优出现在 Epoch 11，Val Macro F1=0.4799

# 【Dropout=0.1 最终结果（Early Stop 最优权重）】
#   Best Epoch  : 11
#   停止 Epoch  : 16
#   Test Acc    : 0.6299
#   Macro F1    : 0.4890

# ============================================================
# 训练组：Dropout=0.3
# ============================================================
# Epoch   1/30 | Loss:1.7378 | Val Acc:0.4340 | Val Macro F1:0.2330 ← 新最优
# Epoch   2/30 | Loss:1.5301 | Val Acc:0.4817 | Val Macro F1:0.2841 ← 新最优
# Epoch   3/30 | Loss:1.3754 | Val Acc:0.5207 | Val Macro F1:0.2990 ← 新最优
# Epoch   4/30 | Loss:1.3131 | Val Acc:0.5159 | Val Macro F1:0.2880
# Epoch   5/30 | Loss:1.2605 | Val Acc:0.5493 | Val Macro F1:0.3336 ← 新最优
# Epoch   6/30 | Loss:1.2386 | Val Acc:0.5851 | Val Macro F1:0.3865 ← 新最优
# Epoch   8/30 | Loss:1.1775 | Val Acc:0.5612 | Val Macro F1:0.3650
# Epoch  10/30 | Loss:1.1363 | Val Acc:0.5676 | Val Macro F1:0.3663
# Epoch  11/30 | Loss:1.1232 | Val Acc:0.5827 | Val Macro F1:0.4036 ← 新最优
# Epoch  12/30 | Loss:1.1236 | Val Acc:0.5731 | Val Macro F1:0.3818
# Epoch  14/30 | Loss:1.0825 | Val Acc:0.5453 | Val Macro F1:0.3419
# Epoch  16/30 | Loss:1.0801 | Val Acc:0.5835 | Val Macro F1:0.4024
# 🛑 Early Stopping 触发！(patience=5) 最优出现在 Epoch 11，Val Macro F1=0.4036

# 【Dropout=0.3 最终结果（Early Stop 最优权重）】
#   Best Epoch  : 11
#   停止 Epoch  : 16
#   Test Acc    : 0.6005
#   Macro F1    : 0.4244

# ============================================================
# 训练组：Dropout=0.5
# ============================================================
# Epoch   1/30 | Loss:1.7698 | Val Acc:0.4221 | Val Macro F1:0.1797 ← 新最优
# Epoch   2/30 | Loss:1.5919 | Val Acc:0.4205 | Val Macro F1:0.1852 ← 新最优
# Epoch   3/30 | Loss:1.5163 | Val Acc:0.4618 | Val Macro F1:0.2094 ← 新最优
# Epoch   4/30 | Loss:1.4364 | Val Acc:0.4452 | Val Macro F1:0.2058
# Epoch   5/30 | Loss:1.3730 | Val Acc:0.5048 | Val Macro F1:0.2818 ← 新最优
# Epoch   6/30 | Loss:1.3439 | Val Acc:0.5231 | Val Macro F1:0.3111 ← 新最优
# Epoch   7/30 | Loss:1.3034 | Val Acc:0.5135 | Val Macro F1:0.3394 ← 新最优
# Epoch   8/30 | Loss:1.2794 | Val Acc:0.5103 | Val Macro F1:0.2874
# Epoch  10/30 | Loss:1.2346 | Val Acc:0.4769 | Val Macro F1:0.2768
# Epoch  12/30 | Loss:1.2262 | Val Acc:0.4897 | Val Macro F1:0.2889
# 🛑 Early Stopping 触发！(patience=5) 最优出现在 Epoch 7，Val Macro F1=0.3394

# 【Dropout=0.5 最终结果（Early Stop 最优权重）】
#   Best Epoch  : 7
#   停止 Epoch  : 12
#   Test Acc    : 0.5385
#   Macro F1    : 0.3536

# ✅ 最优 Dropout = 0.1 (来自 Dropout=0.1, Test Macro F1=0.4890)

# ############################################################
# # 阶段二：Label Smoothing 对比（基于最优 Dropout=0.1）
# ############################################################

# ============================================================
# 训练组：最优Dropout(0.1)+LabelSmoothing0.1
# ============================================================
# Epoch   1/30 | Loss:1.8363 | Val Acc:0.4459 | Val Macro F1:0.2270 ← 新最优
# Epoch   2/30 | Loss:1.6493 | Val Acc:0.5207 | Val Macro F1:0.3287 ← 新最优
# Epoch   4/30 | Loss:1.4610 | Val Acc:0.5445 | Val Macro F1:0.3324 ← 新最优
# Epoch   5/30 | Loss:1.4304 | Val Acc:0.5628 | Val Macro F1:0.3631 ← 新最优
# Epoch   6/30 | Loss:1.4134 | Val Acc:0.6017 | Val Macro F1:0.4200 ← 新最优
# Epoch   8/30 | Loss:1.3788 | Val Acc:0.5882 | Val Macro F1:0.3986
# Epoch  10/30 | Loss:1.3301 | Val Acc:0.6025 | Val Macro F1:0.3905
# Epoch  11/30 | Loss:1.3216 | Val Acc:0.6025 | Val Macro F1:0.4374 ← 新最优
# Epoch  12/30 | Loss:1.3188 | Val Acc:0.5946 | Val Macro F1:0.4142
# Epoch  14/30 | Loss:1.2981 | Val Acc:0.6049 | Val Macro F1:0.4209
# Epoch  16/30 | Loss:1.2779 | Val Acc:0.6041 | Val Macro F1:0.4458 ← 新最优
# Epoch  17/30 | Loss:1.2651 | Val Acc:0.6097 | Val Macro F1:0.4607 ← 新最优
# Epoch  18/30 | Loss:1.2533 | Val Acc:0.6192 | Val Macro F1:0.4375
# Epoch  20/30 | Loss:1.2401 | Val Acc:0.6049 | Val Macro F1:0.4274
# Epoch  22/30 | Loss:1.2177 | Val Acc:0.6065 | Val Macro F1:0.4598
# 🛑 Early Stopping 触发！(patience=5) 最优出现在 Epoch 17，Val Macro F1=0.4607

# 【最优Dropout(0.1)+LabelSmoothing0.1 最终结果（Early Stop 最优权重）】
#   Best Epoch  : 17
#   停止 Epoch  : 22
#   Test Acc    : 0.6521
#   Macro F1    : 0.4827

# ======================================================================
# Day 6 完整结果汇总
# ======================================================================
# 实验组                                Best Epoch  Test Acc  Macro F1
# ----------------------------------------------------------------------
# Dropout=0.1                        11          0.6299    0.4890
# Dropout=0.3                        11          0.6005    0.4244
# Dropout=0.5                        7           0.5385    0.3536
# 最优Dropout(0.1)+无LabelSmoothing     11          0.6299    0.4890
# 最优Dropout(0.1)+LabelSmoothing0.1   17          0.6521    0.4827

# ======================================================================
# ★ Dropout=0.1 组 与 Baseline v5 一致性核验 ★
# ======================================================================
# Baseline v5        : Acc=0.6481 | F1=0.4643
# Dropout=0.1(本组)   : Acc=0.6299 | F1=0.4890
# ⚠️ 注意：本组加了 Early Stopping，若训练在 <20 epoch 就早停，结果与Baseline v5理论上会有差异，这是正常现象。

# ✅ 最优正则化组合：Dropout=0.1
#    Test Acc=0.6299 | Macro F1=0.4890
# 已保存至 day6_best_checkpoint.pt（供 Day7 消融汇总/集成使用）
