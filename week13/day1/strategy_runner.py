import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # 必须在其他CUDA操作之前设置

import torch
torch.use_deterministic_algorithms(True)   # 比 cudnn.deterministic 更严格，会强制报错提示哪些算子不确定
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import sys
import random                          # ← 新增
from copy import deepcopy

# ── 路径 ──────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from esm2_embed import load_esm2
from protein_dataset import ProteinDataset, make_collate_fn, split_dataset, load_localization_data
from train_classifier import ProteinClassifier

# ══════════════════════════════════════════════════════
# 0. 【新增】全局种子固定函数
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

# ── 设备 ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════
# 1. 数据加载（复用 W11）—— DataLoader 加 generator
# ══════════════════════════════════════════════════════
def get_dataloaders(tokenizer, batch_size=32, seed=42):   # ← 新增 seed 参数
    sequences, labels = load_localization_data()
    train_dataset, val_dataset, test_dataset = split_dataset(sequences, labels)

    collate_fn = make_collate_fn(tokenizer)

    # ← 新增：带种子的 generator，专门控制 shuffle 顺序
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              generator=g)                 # ← 新增
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════
# 2. 三策略模型构建
# ══════════════════════════════════════════════════════
def build_model_frozen(base_model, num_classes):
    """
    策略 A：Frozen
    - backbone 权重全部冻结（requires_grad=False）
    - 只训练分类头
    """
    esm2_model = deepcopy(base_model).to(device)

    # 冻结 backbone 所有参数
    for param in esm2_model.parameters():
        param.requires_grad_(False)

    # 分类头（可训练）
    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)
    return esm2_model, classifier_head

def build_model_finetune(base_model, num_classes):
    """
    策略 B：Fine-tune
    - backbone + head 全部可训练
    - backbone 用更小的 lr（Day 2 深入实现，今天只搭骨架）
    """
    esm2_model = deepcopy(base_model).to(device)

    # 全部参数可训练（默认就是，无需额外操作）
    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)
    return esm2_model, classifier_head


def build_model_probing(base_model, num_classes):
    """
    策略 C：Probing（比 Frozen 更严格）
    - backbone 冻结，且设为 eval() 模式（BN/Dropout 不更新）
    - 只训练线性分类头
    """
    esm2_model = deepcopy(base_model).to(device)

    for param in esm2_model.parameters():
        param.requires_grad_(False)
    esm2_model.eval()   # ← Probing 与 Frozen 的关键区别

    hidden_dim = esm2_model.config.hidden_size
    classifier_head = nn.Linear(hidden_dim, num_classes).to(device)

    return esm2_model, classifier_head


# ══════════════════════════════════════════════════════
# 3. 通用训练/评估函数（骨架）
# ══════════════════════════════════════════════════════
def get_esm2_embedding(esm2_model, input_ids, attention_mask, strategy):
    """
    从 ESM-2 提取序列嵌入（mean pooling）
    strategy: 'frozen'/'probing' 时用 torch.no_grad()，'finetune' 时正常前向
    从已 tokenized 的输入提取 ESM-2 嵌入（mean pooling）
    tokenization 已在 collate_fn 完成，这里直接前向传播
    """
    from protein_dataset import mean_pooling_with_mask

    if strategy in ('frozen', 'probing'):
        with torch.no_grad():
            outputs = esm2_model(input_ids=input_ids,
                                 attention_mask=attention_mask)
    else:
        outputs = esm2_model(input_ids=input_ids,
                             attention_mask=attention_mask)

    return mean_pooling_with_mask(outputs.last_hidden_state, attention_mask)


def train_one_epoch(esm2_model, classifier_head,
                    optimizer, criterion, train_loader, strategy):
    classifier_head.train()
    if strategy == 'finetune':
        esm2_model.train()
    else:
        esm2_model.eval()

    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch          # ← 元组解包
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        embeddings = get_esm2_embedding(esm2_model, input_ids,
                                        attention_mask, strategy)
        logits = classifier_head(embeddings)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def quick_evaluate(esm2_model, classifier_head, loader, strategy):
    """快速评估：返回 Accuracy 和 Macro F1（用 sklearn 临时计算）"""
    esm2_model.eval()
    classifier_head.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            embeddings = get_esm2_embedding(esm2_model, input_ids,
                                            attention_mask, strategy)
            preds = classifier_head(embeddings).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1


# ══════════════════════════════════════════════════════
# 4. 骨架跑通验证（只跑 3 epoch，确认 loss 下降）
# ══════════════════════════════════════════════════════
def run_strategy_skeleton(strategy_name, esm2_model, classifier_head,
                          train_loader, val_loader, epochs=3):
    """
    骨架验证：只跑 3 epoch，确认三种策略均可正常训练
    Day 2~6 会在此基础上深化各策略
    """
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in list(esm2_model.parameters())
                        + list(classifier_head.parameters())
                        if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    print(f"\n{'='*50}")
    print(f"策略：{strategy_name}")
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count     = sum(p.numel() for p in list(esm2_model.parameters())
                          + list(classifier_head.parameters()))
    print(f"可训练参数：{trainable_count:,} / 总参数：{total_count:,}"
          f"（{100*trainable_count/total_count:.1f}%）")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(esm2_model, classifier_head, optimizer,
                                     criterion, train_loader, strategy_name)
        val_acc, val_f1 = quick_evaluate(esm2_model, classifier_head,
                                         val_loader, strategy_name)  # ← 也没有 tokenizer
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    return esm2_model, classifier_head


# ══════════════════════════════════════════════════════
# 5. Baseline 固定（自实现模型，完整训练）
# ══════════════════════════════════════════════════════
def run_baseline(train_loader, val_loader, test_loader,
                 tokenizer, num_classes, epochs=20, seed=42):
    """
    Baseline：自实现 ProteinClassifier（W12）
    - 无数据增强
    - 标准 CrossEntropy
    - dropout=0.1
    - 无 Early Stopping
    - 固定 epoch 数
    """
    set_seed(seed)   # ★★★ 新增：确保 Baseline 模型创建时是干净的种子状态，不受前置3组热身实验影响 ★★★
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    print("\n" + "="*50)
    print("【Baseline】自实现 ProteinClassifier")
    print("="*50)

    VOCAB_SIZE = len(tokenizer)   # ← 从 tokenizer 拿词表大小

    model = ProteinClassifier(
        num_classes = num_classes,
        vocab_size  = VOCAB_SIZE,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 3,
        d_ff        = 512,
        max_len     = 512,
        dropout     = 0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for input_ids, mask, labels in train_loader:   # ← 元组解包
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)            # ← 传 mask
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        # 验证
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

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}"
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

    print(f"\n{'='*50}")
    print(f"【Baseline 最终结果】")
    print(f"  Test Acc    : {test_acc:.4f}")
    print(f"  Macro F1    : {test_f1:.4f}")
    print(f"{'='*50}")

    # ── 保存 checkpoint（Day 4、Day 7 复用）──────────
    save_path = os.path.join(os.path.dirname(__file__), 'baseline_checkpoint.pt')
    torch.save({
        'model_state_dict' : model.state_dict(),
        'test_acc'         : test_acc,
        'macro_f1'         : test_f1,
        'seed'             : 42,                    # ← 新增
        'version'          : 'v5_seeded_freshloader',           # ← 新增，标记这是固定种子后的版本
        'config': {
            'num_classes' : num_classes,
            'dropout'     : 0.1,
            'epochs'      : epochs,
            'lr'          : 1e-3,
            'augmentation': False,
            'loss'        : 'CrossEntropy',
            'early_stop'  : False,
        }
    }, save_path)

    print(f"Baseline checkpoint 已保存：{save_path}")

    return model, test_acc, test_f1


# ══════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    from copy import deepcopy

    set_seed(42)                      # ← 关键新增：必须在创建任何模型/DataLoader之前调用

    NUM_CLASSES = 10

    tokenizer, base_model = load_esm2()

    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, batch_size=32, seed=42)   # ← 传 seed

    strategies = {
        'frozen'  : build_model_frozen(base_model, NUM_CLASSES),
        'finetune': build_model_finetune(base_model, NUM_CLASSES),
        'probing' : build_model_probing(base_model, NUM_CLASSES),
    }

    for name, (esm2_model, classifier_head) in strategies.items():
        run_strategy_skeleton(name, esm2_model, classifier_head,
                              train_loader, val_loader, epochs=3)

    baseline_model, test_acc, test_f1 = run_baseline(
        train_loader, val_loader, test_loader,
        tokenizer=tokenizer,
        num_classes=NUM_CLASSES,
        epochs=20
    )

    print("\n【Day 1 重跑（Baseline v5，已固定种子）完成】")
    print(f"  Baseline Test Acc : {test_acc:.4f}")
    print(f"  Baseline Macro F1 : {test_f1:.4f}")


# 输出
# ==================================================
# 策略：frozen
# 可训练参数：3,210 / 总参数：7,515,011（0.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.7661 | Val Acc: 0.4841 | Val Macro F1: 0.2469
# Epoch 2/3 | Loss: 1.4377 | Val Acc: 0.5294 | Val Macro F1: 0.3090
# Epoch 3/3 | Loss: 1.3077 | Val Acc: 0.5612 | Val Macro F1: 0.3534

# ==================================================
# 策略：finetune
# 可训练参数：7,515,011 / 总参数：7,515,011（100.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.4218 | Val Acc: 0.5533 | Val Macro F1: 0.2836
# Epoch 2/3 | Loss: 1.1607 | Val Acc: 0.5739 | Val Macro F1: 0.3698
# Epoch 3/3 | Loss: 1.0584 | Val Acc: 0.6200 | Val Macro F1: 0.4467

# ==================================================
# 策略：probing
# 可训练参数：3,210 / 总参数：7,515,011（0.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.7482 | Val Acc: 0.4897 | Val Macro F1: 0.2626
# Epoch 2/3 | Loss: 1.4336 | Val Acc: 0.5350 | Val Macro F1: 0.3253
# Epoch 3/3 | Loss: 1.3058 | Val Acc: 0.5707 | Val Macro F1: 0.3641

# ==================================================
# 【Baseline】自实现 ProteinClassifier
# ==================================================
# Epoch   5/20 | Loss: 1.2081 | Val Acc: 0.5684 | Val Macro F1: 0.3504
# Epoch  10/20 | Loss: 1.0824 | Val Acc: 0.5970 | Val Macro F1: 0.4027
# Epoch  15/20 | Loss: 1.0084 | Val Acc: 0.6049 | Val Macro F1: 0.4205
# Epoch  20/20 | Loss: 0.9548 | Val Acc: 0.6121 | Val Macro F1: 0.4393

# ==================================================
# 【Baseline 最终结果】
#   Test Acc    : 0.6124
#   Macro F1    : 0.4527
# ==================================================
# Baseline checkpoint 已保存：/home/tratr/dl-study/week13/day1/baseline_checkpoint.pt

# 【Day 1 完成】
#   三策略骨架：✅ 均可正常训练
#   Baseline Test Acc : 0.6124
#   Baseline Macro F1 : 0.4527
#   Checkpoint 已保存，供 Day 4 / Day 7 使用

# Q1: Frozen 和 Probing 有什么区别？ 什么情况下用 Probing 更合适？
# Frozen针对分类头（含BN），Probing则仅线性分类头，且BN/Dropout不更新。
# Probing更适合小数据量场景，因为它减少了过拟合的风险。

# Q2: 用"可训练参数比例"把五种策略排个序，并说明各自适用的数据量场景。
# Probing < Frozen < Adapter ≈ LoRA < Full Fine-tune
# Probing：小数据量，测试预训练特征
# Frozen：中等数据量，仅更新分类特征时
# Adapter/LoRA：中等到大数据量，参数高效微调
# Full Fine-tune：大数据量，充分利用预训练模型能力

# Q3: 你固定的 Baseline 配置是什么？ Test Acc 和 Macro F1 分别是多少？
# Baseline 配置：自实现 ProteinClassifier，训练 20 个 epoch
# Test Acc: 0.6124
# Macro F1: 0.4527

# 重跑输出
# Using device: cuda
# [set_seed] 全局随机种子已固定为 42
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1425.92it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
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
# 策略：frozen
# 可训练参数：3,210 / 总参数：7,515,011（0.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.7541 | Val Acc: 0.4849 | Val Macro F1: 0.2493
# Epoch 2/3 | Loss: 1.4340 | Val Acc: 0.5429 | Val Macro F1: 0.3302
# Epoch 3/3 | Loss: 1.3066 | Val Acc: 0.5763 | Val Macro F1: 0.3621

# ==================================================
# 策略：finetune
# 可训练参数：7,515,011 / 总参数：7,515,011（100.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.3690 | Val Acc: 0.5588 | Val Macro F1: 0.3165
# Epoch 2/3 | Loss: 1.1695 | Val Acc: 0.5723 | Val Macro F1: 0.3466
# Epoch 3/3 | Loss: 1.0843 | Val Acc: 0.6351 | Val Macro F1: 0.4383

# ==================================================
# 策略：probing
# 可训练参数：3,210 / 总参数：7,515,011（0.0%）
# ==================================================
# Epoch 1/3 | Loss: 1.7525 | Val Acc: 0.4936 | Val Macro F1: 0.2407
# Epoch 2/3 | Loss: 1.4321 | Val Acc: 0.5254 | Val Macro F1: 0.3103
# Epoch 3/3 | Loss: 1.3054 | Val Acc: 0.5684 | Val Macro F1: 0.3681

# ==================================================
# 【Baseline】自实现 ProteinClassifier
# ==================================================
# Epoch   5/20 | Loss: 1.2224 | Val Acc: 0.5429 | Val Macro F1: 0.3520
# Epoch  10/20 | Loss: 1.0854 | Val Acc: 0.5851 | Val Macro F1: 0.4070
# Epoch  15/20 | Loss: 1.0181 | Val Acc: 0.6097 | Val Macro F1: 0.4515
# Epoch  20/20 | Loss: 0.9483 | Val Acc: 0.6184 | Val Macro F1: 0.4675

# ==================================================
# 【Baseline 最终结果】
#   Test Acc    : 0.6378
#   Macro F1    : 0.4890
# ==================================================
# Baseline checkpoint 已保存：/home/tratr/dl-study/week13/day1/baseline_checkpoint.pt

# 【Day 1 重跑（Baseline v2，已固定种子）完成】
#   Baseline Test Acc : 0.6378
#   Baseline Macro F1 : 0.4890
