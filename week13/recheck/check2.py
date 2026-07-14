import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))

from protein_dataset import make_collate_fn, split_dataset, load_localization_data
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 加载 tokenizer（拿 vocab_size，不需要 base_model）---
tokenizer, _ = load_esm2()
vocab_size = len(tokenizer)

# --- 2. 【关键排查点】连续两次调用 split_dataset，对比 test set 是否一致 ---
sequences, labels = load_localization_data()

_, _, test_dataset_run1 = split_dataset(sequences, labels)
_, _, test_dataset_run2 = split_dataset(sequences, labels)

# 对比两次切分出的 test set 是否完全相同（比如比较序列内容或索引）
print(f"Run1 test set 大小: {len(test_dataset_run1)}")
print(f"Run2 test set 大小: {len(test_dataset_run2)}")

# 如果 test_dataset 是个 list/Dataset，尝试对比前几条内容
try:
    same = all(test_dataset_run1[i] == test_dataset_run2[i] for i in range(min(5, len(test_dataset_run1))))
    print(f"两次 split 的前5条 test 样本是否完全一致: {same}")
except Exception as e:
    print(f"无法直接对比样本内容: {e}")
    # 备用方案：对比标签分布
    print(f"Run1 test labels 分布: {np.bincount([test_dataset_run1[i][1] for i in range(len(test_dataset_run1))])}")
    print(f"Run2 test labels 分布: {np.bincount([test_dataset_run2[i][1] for i in range(len(test_dataset_run2))])}")

# --- 3. 加载模型（用 Day1/Day4 一致的参数）---
model = ProteinClassifier(
    num_classes=10,
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=3,
    d_ff=512,
    max_len=512,
    dropout=0.1
).to(device)

ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'day1', 'baseline_checkpoint.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(state_dict)
model.eval()

# --- 4. 用 run1 的 test set 推理两次，验证模型推理本身是否稳定 ---
collate_fn = make_collate_fn(tokenizer)
test_loader = DataLoader(test_dataset_run1, batch_size=32, shuffle=False, collate_fn=collate_fn)

def infer(loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, mask, lbls in loader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            preds = model(input_ids, mask).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1

acc1, f1_1 = infer(test_loader)
acc2, f1_2 = infer(test_loader)
print(f"\n同一 test set 推理两次: Acc1={acc1:.4f}/F1_1={f1_1:.4f} | Acc2={acc2:.4f}/F1_2={f1_2:.4f}")
print(f"与官方 Baseline (0.6331/0.4860) 差异: ΔAcc={abs(acc1-0.6331):.4f}, ΔF1={abs(f1_1-0.4860):.4f}")
