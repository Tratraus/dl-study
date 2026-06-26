import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
from esm2_embed import load_esm2, MODEL_NAME


# ── 常量 ──────────────────────────────────────────────────────
LOCALIZATION_CLASSES = {
    0: "Cell membrane",
    1: "Cytoplasm",
    2: "Endoplasmic reticulum",
    3: "Golgi apparatus",
    4: "Lysosome/Vacuole",
    5: "Mitochondria",
    6: "Nucleus",
    7: "Peroxisome",
    8: "Plastid",
    9: "Extracellular",
}
NUM_CLASSES = len(LOCALIZATION_CLASSES)


# ── TODO 1：数据加载 ──────────────────────────────────────────
def load_localization_data() -> tuple[list[str], list[int]]:
    """
    加载亚细胞定位分类数据集。

    字段说明：
      - prot_seq：氨基酸序列，氨基酸之间有空格，需要 .replace(' ', '') 去掉
      - localization：0~9 的整数标签

    处理步骤：
      1. load_dataset("mila-intel/ProtST-SubcellularLocalization", split="train")
      2. 去掉 prot_seq 中的空格
      3. 截断序列长度到 512
      4. 过滤长度 < 50 的序列
      5. 打印数据集大小和类别分布
    """
    from datasets import load_dataset
    from collections import Counter

    print("加载 ProtST-SubcellularLocalization 数据集...")
    ds = load_dataset("mila-intel/ProtST-SubcellularLocalization", split="train")

    sequences, labels = [], []
    for item in ds:
        seq = item['prot_seq'].replace(' ', '')[:512]
        lbl = int(item['localization'])
        if len(seq) >= 50:
            sequences.append(seq)
            labels.append(lbl)

    print(f"过滤后数据集大小：{len(sequences)}")
    dist = Counter(labels)
    for i, name in LOCALIZATION_CLASSES.items():
        print(f"  类别 {i}（{name:25s}）：{dist.get(i, 0):4d} 条")

    return sequences, labels





# ── TODO 2：Dataset 类 ────────────────────────────────────────
class ProteinDataset(Dataset):
    """
    蛋白质功能分类 Dataset。

    __init__：
      - 接收 sequences: list[str] 和 labels: list[int]
      - 断言两者长度相同

    __len__：
      - 返回数据集大小

    __getitem__(i)：
      - 返回 (sequences[i], labels[i])
      - 注意：返回字符串，不是 tensor
        （tokenization 在 collate_fn 里做）
    """
    def __init__(self, sequences: list[str], labels: list[int]):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]


# ── TODO 3：train/val/test 划分 ───────────────────────────────
def split_dataset(
    sequences: list[str],
    labels:    list[int],
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[ProteinDataset, ProteinDataset, ProteinDataset]:
    """
    将数据集划分为 train / val / test 三份。

    要求：
      - 按 train_ratio / val_ratio / (1 - train_ratio - val_ratio) 划分
      - 划分前先打乱（用 seed 固定随机性）
      - 返回三个 ProteinDataset 对象

    提示：
      - random.seed(seed) 后再 shuffle
      - test_ratio = 1 - train_ratio - val_ratio
    """
    random.seed(seed)
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    sequences, labels = list(sequences), list(labels)
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ds = ProteinDataset(sequences[:train_end], labels[:train_end])
    val_ds   = ProteinDataset(sequences[train_end:val_end], labels[train_end:val_end])
    test_ds  = ProteinDataset(sequences[val_end:], labels[val_end:])

    return train_ds, val_ds, test_ds


# ── TODO 4：collate_fn ────────────────────────────────────────
def make_collate_fn(tokenizer, max_length: int = 512):
    """
    返回一个 collate_fn 函数，用于 DataLoader。

    collate_fn(batch) 的逻辑：
      - batch: list of (sequence_str, label_int)
      - 分离出 sequences 和 labels
      - 调用 tokenizer(sequences, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length)
      - 将 labels 转为 torch.long tensor
      - 返回 (input_ids, attention_mask, labels)
        shapes:
          input_ids:      (B, L)
          attention_mask: (B, L)
          labels:         (B,)

    注意：这里用闭包（closure）捕获 tokenizer
    """
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        tokenized = tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, labels_tensor
    return collate_fn


# ── TODO 5：正确的 Mean Pooling（修复 Day 1 的 padding 问题）──
def mean_pooling_with_mask(
    last_hidden_state: torch.Tensor,   # (B, L, 320)
    attention_mask:    torch.Tensor,   # (B, L)
) -> torch.Tensor:
    """
    用 attention_mask 做掩码平均，正确处理 padding。

    步骤：
      1. 去掉首尾特殊 token：
           hidden = last_hidden_state[:, 1:-1, :]   # (B, L-2, 320)
           mask   = attention_mask[:, 1:-1]          # (B, L-2)
      2. 将 mask 扩展到 hidden 的维度：
           mask = mask.unsqueeze(-1).float()          # (B, L-2, 1)
      3. 掩码加权求和：
           masked_sum = (hidden * mask).sum(dim=1)    # (B, 320)
      4. 有效位置数量：
           valid_count = mask.sum(dim=1)              # (B, 1)
      5. 求均值：
           return masked_sum / valid_count            # (B, 320)

    注意：valid_count 可能为 0（极短序列），加 1e-8 防止除零
    """
    hidden = last_hidden_state[:, 1:-1, :]  # (B, L-2, 320)
    mask = attention_mask[:, 1:-1]          # (B, L-2)
    mask = mask.unsqueeze(-1).float()       # (B, L-2, 1)
    masked_sum = (hidden * mask).sum(dim=1) # (B, 320)
    valid_count = mask.sum(dim=1)           # (B, 1)
    return masked_sum / (valid_count + 1e-8)


# ── TODO 6：主程序 ────────────────────────────────────────────
def main():
    device = torch.device('cpu')

    # 1. 加载数据
    sequences, labels = load_localization_data()

    # 2. 划分数据集
    train_ds, val_ds, test_ds = split_dataset(sequences, labels)
    print(f"\n数据集划分：")
    print(f"  训练集：{len(train_ds)} 条")
    print(f"  验证集：{len(val_ds)} 条")
    print(f"  测试集：{len(test_ds)} 条")

    # 3. 加载 tokenizer
    print("\n加载 ESM-2 tokenizer...")
    tokenizer, model = load_esm2(device=device)

    # 4. 构建 DataLoader
    collate_fn  = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    # 5. 验证 DataLoader 输出
    print("\n── DataLoader 输出验证 ──")
    batch = next(iter(train_loader))
    input_ids, attention_mask, batch_labels = batch
    print(f"input_ids shape：     {input_ids.shape}")
    print(f"attention_mask shape：{attention_mask.shape}")
    print(f"labels shape：        {batch_labels.shape}")
    print(f"labels 示例：         {batch_labels.tolist()}")

    # 6. 验证 mean_pooling_with_mask
    print("\n── Mean Pooling 验证 ──")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device))
    embeddings = mean_pooling_with_mask(
        outputs.last_hidden_state, attention_mask.to(device))
    print(f"嵌入 shape：{embeddings.shape}")
    print(f"嵌入 dtype：{embeddings.dtype}")

    # 7. 验证：同一条序列，Day1 的简单 pooling vs 今天的掩码 pooling
    print("\n── Day1 vs Day2 Mean Pooling 对比 ──")
    simple_pool = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
    masked_pool = embeddings
    diff = (simple_pool - masked_pool).abs().max().item()
    print(f"最大绝对差异：{diff:.6f}")
    print(f"（batch 内序列长度差异越大，差异越明显）")

    # 8. 打印 batch 内序列长度分布，解释差异来源
    print("\n── batch 内序列长度分布 ──")
    lengths = attention_mask.sum(dim=1).tolist()
    print(f"各序列有效长度（含特殊token）：{[int(l) for l in lengths]}")
    print(f"最长：{max(lengths):.0f}，最短：{min(lengths):.0f}，"
          f"差值：{max(lengths)-min(lengths):.0f}")


if __name__ == "__main__":
    main()

# 输出

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

# 数据集划分：
#   训练集：5871 条
#   验证集：1258 条
#   测试集：1259 条

# 加载 ESM-2 tokenizer...
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1697.07it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

# ── DataLoader 输出验证 ──
# input_ids shape：     torch.Size([8, 512])
# attention_mask shape：torch.Size([8, 512])
# labels shape：        torch.Size([8])
# labels 示例：         [2, 6, 6, 9, 6, 5, 0, 9]

# ── Mean Pooling 验证 ──
# 嵌入 shape：torch.Size([8, 320])
# 嵌入 dtype：torch.float32

# ── Day1 vs Day2 Mean Pooling 对比 ──
# 最大绝对差异：0.605475
# （batch 内序列长度差异越大，差异越明显）

# ── batch 内序列长度分布 ──
# 各序列有效长度（含特殊token）：[320, 357, 512, 219, 254, 356, 512, 484]
# 最长：512，最短：219，差值：293

# Q1：collate_fn 为什么用闭包（make_collate_fn 返回一个函数）而不是直接写一个普通函数？
# 因为 collate_fn 需要访问外部的 tokenizer，而直接写一个普通函数无法捕获外部变量。
# 使用闭包可以让 collate_fn 内部访问到 make_collate_fn 的参数 tokenizer，从而实现灵活的 tokenization。

# Q2：split_dataset 中为什么要用 seed 固定随机性？如果每次划分结果不同会有什么问题？
# 固定 seed 可以保证每次划分数据集的结果相同，便于实验的可重复性。
# 如果每次划分结果不同，模型的训练和评估结果可能会有较大波动，难以比较不同方法的效果。

# Q3：Day1 vs Day2 的 pooling 最大差异是多少？这个差异在什么情况下会变大？
# 最大差异取决于序列中 padding 的比例。序列越短，
# padding 越多，差异可能越大；序列越长，padding 越少，差异可能越小。
