# Week 11 Day 2：真实蛋白质功能分类数据集构造

## 最小必要理论（10 分钟）

### 1. 今天用什么数据？

我们使用 **Swiss-Prot 酶功能分类** 数据集。选它的原因：

```text
✅ 数据量适中（~1000 条）：CPU 上可以跑
✅ 标签清晰：EC 编号大类，共 6 类
✅ 序列来自真实生物体：能体现 ESM-2 预训练的价值
✅ 是蛋白质功能预测领域的经典 benchmark
```

**EC 编号（Enzyme Commission）**：国际酶学委员会对酶的分类体系，第一位数字代表大类：

| EC 大类 | 名称 | 功能 |
|---------|------|------|
| 1 | Oxidoreductases | 氧化还原反应 |
| 2 | Transferases | 基团转移 |
| 3 | Hydrolases | 水解反应 |
| 4 | Lyases | 裂解反应 |
| 5 | Isomerases | 异构化 |
| 6 | Ligases | 连接反应 |

---

### 2. 数据从哪里来？

今天我们**不依赖外部下载**，用代码直接从 HuggingFace Datasets 加载一个预处理好的版本：

```python
from datasets import load_dataset
ds = load_dataset("mila-intel/ProtST-EC", split="train")
```

如果网络有问题，备用方案是手动构造一个小型模拟数据集（代码中会提供）。

---

### 3. 今天修复的核心问题：正确的 Mean Pooling

Day 1 留下了一个 bug：`[:, 1:-1, :]` 在有 padding 的 batch 里会把 PAD token 算进均值。

正确做法是用 `attention_mask` 做**掩码平均**：

```text
步骤：
  1. attention_mask 形状：(B, L+2)，1=有效，0=PAD
  2. 去掉首尾特殊 token 对应的 mask：[:, 1:-1]
  3. 用 mask 过滤 hidden_states，只对有效位置求均值

公式：
  embedding = Σ(hidden[i] * mask[i]) / Σ(mask[i])
```

图示：

```text
序列 B（长度 3，padding 到长度 5）：
  hidden:  [cls] [B1] [B2] [B3] [eos] [PAD] [PAD]
  mask:      1    1    1    1    1     0     0

  去掉首尾后：
  hidden:  [B1]  [B2]  [B3]  [eos] [PAD] [PAD]
  mask:     1     1     1     1     0     0
                                    ↑
                              eos 的 mask=1，还在！

  更正确的做法：只保留真实氨基酸位置
  mask:     1     1     1     0     0     0
            ↑ 用 attention_mask[:, 1:-1] 并手动去掉 eos
```

实际工程中常用的简化：**直接用 `attention_mask[:, 1:-1]` 做掩码**，把 `<eos>` 的影响接受为轻微误差（因为 `<eos>` 只有 1 个位置，对长序列影响极小）。今天采用这个简化方案。

---

### 4. `Dataset` 和 `DataLoader` 的设计

```text
ProteinDataset.__getitem__(i)
  → 返回 (sequence_str, label_int)
  → 注意：返回字符串，不是 token ids
  → tokenization 放在 collate_fn 里批量处理（更高效）

collate_fn(batch)
  → 输入：[(seq1, label1), (seq2, label2), ...]
  → 调用 tokenizer 批量处理序列
  → 返回 (input_ids, attention_mask, labels)
```

为什么 tokenization 放在 `collate_fn` 而不是 `__getitem__`？

```text
放在 __getitem__：每条序列单独 tokenize，无法利用 padding 对齐
放在 collate_fn：整个 batch 一起 tokenize，自动 padding 到 batch 内最长序列
                 → 更高效，padding 量更少
```

---

## 代码任务

新建 `week11/day2/protein_dataset.py`：

```python
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
    def __init__(self, sequences: list[str], labels: list[int]):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        # 返回字符串，不是 tensor
        # tokenization 在 collate_fn 里批量处理
        return self.sequences[i], self.labels[i]


# ── TODO 3：train/val/test 划分 ───────────────────────────────
def split_dataset(
    sequences:   list[str],
    labels:      list[int],
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple['ProteinDataset', 'ProteinDataset', 'ProteinDataset']:
    """
    划分为 train / val / test 三份。
    test_ratio = 1 - train_ratio - val_ratio = 0.15
    """
    random.seed(seed)
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    seqs, lbls = zip(*combined)
    seqs, lbls = list(seqs), list(lbls)

    n       = len(seqs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ds = ProteinDataset(seqs[:n_train],           lbls[:n_train])
    val_ds   = ProteinDataset(seqs[n_train:n_train+n_val], lbls[n_train:n_train+n_val])
    test_ds  = ProteinDataset(seqs[n_train+n_val:],     lbls[n_train+n_val:])

    return train_ds, val_ds, test_ds


# ── TODO 4：collate_fn ────────────────────────────────────────
def make_collate_fn(tokenizer, max_length: int = 512):
    """
    闭包：捕获 tokenizer，返回 collate_fn。

    为什么用闭包？
      - collate_fn 的签名必须是 fn(batch)，不能有额外参数
      - 用闭包把 tokenizer "打包进去"，避免使用全局变量
    """
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        sequences = list(sequences)

        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return inputs['input_ids'], inputs['attention_mask'], label_tensor

    return collate_fn


# ── TODO 5：正确的 Mean Pooling（修复 Day 1 的 padding 问题）──
def mean_pooling_with_mask(
    last_hidden_state: torch.Tensor,   # (B, L, 320)
    attention_mask:    torch.Tensor,   # (B, L)
) -> torch.Tensor:
    """
    用 attention_mask 做掩码平均。

    关键：[:, 1:-1, :] 去掉 <cls> 和 <eos>
         mask 同步去掉首尾，确保 PAD 位置权重为 0
    """
    hidden = last_hidden_state[:, 1:-1, :]     # (B, L-2, 320)
    mask   = attention_mask[:, 1:-1]           # (B, L-2)
    mask   = mask.unsqueeze(-1).float()        # (B, L-2, 1)

    masked_sum  = (hidden * mask).sum(dim=1)   # (B, 320)
    valid_count = mask.sum(dim=1).clamp(min=1e-8)  # (B, 1)

    return masked_sum / valid_count            # (B, 320)


# ── 主程序 ────────────────────────────────────────────────────
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

    # 3. 加载 tokenizer 和模型
    print("\n加载 ESM-2 tokenizer 和模型...")
    tokenizer, model = load_esm2(device=device)

    # 4. 构建 DataLoader
    collate_fn   = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    # 5. 验证 DataLoader 输出
    print("\n── DataLoader 输出验证 ──")
    batch = next(iter(train_loader))
    input_ids, attention_mask, batch_labels = batch
    print(f"input_ids shape：      {input_ids.shape}")
    print(f"attention_mask shape： {attention_mask.shape}")
    print(f"labels shape：         {batch_labels.shape}")
    print(f"labels 示例：          {batch_labels.tolist()}")
    print(f"attention_mask 示例（第1条）：{attention_mask[0].tolist()}")

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

    # 7. Day1 简单 pooling vs Day2 掩码 pooling 对比
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

```

---

## 完成标准

1. `load_ec_data` 成功加载数据（HuggingFace 或模拟数据均可），打印类别分布
2. `split_dataset` 正确划分三份，比例约为 70/15/15
3. `collate_fn` 输出 shape 正确：`input_ids (8, L)`，`labels (8,)`
4. `mean_pooling_with_mask` 输出 shape `(8, 320)`
5. Day1 vs Day2 的 pooling 差异打印出来（不要求具体数值，能运行即可）

---

## 输出问题

**Q1**：`collate_fn` 为什么用**闭包**（`make_collate_fn` 返回一个函数）而不是直接写一个普通函数？

**Q2**：`split_dataset` 中为什么要用 `seed` 固定随机性？如果每次划分结果不同会有什么问题？

**Q3**：Day1 vs Day2 的 pooling 最大差异是多少？这个差异在什么情况下会变大？

---

准备好后提交代码和终端输出。