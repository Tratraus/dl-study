# Week 10 Day 4：序列嵌入提取

## 最小必要理论（10 分钟）

### 1. 什么是序列嵌入？

MLM 预训练结束后，`ProteinBERT` 的 Encoder 部分已经学会了把氨基酸序列映射到一个高维空间。

```text
序列：  A  C  D  E  F  G  H  ...
         ↓
  ProteinBERT Encoder
         ↓
每个位置都有一个向量：(seq_len, d_model)
```

但下游任务（比如分类）需要的是**整条序列的一个向量**，而不是每个位置的向量。

这个"把变长序列压缩成一个固定维度向量"的操作，叫做 **池化（Pooling）**。

---

### 2. 两种池化策略

#### 策略 A：`[CLS]` 池化

在序列开头插入一个特殊 token `[CLS]`（在我们的词表里复用 `BOS`，id=1）：

```text
输入：[BOS] A  C  D  E  F
             ↓
        ProteinBERT Encoder
             ↓
输出：  h0   h1 h2 h3 h4 h5
        ↑
    取第 0 个位置的向量作为序列表示
```

`h0` 的 shape 是 `(batch, d_model)`。

**直觉**：`[BOS]` 位置没有对应的氨基酸，它的输出向量完全来自对整条序列的 attention 聚合，天然适合作为全局表示。

---

#### 策略 B：平均池化（Mean Pooling）

对所有**非 PAD 位置**的输出向量取平均：

```text
输出：h0 h1 h2 h3 [PAD] [PAD]
       ↓
  只对非 PAD 位置取平均
       ↓
  mean(h0, h1, h2, h3)  →  (batch, d_model)
```

**直觉**：每个氨基酸都贡献一份，最终表示是所有位置信息的均值。

---

### 3. 两种策略的对比

| | `[CLS]` 池化 | 平均池化 |
|--|--|--|
| 实现复杂度 | 需要在输入前插入 `[BOS]` | 只需 mask 掉 PAD 位置 |
| 对 PAD 的处理 | 天然不受 PAD 影响 | 需要手动排除 PAD |
| 适用场景 | BERT 原版设计，分类任务 | 更简单，序列长度变化大时稳定 |
| 实际效果 | 差异不大，取决于预训练质量 | 差异不大，取决于预训练质量 |

今天两种都实现，Day 5 分类任务时选其中一种用。

---

### 4. 关键细节：提取嵌入时要关掉 MLM Head

`ProteinBERT` 的结构是：

```text
Encoder → LayerNorm → MLM Head (Linear)
```

提取嵌入时，我们要的是 **LayerNorm 之后、MLM Head 之前** 的向量：

```text
Encoder → LayerNorm → [在这里提取]  →  MLM Head（提取时不用）
```

实现方式：在 `ProteinBERT` 里加一个 `encode` 方法，只跑到 LayerNorm，不过 MLM Head。

---

## 代码任务

**分两步完成：**

---

### Step 1：给 `ProteinBERT` 加 `encode` 方法

在 `week10/day2/protein_bert.py` 里，给 `ProteinBERT` 新增一个方法：

```python
def encode(
    self,
    src: torch.Tensor,
    src_key_padding_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    只跑 Encoder 部分，返回每个位置的上下文向量。
    不经过 MLM Head。

    输入：
      src:                  (batch, seq_len)
      src_key_padding_mask: (batch, seq_len)，True 表示 PAD

    输出：
      hidden: (batch, seq_len, d_model)
              每个位置的上下文表示，已经过 LayerNorm

    提示：
      复用 forward 里的前几步，在 self.norm(x) 之后直接 return，
      不要过 self.mlm_head。
    """
    ...
```

---

### Step 2：新建 `week10/day4/extract_embedding.py`

```python
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))

from mlm_data import (
    VOCAB_SIZE, PAD, BOS, IGNORE_INDEX,
    make_mlm_batch, AA_IDS, token2id, id2token
)
from protein_bert import ProteinBERT


# ── TODO 1：CLS 池化 ──────────────────────────────────────────
def cls_pooling(
    model: ProteinBERT,
    src: torch.Tensor,
) -> torch.Tensor:
    """
    使用 [BOS] token 作为序列全局表示。

    步骤：
      1. 在 src 每条序列的开头插入 BOS token
         原始 src:  (batch, seq_len)
         插入后:    (batch, seq_len + 1)
      2. 构造 src_key_padding_mask（插入 BOS 后，BOS 位置不是 PAD）
      3. 调用 model.encode() 得到 hidden (batch, seq_len+1, d_model)
      4. 取第 0 个位置：hidden[:, 0, :]  →  (batch, d_model)

    提示：
      bos_col = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
      src_with_bos = torch.cat([bos_col, src], dim=1)
    """
    ...


# ── TODO 2：平均池化 ──────────────────────────────────────────
def mean_pooling(
    model: ProteinBERT,
    src: torch.Tensor,
) -> torch.Tensor:
    """
    对所有非 PAD 位置的输出向量取平均。

    步骤：
      1. 构造 src_key_padding_mask: (batch, seq_len)，PAD 位置为 True
      2. 调用 model.encode() 得到 hidden (batch, seq_len, d_model)
      3. 构造 attention mask（非 PAD 位置为 1.0，PAD 位置为 0.0）
         mask: (batch, seq_len, 1)，用于广播
      4. 对非 PAD 位置求加权平均：
         sum(hidden * mask) / sum(mask)

    提示：
      mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
      embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    """
    ...


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_embeddings():
    """
    验证两种池化方法的输出 shape 和基本性质。

    步骤：
      1. 加载 Day 3 保存的 checkpoint（protein_bert_mlm.pt）
      2. 生成一批等长序列（batch=4, seq_len=20）
      3. 分别用 cls_pooling 和 mean_pooling 提取嵌入
      4. 打印 shape、L2 范数

    额外验证：
      - 同一条序列，两种方法得到的嵌入是否相同？（应该不同）
      - 同一条序列输入两次，同一种方法得到的嵌入是否相同？
        （model.eval() 下应该相同，model.train() 下可能不同，为什么？）
    """
    device = torch.device('cpu')

    # 加载 checkpoint
    checkpoint_path = 'week10/day3/protein_bert_mlm.pt'
    model = ProteinBERT().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()   # ← 重要

    print(f"✅ 模型加载成功，final_loss = {checkpoint['final_loss']:.4f}")
    print()

    # 生成测试数据（不需要 MLM mask，直接用原始序列）
    src = torch.tensor(
        [__import__('random').choices(AA_IDS, k=20) for _ in range(4)],
        dtype=torch.long,
        device=device
    )

    # CLS 池化
    with torch.no_grad():
        cls_emb  = cls_pooling(model, src)
        mean_emb = mean_pooling(model, src)

    print(f"CLS  pooling shape: {cls_emb.shape}")
    print(f"Mean pooling shape: {mean_emb.shape}")
    print()

    # 打印每条序列的 L2 范数
    print("各序列嵌入的 L2 范数：")
    print(f"{'序列':>4}  {'CLS 范数':>10}  {'Mean 范数':>10}")
    print("-" * 30)
    for i in range(4):
        cls_norm  = cls_emb[i].norm().item()
        mean_norm = mean_emb[i].norm().item()
        print(f"  {i:2d}  {cls_norm:10.4f}  {mean_norm:10.4f}")

    print()

    # 验证：两种方法是否相同？
    diff = (cls_emb - mean_emb).norm(dim=-1)
    print(f"CLS vs Mean 差异（L2）：{diff.tolist()}")
    print("（应该不同，因为提取位置不同）")

    print()

    # 验证：同一输入，两次推理结果是否一致？
    with torch.no_grad():
        emb1 = mean_pooling(model, src)
        emb2 = mean_pooling(model, src)
    diff2 = (emb1 - emb2).norm(dim=-1).max().item()
    print(f"同一输入两次推理的最大差异：{diff2:.6f}")
    print("（model.eval() 下应该接近 0）")


if __name__ == "__main__":
    verify_embeddings()
```

---

## 完成标准

1. `ProteinBERT.encode` 方法实现正确，输出 shape `(batch, seq_len, d_model)`
2. `cls_pooling` 输出 shape `(batch, d_model)` = `(4, 128)`
3. `mean_pooling` 输出 shape `(batch, d_model)` = `(4, 128)`
4. CLS 和 Mean 的嵌入**不相同**（diff > 0）
5. 同一输入两次推理结果**相同**（diff ≈ 0，因为 `model.eval()`）

---

## 输出问题

**Q1**：`encode` 方法和 `forward` 方法的区别是什么？为什么不直接用 `forward` 的输出来做池化？

**Q2**：`model.eval()` 对嵌入提取有什么影响？如果忘记写 `model.eval()`，同一输入两次推理的结果会一样吗？

**Q3**：平均池化时，为什么要用 `mask.sum(dim=1)` 作为除数，而不是直接用 `seq_len`？

---

准备好后提交代码、`verify_embeddings()` 的终端输出和三个问题的回答。