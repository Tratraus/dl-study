# Week 10 Day 2：Encoder-only ProteinBERT 实现

## 最小必要理论（10 分钟）

### 1. 从 Week 9 的 Encoder 到 ProteinBERT

Week 9 你实现的 Encoder 是这样的：

```text
src → Embedding → + PE → TransformerEncoderLayer × N → LayerNorm → memory
```

`memory` 的 shape 是 `(batch, seq_len, d_model)`，然后传给 Decoder。

今天我们做的事情很简单：

```text
src → Embedding → + PE → TransformerEncoderLayer × N → LayerNorm → Linear(d_model, vocab_size) → logits
```

**去掉 Decoder，加一个线性投影层**，就是 BERT 的 MLM 结构。

---

### 2. 三个结构差异

| | Week 9 Encoder | ProteinBERT |
|--|--|--|
| 后接模块 | Decoder（生成） | MLM Head（预测） |
| Causal Mask | Decoder 需要 | **不需要** |
| 输出 | `(batch, seq_len, d_model)` | `(batch, seq_len, vocab_size)` |

注意：Encoder 本身没有变化，变化只在**输出端**。

---

### 3. `[CLS]` token 的位置

BERT 在序列开头插入一个特殊 token `[CLS]`：

```text
原始序列：  A  C  D  E  F
BERT 输入：[CLS] A  C  D  E  F
```

`[CLS]` 位置的输出向量会被用作**整个序列的全局表示**，用于下游分类任务（Day 4~5 会用到）。

今天 Day 2 先不强制要求加 `[CLS]`，但词表里已经有 `BOS` token（id=1），后续可以直接复用。

---

### 4. MLM Head 的结构

MLM Head 通常是：

```python
nn.Linear(d_model, vocab_size)
```

有些实现会在 Linear 前加一层 `GELU + LayerNorm`，称为 **dense projection**：

```python
nn.Linear(d_model, d_model) → GELU → LayerNorm → nn.Linear(d_model, vocab_size)
```

今天用最简单的单层 Linear 即可，Day 3 训练时如果 loss 不收敛再考虑加深。

---

## 代码任务

新建文件：`week10/day2/protein_bert.py`

```python
import torch
import torch.nn as nn
import sys
import os

# 复用 Day 1 的词表定义
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
from mlm_data import (
    VOCAB_SIZE, PAD, MASK, IGNORE_INDEX,
    make_mlm_batch, token2id, id2token
)


# ── TODO 1：实现 PositionalEncoding ──────────────────────────
class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learnable PE）。
    和 Week 9 一致，用 nn.Embedding 实现。

    输入：x，shape (batch, seq_len, d_model)
    输出：x + pos_emb，shape (batch, seq_len, d_model)

    提示：
      positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
      pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


# ── TODO 2：实现 ProteinBERT ──────────────────────────────────
class ProteinBERT(nn.Module):
    """
    BERT 式蛋白质语言模型。

    结构：
      Embedding(vocab_size, d_model, padding_idx=PAD)
      + PositionalEncoding
      + Dropout
      → TransformerEncoderLayer × num_layers
      → LayerNorm
      → Linear(d_model, vocab_size)   ← MLM Head

    参数：
      vocab_size:  词表大小（直接用 VOCAB_SIZE）
      d_model:     嵌入维度，默认 128
      num_heads:   注意力头数，默认 4
      num_layers:  Encoder 层数，默认 3
      d_ff:        FFN 中间维度，默认 256
      max_len:     最大序列长度，默认 512
      dropout:     dropout 比例，默认 0.1

    forward 输入：
      src:                  (batch, seq_len)，含 MASK token 的序列
      src_key_padding_mask: (batch, seq_len)，True 表示该位置是 PAD，默认 None

    forward 输出：
      logits: (batch, seq_len, vocab_size)

    注意：
      - 不需要 Causal Mask
      - src_key_padding_mask 传给每一个 TransformerEncoderLayer
      - 最终输出是 logits，不是 softmax 后的概率（loss 函数会处理）
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model:    int = 128,
        num_heads:  int = 4,
        num_layers: int = 3,
        d_ff:       int = 256,
        max_len:    int = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()
        # TODO：定义以下模块
        # self.embedding = ...
        # self.pe = ...
        # self.dropout = ...
        # self.layers = ...   ← nn.ModuleList，每个元素是 TransformerEncoderLayer
        # self.norm = ...     ← LayerNorm
        # self.mlm_head = ... ← Linear(d_model, vocab_size)
        ...

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # TODO：
        # 1. Embedding + PE + Dropout
        # 2. 逐层过 TransformerEncoderLayer（传入 src_key_padding_mask）
        # 3. LayerNorm
        # 4. MLM Head → logits
        ...


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_model():
    """
    验证 ProteinBERT 的 forward 输出 shape 是否正确。

    步骤：
      1. 用 make_mlm_batch 生成一批数据（batch=4, seq_len=20）
      2. 构造 src_key_padding_mask（全 False，因为当前序列等长无 PAD）
      3. 实例化 ProteinBERT（用默认参数）
      4. 前向传播，打印 logits 的 shape
      5. 验证 shape == (4, 20, 25)

    额外验证：
      - 统计模型总参数量（可训练参数数）
      - 打印每个子模块的参数量
    """
    device = torch.device('cpu')
    masked_src, labels = make_mlm_batch(batch_size=4, seq_len=20, device=device)

    # src_key_padding_mask：当前无 PAD，全为 False
    src_key_padding_mask = (masked_src == PAD)   # (4, 20)，全 False

    model = ProteinBERT().to(device)
    logits = model(masked_src, src_key_padding_mask=src_key_padding_mask)

    print(f"logits shape: {logits.shape}")
    assert logits.shape == (4, 20, VOCAB_SIZE), \
        f"shape 错误！期望 (4, 20, {VOCAB_SIZE})，实际 {logits.shape}"
    print("✅ shape 验证通过")
    print()

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量：    {total:,}")
    print(f"可训练参数量：{trainable:,}")
    print()

    # 打印各子模块参数量
    print("各子模块参数量：")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {params:,}")


if __name__ == "__main__":
    verify_model()
```

---

## 完成标准

1. `PositionalEncoding.forward` 输出 shape 与输入相同：`(batch, seq_len, d_model)`
2. `ProteinBERT.forward` 输出 shape：`(4, 20, 25)`
3. `assert` 通过，打印 `✅ shape 验证通过`
4. 打印出总参数量和各子模块参数量

---

## 输出问题

**Q1**：`ProteinBERT` 和 Week 9 的 `Encoder` 在代码上有哪两处关键差异？

**Q2**：`src_key_padding_mask = (masked_src == PAD)` 这一行，当前输出全是 `False`。那么在 Day 3 引入变长序列后，这个 mask 的作用是什么？

**Q3**：MLM Head 是 `Linear(d_model, vocab_size)`，输出是 logits 而不是概率。为什么不在模型里加 `softmax`，而是把它留给 loss 函数处理？

---

准备好后提交代码、`verify_model()` 的终端输出和三个问题的回答。