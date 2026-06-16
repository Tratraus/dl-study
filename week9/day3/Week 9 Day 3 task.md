# Week 9 Day 3：完整 Seq2Seq 模型组装 + Teacher Forcing 训练

---

## 任务目标

把 Day 1 的 Encoder/Decoder 组装成完整的 `Seq2Seq` 模型，写出一个完整的训练循环，跑通合成数据上的训练。

---

## 最小必要理论

### 1. 完整的训练数据流

```text
原始数据：
  src = [1, 3, 5, 2, 4]           长度 5
  tgt = [<BOS>, 4, 2, 5, 3, 1, <EOS>]  长度 7

训练时切分：
  tgt_input  = tgt[:, :-1]  → [<BOS>, 4, 2, 5, 3, 1]    长度 6
  tgt_output = tgt[:, 1:]   → [4, 2, 5, 3, 1, <EOS>]    长度 6

前向传播：
  memory = Encoder(src)                         (batch, 5, d_model)
  logits = Decoder(tgt_input, memory)           (batch, 6, vocab_size)

计算 loss：
  loss = CrossEntropyLoss(
      logits.view(-1, vocab_size),   (batch*6, vocab_size)
      tgt_output.view(-1)            (batch*6,)
  )
```

---

### 2. 因果 Mask 的生成

Decoder 的 Self-Attention 必须加因果 mask，防止"看到未来"：

```python
def make_causal_mask(tgt_len, device):
    return torch.triu(
        torch.full((tgt_len, tgt_len), float('-inf'), device=device),
        diagonal=1
    )
```

---

### 3. 合成任务：序列翻转

为了快速验证模型能学到东西，用一个极简任务：

```text
输入：[1, 2, 3, 4, 5]
输出：[5, 4, 3, 2, 1]
```

如果 loss 能从随机水平（≈ log(vocab_size)）下降，说明模型和训练循环都是正确的。

---

## 代码任务

新建文件：`week9/day3/train_seq2seq.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── 复用 Day 1 的 Encoder / Decoder ──────────────────────────
# 直接把 Day 1 的代码粘贴过来，或者 import

# ── Seq2Seq 组装 ─────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """
    组装 Encoder 和 Decoder。

    forward 输入：
      src:        (batch, src_len)
      tgt_input:  (batch, tgt_len)   ← 已经是 tgt[:, :-1]
    forward 输出：
      logits:     (batch, tgt_len, vocab_size)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        # TODO 1：实例化 Encoder 和 Decoder
        ...

    def forward(self, src, tgt_input):
        # TODO 2：
        # step 1: 生成因果 mask，shape = (tgt_len, tgt_len)
        # step 2: Encoder 前向，得到 memory
        # step 3: Decoder 前向，传入 tgt_input、memory、tgt_mask
        # 返回 logits
        ...


# ── 合成数据生成 ──────────────────────────────────────────────
BOS, EOS, PAD = 1, 2, 0
VOCAB_SIZE = 20   # token id 范围 3~19

def make_batch(batch_size, seq_len, device):
    """
    生成一批序列翻转任务的数据。
    src:  随机序列，token id 在 [3, VOCAB_SIZE) 之间
    tgt:  <BOS> + src 翻转 + <EOS>
    """
    src = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    eos = torch.full((batch_size, 1), EOS, device=device)
    tgt = torch.cat([bos, src.flip(dims=[1]), eos], dim=1)
    return src, tgt


# ── 训练循环 ─────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    D_MODEL    = 64
    NUM_HEADS  = 4
    NUM_LAYERS = 2
    D_FF       = 128
    BATCH_SIZE = 64
    SEQ_LEN    = 10
    STEPS      = 500

    model = Seq2Seq(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO 3：写训练循环
    # 每个 step：
    #   1. make_batch 生成数据
    #   2. 切分 tgt_input 和 tgt_output
    #   3. 前向传播得到 logits
    #   4. 计算 CrossEntropyLoss（注意 ignore_index=PAD）
    #   5. 反向传播 + optimizer.step()
    #   6. 每 100 步打印一次 loss

    for step in range(1, STEPS + 1):
        ...

    print("训练完成")


if __name__ == "__main__":
    train()
```

---

## 完成标准

1. 代码无报错，能跑完 500 步
2. loss 从初始值（约 `log(20) ≈ 3.0`）下降到 `0.5` 以下
3. 能回答下面三个问题

---

## 输出问题

**Q1**：`CrossEntropyLoss` 为什么需要 `ignore_index=PAD`？在这个合成任务里，PAD 出现了吗？如果没有，为什么还要写？

**Q2**：`logits.view(-1, vocab_size)` 和 `tgt_output.view(-1)` 是为了什么？不 view 直接传给 loss 会发生什么？

**Q3**：训练 500 步后，loss 降到了多少？如果 loss 没有下降，你会先检查哪里？

---

准备好后提交代码、终端输出和三个问题的回答。