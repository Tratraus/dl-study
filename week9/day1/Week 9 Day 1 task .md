# Week 9 Day 1：Encoder-Decoder 结构

---

## 任务目标

理解为什么序列翻译任务需要 Encoder-Decoder 架构，搞清楚信息是如何从 Encoder 流向 Decoder 的，并写出两者的骨架代码。

---

## 最小必要理论

### 1. 为什么只用 Encoder 不够？

Week 7~8 你一直在用 ESM-2 做 **token-level 分类**：

```text
输入序列（长度 L）→ Encoder → 每个位置输出一个向量 → 分类头
```

输入和输出**长度相同**，所以 Encoder 够用。

但序列翻译任务不同：

```text
输入：ACDEF（长度 5）
输出：FEDCA（长度 5，但可能是任意长度）
```

输出序列的长度是**不确定的**，而且每个输出 token 都依赖**之前已经生成的 token**。Encoder 没有"逐步生成"的能力，所以需要 Decoder。

---

### 2. Encoder-Decoder 的信息流

```text
src 序列
  ↓
[Encoder]
  ↓
memory（形状：batch × src_len × d_model）
  ↓ ↓ ↓ ↓ ↓（传给 Decoder 的每一层）

tgt 序列（已生成部分）
  ↓
[Decoder]（每层内部：Self-Attn → Cross-Attn ← memory → FFN）
  ↓
输出向量
  ↓
[Linear 投影]
  ↓
下一个 token 的概率分布
```

关键点：
- **Encoder 只跑一次**，输出 `memory`
- **Decoder 每层都要用 `memory`**（通过 Cross-Attention）
- Decoder 的输入是"已经生成的部分"，输出是"下一个 token 的预测"

---

### 3. Teacher Forcing

训练时有一个问题：Decoder 需要"已生成的部分"作为输入，但一开始模型什么都不会，预测全是错的，用错误的预测继续生成会导致误差累积。

解决方案：**训练时直接把真实标签喂给 Decoder**，而不是用模型自己的预测。

```text
真实序列：<BOS> F E D C A <EOS>

训练时 Decoder 输入：<BOS> F E D C A   （去掉最后一个）
训练时 Decoder 目标：F E D C A <EOS>   （去掉第一个）

即：tgt_input  = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
```

这就是 **Teacher Forcing**：用真实答案"强制"引导 Decoder，让训练稳定。

推理时没有真实答案，只能用自己上一步的预测，这就是**自回归解码**（Day 6 的内容）。

---

## 代码任务

新建文件：`week9/day1/seq2seq_skeleton.py`

写出 Encoder 和 Decoder 的骨架，**重点是 shape 注释**，不需要实现内部细节。

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    把输入序列编码成上下文表示（memory）。

    输入：src tokens，shape = (batch, src_len)
    输出：memory，shape = (batch, src_len, d_model)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        # TODO 1：定义以下组件（只写 __init__，不写 forward）
        # - token embedding：把 token id 映射成向量
        # - positional encoding：给每个位置加上位置信息
        # - N 个 TransformerEncoderLayer
        # - 最后一个 LayerNorm
        ...

    def forward(self, src, src_key_padding_mask=None):
        # TODO 2：写出 forward 的数据流，用注释标出每步的 shape
        # step 1: embedding，shape = ?
        # step 2: 加 positional encoding，shape = ?
        # step 3: 过 N 层 TransformerEncoderLayer，shape = ?
        # step 4: LayerNorm，shape = ?
        # 返回 memory
        ...


class Decoder(nn.Module):
    """
    根据 memory 和已生成序列，预测下一个 token。

    输入：
      tgt tokens（已生成部分），shape = (batch, tgt_len)
      memory（来自 Encoder），shape = (batch, src_len, d_model)
    输出：
      logits，shape = (batch, tgt_len, vocab_size)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        # TODO 3：定义以下组件
        # - token embedding
        # - positional encoding
        # - N 个 TransformerDecoderLayer
        # - 最后一个 LayerNorm
        # - 输出投影：Linear(d_model, vocab_size)
        ...

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # TODO 4：写出 forward 的数据流，用注释标出每步的 shape
        # step 1: embedding，shape = ?
        # step 2: 加 positional encoding，shape = ?
        # step 3: 过 N 层 TransformerDecoderLayer，shape = ?
        # step 4: LayerNorm，shape = ?
        # step 5: 输出投影，shape = ?
        # 返回 logits
        ...
```

---

## 完成标准

1. `__init__` 中所有组件都定义了（用 `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` 即可，不需要手写）
2. `forward` 中每一步都有 shape 注释，且 shape 正确
3. 能回答下面三个问题

---

## 输出问题

**Q1**：Encoder 的输出 `memory` 的形状是什么？它在 Decoder 的哪个子层被使用？（提示：Decoder 每层有三个子层）

**Q2**：Teacher Forcing 中，`tgt_input` 和 `tgt_output` 分别是什么？如果原始目标序列是 `[<BOS>, 3, 1, 4, 1, 5, <EOS>]`（长度 7），那么 `tgt_input` 和 `tgt_output` 各是什么？

**Q3**：`nn.TransformerDecoderLayer` 接收哪几个参数？其中哪个参数对应 `memory`？（查一下 PyTorch 文档或回忆一下函数签名）

---

准备好后提交代码和三个问题的回答。