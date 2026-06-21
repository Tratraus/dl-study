# Week 10 Day 1：MLM 原理 + 数据构造

## 最小必要理论（10 分钟）

### 1. 从 Seq2Seq 到 BERT：结构上发生了什么？

Week 9 我们做的是 **Encoder-Decoder**：

```text
Seq2Seq：
  src → [Encoder] → memory → [Decoder] → tgt
  Decoder 是单向的（Causal Mask），只能看左边
```

BERT 只保留 Encoder，**去掉 Decoder**：

```text
BERT：
  seq → [Encoder × N] → 每个位置的上下文表示
  Encoder 是双向的，每个位置能看到整个序列
```

这一个改动意义重大：

| | Seq2Seq Decoder | BERT Encoder |
|--|--|--|
| 注意力方向 | 单向（只看左边） | 双向（看全局） |
| 适合任务 | 生成（翻译、摘要） | 理解（分类、标注） |
| 蛋白质场景 | 序列生成 | 功能预测、结构预测 |

---

### 2. MLM：BERT 的预训练任务

BERT 的训练目标叫 **Masked Language Modeling（MLM）**：

```text
原始序列：  A  C  D  E  F  G  H
随机 mask：  A [M] D  E [M] G  H   ← 15% 位置被 mask
模型预测：       C        F         ← 只预测被 mask 的位置
```

**为什么不预测所有位置？**

> 如果对所有位置算 loss，模型可以"抄近道"——直接复制输入。
> 只预测 mask 位置，模型被迫利用**双向上下文**来推断缺失信息。

---

### 3. BERT 的 80-10-10 策略

被选中的 15% 位置，不是全部替换为 `[MASK]`，而是：

```text
80% → 替换为 [MASK]     （让模型学会利用上下文）
10% → 替换为随机 token  （让模型对所有位置保持警觉）
10% → 保持原 token 不变 （让模型学会"确认"已知信息）
```

**为什么不全部替换为 `[MASK]`？**

> 推理时序列里没有 `[MASK]`，如果训练时全是 `[MASK]`，
> 会造成训练和推理的**分布偏移（distribution shift）**。
> 80-10-10 让模型在任意位置都保持"可能需要预测"的状态。

---

### 4. MLM Loss 的关键细节

```python
# labels 的构造规则：
#   被 mask 的位置 → 填原始 token id（用于计算 loss）
#   其余位置      → 填 -100（cross_entropy 会自动跳过）

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100   # ← 关键：只对 mask 位置算 loss
)
```

---

## 代码任务

新建文件：`week10/day1/mlm_data.py`

```python
import torch
import random

# ── 词表定义（在 Week 9 基础上新增 MASK token）────────────────
PAD_TOKEN  = '<PAD>'
BOS_TOKEN  = '<BOS>'
EOS_TOKEN  = '<EOS>'
UNK_TOKEN  = '<UNK>'
MASK_TOKEN = '<MASK>'   # 🆕 新增

AA_CHARS = list('ACDEFGHIKLMNPQRSTVWY')

VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN] + AA_CHARS
# PAD=0, BOS=1, EOS=2, UNK=3, MASK=4, A=5, C=6, ...

token2id = {tok: i for i, tok in enumerate(VOCAB)}
id2token = {i: tok for i, tok in enumerate(VOCAB)}

PAD  = token2id[PAD_TOKEN]
BOS  = token2id[BOS_TOKEN]
EOS  = token2id[EOS_TOKEN]
UNK  = token2id[UNK_TOKEN]
MASK = token2id[MASK_TOKEN]
VOCAB_SIZE = len(VOCAB)   # 25

AA_IDS = [token2id[aa] for aa in AA_CHARS]   # 氨基酸的 id 列表，用于随机替换

IGNORE_INDEX = -100   # cross_entropy 跳过的标记


def encode(seq: str) -> list[int]:
    return [token2id.get(aa, UNK) for aa in seq.upper()]

def decode(ids: list[int]) -> str:
    return ''.join(
        id2token[i] for i in ids
        if i not in (PAD, BOS, EOS, UNK, MASK)
    )


# ── TODO 1：实现 mask_sequence ────────────────────────────────
def mask_sequence(token_ids: list[int], mask_prob: float = 0.15) -> tuple[list[int], list[int]]:
    """
    对一条序列应用 BERT 的 80-10-10 mask 策略。

    输入：
      token_ids:  List[int]，原始 token id 列表（只含氨基酸，不含特殊 token）
      mask_prob:  float，被选中进行 mask 的概率，默认 0.15

    输出：
      masked_ids: List[int]，经过 mask 处理后的序列
      labels:     List[int]，被 mask 位置填原始 token id，其余位置填 IGNORE_INDEX

    策略：
      1. 遍历每个位置，以 mask_prob 的概率选中该位置
      2. 对选中的位置：
         - 80% 概率：替换为 MASK token
         - 10% 概率：替换为随机氨基酸 id（从 AA_IDS 中随机选）
         - 10% 概率：保持原 token 不变
      3. 未选中的位置：labels 填 IGNORE_INDEX，masked_ids 保持不变

    提示：
      r = random.random()
      if r < 0.8:    → MASK
      elif r < 0.9:  → 随机 token
      else:          → 保持不变
    """
    ...


# ── TODO 2：实现 make_mlm_batch ───────────────────────────────
def make_mlm_batch(batch_size: int, seq_len: int, device) -> tuple:
    """
    生成一批 MLM 训练数据。

    步骤：
      1. 随机生成 batch_size 条氨基酸序列，每条长度为 seq_len
      2. 对每条序列调用 mask_sequence
      3. 把 masked_ids 和 labels 分别 stack 成 tensor

    返回：
      masked_src: (batch, seq_len)，含 MASK token 的输入序列
      labels:     (batch, seq_len)，只有 mask 位置有有效 id，其余为 IGNORE_INDEX

    注意：这里暂时不加 PAD（序列等长），PAD 处理留到 Day 2 的模型里。
    """
    ...


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_batch():
    """
    生成一个小 batch，打印第一条样本，肉眼验证：
      1. masked_src 中约 15% 位置被替换（MASK=4 或随机氨基酸）
      2. labels 中只有被选中位置有有效 id，其余为 -100
      3. masked_src 和 labels 中，被选中位置的原始 token 和 label 对应正确
    """
    device = torch.device('cpu')
    masked_src, labels = make_mlm_batch(batch_size=4, seq_len=20, device=device)

    print(f"masked_src shape: {masked_src.shape}")
    print(f"labels shape:     {labels.shape}")
    print()

    # 打印第一条样本
    src_0   = masked_src[0].tolist()
    label_0 = labels[0].tolist()

    print("位置  masked_src  label")
    print("----  ----------  -----")
    for i, (s, l) in enumerate(zip(src_0, label_0)):
        marker = "← MASKED" if l != IGNORE_INDEX else ""
        print(f"  {i:2d}  {id2token.get(s, '?'):>10}  {l:5d}  {marker}")

    # 统计 mask 比例
    total = len(label_0)
    masked = sum(1 for l in label_0 if l != IGNORE_INDEX)
    print(f"\nmask 比例: {masked}/{total} = {masked/total:.1%}（理论约 15%）")


if __name__ == "__main__":
    verify_batch()
```

---

## 完成标准

1. `mask_sequence` 实现正确：
   - `labels` 中被选中位置填原始 token id，其余填 `-100`
   - `masked_ids` 中 80% 替换为 `MASK`，10% 替换为随机氨基酸，10% 不变
2. `make_mlm_batch` 返回 shape 正确：`(batch, seq_len)`
3. `verify_batch` 输出中，mask 比例在 **10%~20%** 之间（随机波动正常）
4. 打印表格中，`label != -100` 的行对应的原始 token 是正确的氨基酸

---

## 输出问题

**Q1**：为什么 MLM 只对 mask 位置计算 loss，而不是对整个序列？

**Q2**：BERT 的 80-10-10 策略中，为什么不把 15% 全部替换为 `[MASK]`？

**Q3**：Encoder-only 和 Seq2Seq 的注意力方向有什么本质区别？对蛋白质序列建模来说，双向注意力为什么更合适？

---

准备好后提交代码、`verify_batch` 的终端输出和三个问题的回答。