好的，安排完全合理。

---

## 本周剩余计划调整

```
Day 6（今天）：蛋白质序列建模——真实任务初体验        ← 现在
Day 7：        注意力可视化 + 模型分析
Day 6.5：      【拓展】长度惩罚 + 温度采样 / Top-k    ← Day 7 前，有余力再做
```

---

# Week 9 Day 6：蛋白质序列建模初体验

## 任务目标

把 Seq2Seq 框架迁移到真实生物序列任务上：**给定蛋白质 N 端序列，预测 C 端序列**。理解真实任务和合成任务的核心差异。

---

## 背景知识（5 分钟）

### 为什么用这个任务？

蛋白质序列由 20 种氨基酸组成，天然适合用词表大小为 20 的序列模型处理。

```text
氨基酸字母表（单字母缩写）：
A R N D C Q E G H I L K M F P S T W Y V
（共 20 种 + 特殊符号 X/- 表示未知/gap）

示例蛋白质序列：
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

### N 端 → C 端预测的生物意义

蛋白质的 N 端（起始端）往往包含信号肽、结构域信息，对 C 端的功能区域有约束关系。这个任务是一个**简化版**的蛋白质语言模型预训练目标。

---

## 数据策略：用 UniProt 的小子集

真实数据集太大，今天用**模拟真实分布的合成数据**：

- 氨基酸词表（20种）+ 特殊 token
- 序列长度分布模拟真实蛋白质（50~200 aa）
- N 端取前 `L//2` 个氨基酸，C 端取后 `L//2` 个

这样数据分布比序列翻转**复杂得多**（没有简单的对称规律），可以直接感受模型容量的瓶颈。

---

## 代码任务

新建文件：`week9/day6/protein_seq2seq.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ── 氨基酸词表 ────────────────────────────────────────────────
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

AA_CHARS = list('ACDEFGHIKLMNPQRSTVWY')  # 20 种标准氨基酸

VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + AA_CHARS
# PAD=0, BOS=1, EOS=2, UNK=3, A=4, C=5, ...

token2id = {tok: i for i, tok in enumerate(VOCAB)}
id2token = {i: tok for i, tok in enumerate(VOCAB)}

PAD = token2id[PAD_TOKEN]
BOS = token2id[BOS_TOKEN]
EOS = token2id[EOS_TOKEN]
UNK = token2id[UNK_TOKEN]
VOCAB_SIZE = len(VOCAB)  # 24

def encode(seq: str) -> list[int]:
    """氨基酸字符串 → token id 列表"""
    return [token2id.get(aa, UNK) for aa in seq.upper()]

def decode(ids: list[int]) -> str:
    """token id 列表 → 氨基酸字符串（跳过特殊 token）"""
    return ''.join(
        id2token[i] for i in ids
        if i not in (PAD, BOS, EOS, UNK)
    )


# ── 合成蛋白质数据生成 ────────────────────────────────────────
def make_protein_batch(batch_size, device, min_len=20, max_len=60):
    """
    模拟蛋白质 N端→C端预测任务。

    生成策略：
      1. 随机生成长度在 [min_len, max_len] 之间的氨基酸序列
      2. 前半段作为 src（N端），后半段作为 tgt（C端）
      3. 用 PAD 对齐到 batch 内最长长度

    返回：
      src:          (batch, max_src_len)，含 PAD
      tgt:          (batch, max_tgt_len+2)，含 BOS/EOS/PAD
      src_padding_mask: (batch, max_src_len)，True 表示 PAD 位置
    """
    src_seqs, tgt_seqs = [], []

    for _ in range(batch_size):
        length = random.randint(min_len, max_len)
        # 随机生成氨基酸序列
        seq = [random.choice(AA_CHARS) for _ in range(length)]
        mid = length // 2
        n_term = seq[:mid]   # N 端
        c_term = seq[mid:]   # C 端

        src_seqs.append(encode(''.join(n_term)))
        tgt_seqs.append(
            [BOS] + encode(''.join(c_term)) + [EOS]
        )

    # TODO 1：对 src_seqs 做 padding，对齐到 batch 内最长长度
    # 提示：用 torch.nn.utils.rnn.pad_sequence，或手动 padding
    # src shape: (batch, max_src_len)
    # src_padding_mask shape: (batch, max_src_len)，PAD 位置为 True
    ...

    # TODO 2：对 tgt_seqs 做 padding
    # tgt shape: (batch, max_tgt_len)
    ...

    return src, tgt, src_padding_mask


# ── 复用 Day 3 的模型 ─────────────────────────────────────────
# 把 Encoder、Decoder、Seq2Seq、make_causal_mask 直接复制过来
# （不需要修改任何模型代码）
...


# ── 训练循环 ─────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL    = 128
    NUM_HEADS  = 4
    NUM_LAYERS = 3
    D_FF       = 256
    BATCH_SIZE = 32
    STEPS      = 1000

    model = Seq2Seq(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, STEPS + 1):
        # TODO 3：调用 make_protein_batch，完成训练循环
        # 注意：这次 src 有 padding，需要把 src_padding_mask 传给 Seq2Seq
        # 需要修改 Seq2Seq.forward 接受外部传入的 src_key_padding_mask
        ...

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("训练完成")
    evaluate(model, device)


# ── 评估函数 ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, device, n_samples=3):
    """
    打印几条样本的 N端输入、真实C端、模型预测C端。
    注意：这个任务没有唯一正确答案（随机生成的序列），
    所以评估指标改为打印序列，肉眼观察模型是否生成了合法的氨基酸序列。
    """
    model.eval()
    src, tgt, src_padding_mask = make_protein_batch(n_samples, device, min_len=20, max_len=30)

    # TODO 4：用 greedy_decode 生成预测序列
    # 注意：greedy_decode 也需要传 src_padding_mask
    ...

    print("\n── 蛋白质序列预测 ──")
    for i in range(n_samples):
        n_term = decode(src[i].tolist())
        c_term_true = decode(tgt[i].tolist())
        c_term_pred = decode(generated[i].tolist())

        print(f"N端输入:   {n_term}")
        print(f"真实C端:   {c_term_true}")
        print(f"预测C端:   {c_term_pred}")
        print(f"长度匹配:  真实={len(c_term_true)}, 预测={len(c_term_pred)}")
        print()


if __name__ == "__main__":
    train()
```

---

## 完成标准

1. `make_protein_batch` 实现完整，padding 正确
2. 训练 1000 步，loss 从初始值下降（**不要求收敛**，这个任务比翻转难很多）
3. `evaluate` 输出的预测序列**全部由合法氨基酸字母组成**（没有特殊 token 混入）
4. 能回答下面三个问题

---

## 输出问题

**Q1**：`src_padding_mask` 的作用是什么？如果不传，会发生什么？

**Q2**：这个任务的初始 loss 理论值是多少？（提示：词表大小 24，但实际只会预测 20 种氨基酸 + EOS）

**Q3**：训练 1000 步后，loss 大概降到了多少？和序列翻转任务相比，为什么这个任务更难收敛？

---

准备好后提交代码、终端输出和三个问题的回答。