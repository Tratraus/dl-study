顺便回答一下思考题：把 c_term 改成 n_term 的某种变换（比如翻转、移位），这样 N 端和 C 端之间就有了确定性的统计关联，任务就变得可学习了。

---

# Week 9 Day 7：注意力可视化

## 任务目标

把训练好的 Seq2Seq 模型的**注意力权重**提取出来，用热力图可视化，直观理解模型"在看哪里"。

---

## 最小必要理论

### 1. 注意力权重是什么

回忆 Attention 的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

中间那个矩阵 $$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$ 就是**注意力权重矩阵**，shape 为 `(tgt_len, src_len)`。

$$A[i][j]$$ 表示：**生成第 i 个输出 token 时，对第 j 个输入 token 的关注程度**。

### 2. 三种注意力的含义

```text
Encoder Self-Attention：
  src 中每个位置对其他位置的关注
  → 理解输入序列的内部结构

Decoder Self-Attention（Causal）：
  tgt 中每个位置对之前位置的关注
  → 生成时的自回归依赖

Decoder Cross-Attention：
  tgt 中每个位置对 src 的关注
  → 最直观：模型在生成每个输出时"看"了输入的哪里
```

今天重点可视化 **Cross-Attention**，这是最有解释性的部分。

### 3. 序列翻转任务的理论预期

```text
src:  [3, 7, 5, 2, 4]
tgt:  [4, 2, 5, 7, 3]

理论上 Cross-Attention 应该是反对角线：
  生成 tgt[0]=4 时，应该 attend 到 src[4]=4
  生成 tgt[1]=2 时，应该 attend 到 src[3]=2
  ...

可视化结果应该接近：
     src: 3  7  5  2  4
tgt: 4  [ 0  0  0  0  1 ]
     2  [ 0  0  0  1  0 ]
     5  [ 0  0  1  0  0 ]
     7  [ 0  1  0  0  0 ]
     3  [ 1  0  0  0  0 ]
```

---

## 代码任务

新建文件：`week9/day7/attention_viz.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 复用之前所有模型代码 ──────────────────────────────────────
# 把 Day 5 的完整代码粘贴过来（Encoder/Decoder/Seq2Seq/make_causal_mask/...）
# 训练完后我们来提取注意力权重

# ── 带注意力权重输出的 Decoder ────────────────────────────────
class DecoderWithAttn(nn.Module):
    """
    在原有 Decoder 基础上，额外返回每一层的 Cross-Attention 权重。

    PyTorch 的 TransformerDecoderLayer 默认不返回注意力权重，
    需要用 need_weights=True 手动调用底层的 MultiheadAttention。

    为了简单，我们只提取最后一层的 Cross-Attention 权重。
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff,
                                       dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        x = self.embedding(tgt)
        x = x + self.pos_embedding(
            torch.arange(x.size(1), device=tgt.device)
        )
        x = self.dropout(x)

        cross_attn_weights = None  # 只保存最后一层

        for i, layer in enumerate(self.layers):
            # TODO 1：对最后一层，手动调用底层 MHA 提取注意力权重
            # 其他层正常调用 layer(x, memory, tgt_mask=tgt_mask)
            #
            # 提示：TransformerDecoderLayer 的底层结构：
            #   layer.self_attn     → Decoder Self-Attention
            #   layer.multihead_attn → Cross-Attention  ← 我们要这个
            #   layer.norm1, norm2, norm3
            #   layer.linear1, linear2
            #
            # 手动调用 Cross-Attention 的方式：
            #   attn_output, attn_weights = layer.multihead_attn(
            #       query=...,   ← 来自 Self-Attention 之后的 x
            #       key=memory,
            #       value=memory,
            #       need_weights=True,
            #       average_attn_weights=True  # 对多头取平均
            #   )
            #
            # 但要完整复现一层的计算，还需要处理：
            #   残差连接、LayerNorm、FFN
            # 这比较繁琐，今天用一个更简单的方法：
            #
            # 简单方法：先正常跑完所有层，
            # 然后单独对最后一层的输入再跑一次 multihead_attn 提取权重
            # （权重不影响前向传播，只用于可视化）
            x = layer(x, memory, tgt_mask=tgt_mask)

        # TODO 2：提取最后一层的 Cross-Attention 权重
        # 此时 x 是最后一层 FFN 之后的输出，但我们需要最后一层的输入
        # 更简单的做法：重新跑一次最后一层的 multihead_attn
        # 注意：这里的输入应该是倒数第二层的输出，即最后一层 self_attn 之后的结果
        # 为了简化，直接用 x（最终输出）作为 query 的近似——误差很小
        #
        # 提示：
        # _, cross_attn_weights = self.layers[-1].multihead_attn(
        #     query=x,
        #     key=memory,
        #     value=memory,
        #     need_weights=True,
        #     average_attn_weights=True
        # )
        # cross_attn_weights shape: (batch, tgt_len, src_len)
        ...

        x = self.norm(x)
        logits = self.proj(x)
        return logits, cross_attn_weights


# ── 带注意力输出的 Seq2Seq ────────────────────────────────────
class Seq2SeqWithAttn(nn.Module):
    """
    用 DecoderWithAttn 替换原来的 Decoder，其余不变。
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.decoder = DecoderWithAttn(vocab_size, d_model, num_heads,
                                        num_layers, d_ff)

    def forward(self, src, tgt_input):
        tgt_len = tgt_input.size(1)
        tgt_mask = make_causal_mask(tgt_len, tgt_input.device)
        src_key_padding_mask = (src == PAD)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        logits, attn_weights = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
        return logits, attn_weights


# ── 注意力可视化 ──────────────────────────────────────────────
def plot_attention(attn_weights, src_tokens, tgt_tokens, title="Cross-Attention"):
    """
    用热力图可视化 Cross-Attention 权重。

    attn_weights: (tgt_len, src_len) numpy array
    src_tokens:   List[str]，x 轴标签
    tgt_tokens:   List[str]，y 轴标签
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=12)
    ax.set_yticklabels(tgt_tokens, fontsize=12)
    ax.set_xlabel("Source (input)", fontsize=12)
    ax.set_ylabel("Target (output)", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("attention.png", dpi=150)
    plt.show()
    print("已保存 attention.png")


# ── 提取并可视化 ──────────────────────────────────────────────
@torch.no_grad()
def visualize(model, device):
    model.eval()

    # 构造一条样本
    src, tgt = make_batch(1, 8, device)
    tgt_input = tgt[:, :-1]

    # TODO 3：前向传播，获取 logits 和 attn_weights
    ...

    # TODO 4：从 attn_weights 中提取 EOS 之前的部分
    # attn_weights shape: (1, tgt_len, src_len)
    # 去掉 batch 维度，转成 numpy
    ...

    # 构造标签
    src_labels = [str(t) for t in src[0].tolist()]
    # tgt_input 是 [BOS, t1, t2, ...]，对应的预测是 [t1, t2, ..., EOS]
    # y 轴标签用预测的 token（即 tgt[:, 1:]）
    tgt_labels = [str(t) for t in tgt[0, 1:].tolist()]  # 去掉 BOS

    # TODO 5：调用 plot_attention
    ...

    print(f"src:      {src[0].tolist()}")
    print(f"tgt:      {tgt[0].tolist()}")


# ── 训练 + 可视化 ─────────────────────────────────────────────
def train_and_viz():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL    = 64
    NUM_HEADS  = 4
    NUM_LAYERS = 2
    D_FF       = 128
    BATCH_SIZE = 64
    SEQ_LEN    = 10
    STEPS      = 500

    model = Seq2SeqWithAttn(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, STEPS + 1):
        src, tgt = make_batch(BATCH_SIZE, SEQ_LEN, device)
        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        # TODO 6：注意 Seq2SeqWithAttn.forward 返回 (logits, attn_weights)
        # 训练时只用 logits
        ...
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("训练完成")
    visualize(model, device)


if __name__ == "__main__":
    train_and_viz()
```

---

## 完成标准

1. 代码无报错，训练正常收敛
2. 生成 `attention.png`，热力图**接近反对角线**（序列翻转任务的理论预期）
3. 能回答下面三个问题

---

## 输出问题

**Q1**：你的热力图是反对角线吗？如果不是完美的反对角线，哪些位置有偏差？试着解释为什么。

**Q2**：`average_attn_weights=True` 对多头取平均。如果不取平均，每个头的注意力模式会不同吗？在序列翻转任务里，你预期不同的头会关注什么？

**Q3**：Cross-Attention 可视化在计算生物学中有什么实际应用？（提示：想想蛋白质-配体对接、RNA 二级结构预测）

---

准备好后提交代码、`attention.png` 截图描述和三个问题的回答。