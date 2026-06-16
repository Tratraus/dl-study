# Week 9 Day 1 Review：Encoder-Decoder 结构

---

## 1. 代码结构分析

### Encoder
```
Encoder.__init__
├── self.embedding        # nn.Embedding(vocab_size, d_model, padding_idx=0)
├── self.pos_embedding    # nn.Embedding(max_len, d_model)  ← 可学习位置编码
├── self.dropout          # nn.Dropout
├── self.layers           # nn.ModuleList[TransformerEncoderLayer × N, batch_first=True]
└── self.norm             # nn.LayerNorm

Encoder.forward(src, src_key_padding_mask=None)
├── embedding + pos_embedding → (batch, src_len, d_model)
├── dropout
├── N 层循环（传入 src_key_padding_mask）
└── LayerNorm → memory (batch, src_len, d_model)
```

### Decoder
```
Decoder.__init__
├── self.embedding        # nn.Embedding(vocab_size, d_model, padding_idx=0)
├── self.pos_embedding    # nn.Embedding(max_len, d_model)
├── self.dropout          # nn.Dropout
├── self.layers           # nn.ModuleList[TransformerDecoderLayer × N, batch_first=True]
├── self.norm             # nn.LayerNorm
└── self.proj             # nn.Linear(d_model, vocab_size)  ← 新增，输出投影

Decoder.forward(tgt, memory, tgt_mask=None, tgt_key_padding_mask=None)
├── embedding + pos_embedding → (batch, tgt_len, d_model)
├── dropout
├── N 层循环（传入 memory, tgt_mask, tgt_key_padding_mask）
├── LayerNorm
└── proj → logits (batch, tgt_len, vocab_size)
```

**结构对比**：Encoder 和 Decoder 的 forward 完全对称，唯一区别：
- Decoder 每层多传一个 `memory`（Cross-Attention 的 K/V 来源）
- Decoder 最后多一个 `proj`（映射到词表大小）

---

## 2. 数据流 / 形状变化追踪

### Encoder
| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 输入 | src token ids | (batch, src_len) |
| step 1 | embedding(src) + pos_embedding(arange) | (batch, src_len, d_model) |
| step 2 | dropout | (batch, src_len, d_model) |
| step 3 | N × TransformerEncoderLayer | (batch, src_len, d_model) |
| step 4 | LayerNorm | (batch, src_len, d_model) |
| 输出 | memory | (batch, src_len, d_model) |

### Decoder
| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 输入 | tgt token ids + memory | (batch, tgt_len) + (batch, src_len, d_model) |
| step 1 | embedding(tgt) + pos_embedding(arange) | (batch, tgt_len, d_model) |
| step 2 | dropout | (batch, tgt_len, d_model) |
| step 3 | N × TransformerDecoderLayer | (batch, tgt_len, d_model) |
| step 4 | LayerNorm | (batch, tgt_len, d_model) |
| step 5 | proj | (batch, tgt_len, vocab_size) |
| 输出 | logits | (batch, tgt_len, vocab_size) |

**关键观察**：Encoder 和 Decoder 的 d_model 维度始终不变，只有最后一步 proj 才改变最后一维。

---

## 3. 关键知识点

### ① Encoder-Decoder 信息流
- Encoder **只跑一次**，输出 `memory`
- Decoder **每层都要用 `memory`**（通过 Cross-Attention）
- Decoder 的输入是"已生成的部分"，输出是"下一个 token 的概率分布"

### ② Cross-Attention 的 Q/K/V 来源
- **Self-Attention**（Encoder/Decoder 各自）：Q/K/V 都来自同一个序列
- **Cross-Attention**（Decoder 内）：Q 来自 Decoder 当前状态，K/V 来自 `memory`
- 直觉：Decoder 用 Q 去"查询" Encoder 的哪些位置最相关

### ③ Teacher Forcing
- 训练时用真实标签作为 Decoder 输入，而不是模型自己的预测
- `tgt_input = tgt[:, :-1]`（去掉最后一个 token）
- `tgt_output = tgt[:, 1:]`（去掉第一个 token，即 <BOS>）
- 推理时没有真实答案，只能自回归逐步生成

### ④ Causal Mask（因果 mask）
- Decoder 的 Self-Attention **必须**加 causal mask，防止看到未来 token
- 形状：(tgt_len, tgt_len) 的上三角矩阵，上三角为 -inf
- Encoder **不需要** causal mask（双向注意力是合理的）
- PyTorch 生成方式：`nn.Transformer.generate_square_subsequent_mask(tgt_len)`

### ⑤ batch_first=True
- PyTorch Transformer 默认 `batch_first=False`，输入 shape 为 `(seq_len, batch, d_model)`
- 设置 `batch_first=True` 后，输入 shape 为 `(batch, seq_len, d_model)`，更直观
- 省去了 forward 中反复 `.transpose(0, 1)` 的麻烦

---

## 4. 踩坑易错点

### 🔴 nn.Module 组件必须赋值给 self
```python
# ❌ 错误：创建了但没注册
nn.Embedding(vocab_size, d_model)

# ✅ 正确：赋值给 self.xxx
self.embedding = nn.Embedding(vocab_size, d_model)
```
不赋值给 self → 不会被 `parameters()` 追踪 → forward 时访问不到 → 静默失败或报错。

### 🔴 max_len 必须作为参数传入
```python
# ❌ 错误：max_len 未定义
self.pos_embedding = nn.Embedding(max_len, d_model)

# ✅ 正确：加到 __init__ 参数里
def __init__(self, ..., max_len=512, dropout=0.1):
```

### 🟡 废代码：调用但不赋值
```python
# ❌ 废代码：结果被丢弃
self.embedding(src)

# ✅ 正确：直接赋值
x = self.embedding(src) + self.pos_embedding(...)
```

### 🟡 memory_key_padding_mask 未传入
- 合成数据（等长序列，无 padding）→ 没问题
- 真实数据（变长序列，有 padding）→ 必须传入 `memory_key_padding_mask`
- 当前阶段可以先不管，Day 4 拼完整模型时再处理

### 🟡 tgt_mask 不能为 None
- 如果 `tgt_mask=None`，Decoder 的 Self-Attention 变成双向的 → 作弊
- 必须生成 causal mask：`nn.Transformer.generate_square_subsequent_mask(tgt_len)`
- Day 4 拼 `Seq2SeqTransformer` 时需要在 forward 里生成

---

## 5. 三个问题回顾

### Q1：Encoder 的输出 memory 的形状是什么？它在 Decoder 的哪个子层被使用？
**答**：(batch, src_len, d_model)。在 Decoder 每一层的第二个子层（Cross-Attention）中被使用，作为 K 和 V。

✅ 正确。补充：Cross-Attention 的 Q 来自 Decoder 自身，K/V 来自 memory。这也是为什么 Decoder 输出长度由 tgt_len 决定，而不是 src_len——Q 的长度决定了输出长度。

### Q2：Teacher Forcing 中，tgt_input 和 tgt_output 分别是什么？
**答**：
- tgt_input = [<BOS>, 3, 1, 4, 1, 5]，长度 6
- tgt_output = [3, 1, 4, 1, 5, <EOS>]，长度 6

✅ 正确。这是 Teacher Forcing 的核心——输入和输出错开一个位置，让模型学习"给定前面的 token，预测下一个 token"。

### Q3：nn.TransformerDecoderLayer 接收哪几个参数？其中哪个参数对应 memory？
**答**：接收 tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask。其中 memory 对应 Encoder 的输出。

✅ 正确。注意 `memory` 是第二个位置参数，`memory_mask` 和 `memory_key_padding_mask` 分别控制 Cross-Attention 的注意力 mask 和 padding mask。

---

## 6. 总结

Day 1 完成了 Encoder-Decoder 骨架搭建，核心收获：
- 理解了为什么序列翻译需要 Decoder（变长输出 + 逐步生成）
- 理解了 Encoder → Decoder 的信息流（memory 通过 Cross-Attention 传递）
- 掌握了 Teacher Forcing 的输入输出偏移关系
- 代码风格统一（batch_first=True、shape 注释清晰）

**待 Day 2 深入**：Cross-Attention 的具体实现（Q/K/V 的线性投影、注意力计算、输出投影）
