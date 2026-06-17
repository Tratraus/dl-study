# Week 9 Day 6 Review：蛋白质序列建模初体验

## 代码结构总览

```
protein_seq2seq.py
├── 词表定义（PAD/BOS/EOS/UNK + 20种氨基酸 → VOCAB_SIZE=24）
├── encode() / decode()
├── make_protein_batch()     ← TODO 1 & 2: padding
├── Encoder / Decoder / Seq2Seq  ← 模型改造
├── make_causal_mask()
├── train()                  ← TODO 3: 训练循环
├── greedy_decode()          ← TODO 4: 自回归推理
└── evaluate()               ← TODO 4: 评估
```

---

## TODO 1：src padding ✅ 正确

```python
src = torch.full((batch_size, max_src_len), PAD, ...)
src_padding_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool, ...)
for i, seq in enumerate(src_seqs):
    src[i, :len(seq)] = torch.tensor(seq, ...)
    src_padding_mask[i, :len(seq)] = False
```

**逻辑**：`True` = PAD 位置，`False` = 有效位置。和 PyTorch 的 `src_key_padding_mask` 语义一致。

---

## TODO 2：tgt padding ✅ 正确（已清理）

去掉了未定义的 `tgt_padding_mask` 行，只保留 tgt 填充。正确，因为：
- 训练时 loss 用 `ignore_index=PAD` 处理
- 推理时 tgt 是逐步生成的，不会有 PAD

---

## 模型改造分析（重点）

### 改动 1：`make_causal_mask` 从 float 改为 bool

```python
# Day 3/4 版本（float）
mask = torch.triu(torch.full((size, size), float('-inf'), ...), diagonal=1)

# Day 6 版本（bool）
mask = torch.triu(torch.ones(size, size, dtype=torch.bool, ...), diagonal=1)
```

**为什么改？** PyTorch 的 `TransformerDecoderLayer` 接受两种格式：
- **bool mask**：`True` 位置被屏蔽（不 attend），`False` 位置正常 attend
- **float mask**：直接加到 attention score 上（`-inf` 屏蔽）

两种都能工作，但 bool 是 PyTorch 推荐的用法。✅ 改得对。

### 改动 2：Encoder/Decoder 使用 `.unsqueeze(0)` 广播

```python
positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)  # (1, src_len)
x = self.embedding(src) + self.pos_embedding(positions)
```

**对比 Day 3/4**：`self.pos_embedding(torch.arange(...))` 直接传 1D 索引。

两种写法功能等价（`nn.Embedding` 接受任意 shape 的 LongTensor），但 `.unsqueeze(0)` 让 shape 更显式：`(1, src_len)` + `(batch, src_len, d_model)` → 广播到 `(batch, src_len, d_model)`。语义更清晰。✅

### 改动 3：Decoder 新增 `memory_key_padding_mask`

```python
def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
    ...
    x = layer(x, memory, tgt_mask=tgt_mask,
              tgt_key_padding_mask=tgt_key_padding_mask,
              memory_key_padding_mask=memory_key_padding_mask)
```

**关键发现：`memory_key_padding_mask` 定义了但没传！**

在 `Seq2Seq.forward` 中：
```python
logits = self.decoder(
    tgt_input, memory,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_padding_mask
    # ← 没传 memory_key_padding_mask！
)
```

在 `greedy_decode` 中：
```python
logits = model.decoder(generated, memory, tgt_mask=tgt_mask)
# ← 也没传 memory_key_padding_mask！
```

### ⚠️ 这是一个 bug

**`memory_key_padding_mask` 的作用**：告诉 Decoder 的 cross-attention 层在 attend Encoder 输出时，忽略 PAD 位置的 memory。

**不传的后果**：Decoder 在做 cross-attention 时，会把 Encoder 输出的 PAD 位置也当作有效信息来 attend。对于短序列（src 长度 < batch 内最长 src），Decoder 会"看到"一堆 PAD 位置的无意义向量。

**影响程度**：
- 训练时：Encoder 的 PAD 位置经过 embedding + 多层 Transformer，输出的不是全零向量，但包含噪声。Decoder 学会忽略它需要额外容量。
- 推理时：单条推理没有 PAD（batch_size=1），不影响。但 batch 推理会受影响。

**修复**：在 `Seq2Seq.forward` 中补上：
```python
logits = self.decoder(
    tgt_input, memory,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_padding_mask,
    memory_key_padding_mask=src_key_padding_mask  # ← 补上
)
```

在 `greedy_decode` 中也补上：
```python
logits = model.decoder(generated, memory, tgt_mask=tgt_mask,
                       memory_key_padding_mask=src_padding_mask)
```

---

## TODO 3：训练循环 ✅ 正确

```python
src, tgt, src_padding_mask = make_protein_batch(BATCH_SIZE, device)
tgt_input  = tgt[:, :-1]
tgt_output = tgt[:, 1:]
logits = model(src, tgt_input, src_key_padding_mask=src_padding_mask)
loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt_output.reshape(-1), ignore_index=PAD)
```

流程完全正确。唯一遗憾是没传 `memory_key_padding_mask`（见上面 bug）。

---

## TODO 4：greedy_decode + evaluate ✅ 正确

`greedy_decode` 结构清晰：
1. Encoder 编码（传了 src_padding_mask ✅）
2. BOS 初始化
3. 自回归循环 + EOS 提前退出
4. 去掉 BOS

同样有 `memory_key_padding_mask` 未传的问题。

---

## 输出分析

```
Step 200, Loss: 2.9406
Step 400, Loss: 2.9044
Step 600, Loss: 2.8994
Step 800, Loss: 2.9108
Step 1000, Loss: 2.9012
```

**Loss 几乎没下降**，从 ~2.94 到 ~2.90，基本平了。

预测输出全是 `EEEEEEEEEEEEEEEEEEEEEEEL`，模型塌陷到只输出最常见氨基酸。

### 为什么？

**根本原因：数据是随机生成的，N端和C端之间没有可学习的映射关系。**

序列翻转任务 loss 能降到 0.003 是因为 `翻转` 是确定性规则——同一个输入永远对应同一个输出。模型学到了精确映射。

而蛋白质任务中，`QQNHPPYWVYKRRG` → `NMAISNIAWLITLN` 只是无数种可能中的一种。下次生成同样的 N端，C端 完全不同。模型无法学到有意义的映射，只能学到**边缘分布**：C端里 E 出现频率最高，L 次之，所以输出 `EEE...EL`。

### 初始 loss 验证

理论值：log(21) ≈ 3.044（20种氨基酸 + EOS，均匀分布）

实际初始 loss ~2.94，接近但略低于 3.04。合理，因为：
1. 模型随机初始化不是完美均匀分布
2. 氨基酸频率不完全均匀（20种里随机采样有波动）

### 预测全是 E 的解释

模型学到了**边缘概率** P(amino acid) ≈ E 最高。这是"坍缩到均值"的经典表现——当条件分布 P(C端|N端) 没有规律时，最优策略就是输出边缘分布中最常见的 token。

---

## 输出问题回答

**Q1: src_padding_mask 的作用是什么？如果不传，会发生什么？**

✅ 回答正确。补充：不传时 Encoder 的 self-attention 会把 PAD 的 embedding 也纳入计算，污染有效 token 的表示。对短序列（PAD 多）影响更大。

**Q2: 初始 loss 理论值？**

✅ 回答正确。log(21) ≈ 3.04。实际 ~2.94 接近，合理。

**Q3: 为什么更难收敛？**

✅ 回答正确。补充一个关键点：序列翻转是**确定性映射**（1个输入→1个输出），而随机蛋白质是**一对多映射**（1个N端→无数种C端）。模型根本不存在"收敛到正确答案"的可能性，只能学到边缘分布。这不是模型容量问题，是任务本身的问题。

---

## 与 Day 3/4 的数据流对比

| 环节 | Day 3/4（序列翻转） | Day 6（蛋白质） |
|------|---------------------|-----------------|
| src | (batch, 10)，无 PAD | (batch, ~10-30)，有 PAD |
| tgt | (batch, 12)，BOS+翻转+EOS | (batch, ~12-32)，BOS+C端+EOS |
| src_padding_mask | 不需要 | ✅ 必须 |
| memory_key_padding_mask | 不需要 | ⚠️ 应该传但没传 |
| Encoder forward | 无 mask | 传 src_key_padding_mask |
| Seq2Seq.forward | 内部算 `(src==PAD)` | 外部传入 |
| Loss | 可降到 ~0 | 只能降到 ~2.9（任务本质） |

---

## 总结

| 项目 | 评价 |
|------|------|
| TODO 1-2 (padding) | ✅ 正确 |
| TODO 3 (训练循环) | ✅ 正确，缺 memory_key_padding_mask 传递 |
| TODO 4 (greedy_decode) | ✅ 正确，同上 |
| 模型改造 | ✅ bool mask / .unsqueeze(0) 改进合理 |
| Bug | ⚠️ `memory_key_padding_mask` 定义了但没传给 Decoder |
| 输出问题回答 | ✅ 全部正确，Q3 可补充"确定性 vs 一对多" |
| 输出现象 | 符合预期——随机数据坍缩到边缘分布是正常行为 |
