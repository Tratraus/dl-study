# Week 10 Day 2 Review — Encoder-only ProteinBERT 实现

## 代码结构分析

```
protein_bert.py
├── 词表定义（从 day1 import）
├── PositionalEncoding    — 可学习位置编码 (nn.Embedding)
├── ProteinBERT           — BERT 式蛋白质语言模型
│   ├── embedding         — Token Embedding (vocab_size → d_model)
│   ├── pe                — PositionalEncoding
│   ├── dropout           — Dropout
│   ├── layers            — TransformerEncoderLayer × 3 (nn.ModuleList)
│   ├── norm              — LayerNorm
│   └── mlm_head          — Linear(d_model, vocab_size)
└── verify_model()        — 验证 shape + 参数量
```

## 数据流 / 形状变化

```
src: (batch, seq_len)
  ↓ self.embedding
x: (batch, seq_len, d_model)        # 128
  ↓ self.pe (+ 位置编码)
x: (batch, seq_len, d_model)
  ↓ self.dropout
x: (batch, seq_len, d_model)
  ↓ TransformerEncoderLayer × 3     # 逐层，传入 src_key_padding_mask
x: (batch, seq_len, d_model)
  ↓ self.norm
x: (batch, seq_len, d_model)
  ↓ self.mlm_head
logits: (batch, seq_len, vocab_size) # 25
```

## 关键知识点

### 1. nn.ModuleList vs 普通 list
- 初始代码用了 `nn.TransformerEncoderLayer(...)` 单个对象，`for layer in self.layers` 报错
- 修复：`nn.ModuleList([...])` 包裹 num_layers 个 layer
- `nn.ModuleList` 会把内部模块注册到模型中（`parameters()`、`to()` 等都会递归处理）
- 普通 Python list 不会注册，梯度不回传，`model.to(device)` 也不会移动里面的参数

### 2. batch_first=True
- `TransformerEncoderLayer` 默认输入 shape 是 `(seq_len, batch, d_model)`
- 设 `batch_first=True` 后变为 `(batch, seq_len, d_model)`，和 Embedding 输出一致
- 不设的话维度对不上，会报错或结果错误

### 3. padding_idx=PAD
- `nn.Embedding(vocab_size, d_model, padding_idx=PAD)`
- PAD 位置的 embedding 向量永远是零向量，不参与梯度更新
- 确保 PAD token 不会干扰模型学习

### 4. 输出是 logits 不是概率
- `mlm_head` 输出 raw logits，不加 softmax
- `F.cross_entropy` 内部会做 log_softmax + NLL，数值更稳定
- 如果模型加了 softmax 再传给 cross_entropy，等于做了两次 softmax → 梯度异常

## 参数量分析

| 模块 | 参数量 | 计算方式 |
|------|--------|----------|
| embedding | 3,200 | 25 × 128 |
| pe | 65,536 | 512 × 128 |
| layers | 397,440 | 3 × TransformerEncoderLayer |
| norm | 256 | 128 × 2 (weight + bias) |
| mlm_head | 3,225 | 128 × 25 + 25 |
| **总计** | **469,657** | 全部可训练 |

- PE 参数量最大（65,536），因为 max_len=512 × d_model=128
- layers 占 84.6%，符合 Transformer 的参数分布规律

## 踩坑 / 易错点

- **`nn.ModuleList`**：本次主要踩坑点。单个 `TransformerEncoderLayer` 不能用 `for` 迭代，必须用 `nn.ModuleList` 包裹。这是 PyTorch 新手常见错误。
- **`batch_first=True`**：不加的话默认 `(seq_len, batch, d_model)`，和 Embedding 输出的 `(batch, seq_len, d_model)` 维度不匹配。

## 输出问题 Review

### Q1 回答
> "去掉了 Decoder 部分，增加了 MLM Head"

✅ 正确但可以更具体。两处关键差异：
1. **去掉 Decoder** → 不需要 Causal Mask，Encoder 变为双向
2. **加 MLM Head** → 输出从 `(batch, seq_len, d_model)` 变为 `(batch, seq_len, vocab_size)`

### Q2 回答
> "src_key_padding_mask 会标记 PAD 位置，确保 Transformer 在计算注意力时不会关注这些填充位置"

✅ 准确。`src_key_padding_mask` 中 `True` 的位置会被 attention 忽略（softmax 前设为 -∞），防止 PAD 信息干扰正常 token 的表示。

### Q3 回答
> "CrossEntropyLoss 会自动将 logits 转换为概率分布并计算损失"

✅ 正确。补充一点：`F.cross_entropy` 内部做的是 `log_softmax + NLLLoss`，一步到位，数值上比手动 `softmax` 再算 `log` 更稳定（避免 log(0) 或 log(极小数) 的问题）。

---

**Day 2 完成** ✅ — ProteinBERT 模型搭建完成，三个 TODO 全部实现，Q1-Q3 回答到位。
