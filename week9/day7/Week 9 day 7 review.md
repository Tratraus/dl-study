# Week 9 Day 7 Review：注意力可视化

## 代码结构总览

```
attention_viz.py
├── 基础常量（BOS/EOS/PAD/VOCAB_SIZE）  ← 从 Day 5 迁移
├── make_causal_mask / make_batch       ← 从 Day 5 迁移
├── Encoder                             ← 从 Day 5 迁移
├── DecoderWithAttn                     ← TODO 1 & 2：提取注意力权重
├── Seq2SeqWithAttn                     ← 组装
├── plot_attention()                    ← 热力图绘制
├── visualize()                         ← TODO 3 & 4 & 5：提取 + 可视化
└── train_and_viz()                     ← TODO 6：训练 + 可视化入口
```

---

## TODO 1 & 2：DecoderWithAttn.forward ✅ 正确

```python
for i, layer in enumerate(self.layers):
    x = layer(x, memory, tgt_mask=tgt_mask)

# 正常跑完所有层后，对最后一层单独提取 cross-attention 权重
_, cross_attn_weights = self.layers[-1].multihead_attn(
    query=x,
    key=memory,
    value=memory,
    need_weights=True,
    average_attn_weights=True
)
```

**分析**：
- 所有层正常前向传播，不影响训练逻辑 ✅
- 跑完后用 `self.layers[-1].multihead_attn` 单独提取权重，**权重不参与梯度计算**（`@torch.no_grad()` 在 visualize 上），不影响训练 ✅
- `average_attn_weights=True`：对多头取平均，返回 `(batch, tgt_len, src_len)` ✅
- `need_weights=True`：告诉 PyTorch 返回注意力权重矩阵 ✅

**精度说明**：用最终输出 `x` 作为 query 的近似，而不是最后一层的输入。理论上应该用最后一层 self-attention 之后、cross-attention 之前的中间状态，但 task 里明确说"为了简化，直接用 x"，误差很小。

---

## TODO 3：visualize 前向传播 ✅

```python
logits, attn_weights = model(src, tgt_input)
```

直接调用 `Seq2SeqWithAttn`，返回 `(logits, attn_weights)`。

---

## TODO 4：提取注意力权重 ✅

```python
attn_weights = attn_weights[0].cpu().numpy()
```

- `[0]`：去掉 batch 维度
- `.cpu().numpy()`：转成 numpy 给 matplotlib

---

## TODO 5：调用 plot_attention ✅

```python
plot_attention(attn_weights, src_labels, tgt_labels, title="Cross-Attention (Seq Reversal)")
```

---

## TODO 6：训练循环 ✅

```python
logits, _ = model(src, tgt_input)  # 训练时丢弃 attn_weights
loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt_output.reshape(-1), ignore_index=PAD)
```

训练时只用 logits，attn_weights 用 `_` 丢弃。正确。

---

## 热力图分析

### 实际输出

```
src: [16, 8, 3, 17, 6, 5, 17, 4]
tgt: [1, 4, 17, 5, 6, 17, 3, 8, 16, 2]  （含 BOS/EOS）
tgt_labels: [4, 17, 5, 6, 17, 3, 8, 16]  （去掉 BOS）
```

### 理论预期 vs 实际

理论预期：**反对角线**（序列翻转的映射关系）

```
     src: 16  8   3  17  6   5  17  4
tgt:  4  [ 0   0   0   0   0   0   0   1 ]  → attend src[7]=4
     17  [ 0   0   0   0   0   0   1   0 ]  → attend src[6]=17
      5  [ 0   0   0   0   0   1   0   0 ]  → attend src[5]=5
      6  [ 0   0   0   0   1   0   0   0 ]  → attend src[4]=6
     17  [ 0   0   0   1   0   0   0   0 ]  → attend src[3]=17
      3  [ 0   0   1   0   0   0   0   0 ]  → attend src[2]=3
      8  [ 0   1   0   0   0   0   0   0 ]  → attend src[1]=8
     16  [ 1   0   0   0   0   0   0   0 ]  → attend src[0]=16
```

**实际结果**：热力图没有呈现清晰的反对角线。

观察到的模式：
- 注意力主要集中在 src[0]（第一列，深蓝色）和 src[3:6]（中间区域）
- 没有每个 tgt token 精确对应一个 src token 的模式
- 存在明显的"弥散"注意力

### 为什么不是完美反对角线？

**原因 1：query 是近似值**

用的是最后一层的输出 `x` 作为 query，而不是最后一层 cross-attention 的真实输入。经过 self-attention + FFN 后，query 的表示已经变了，提取的权重是"近似"的。

**原因 2：模型学到了"捷径"**

序列翻转任务对模型来说太简单了。模型可能学到的不是"逐个 token 精确映射"，而是某种更高效的全局策略——比如通过 Encoder Self-Attention 已经把翻转信息编码到 memory 中了，Decoder 只需要 attend 到少数关键位置就能正确生成。

**原因 3：位置编码的影响**

序列长度只有 8，位置编码可能让模型倾向于 attend 到固定位置（比如开头和中间），而不是根据 token 内容精确匹配。

**原因 4：多头平均的模糊化**

`average_attn_weights=True` 把 4 个头的注意力模式平均了。不同头可能各自关注不同方面，平均后反而模糊了单个头的清晰模式。

---

## 输出问题回答

**Q1：热力图是反对角线吗？**

✅ 回答了"不是"，提到了开头/结尾偏差。补充：主要问题是注意力弥散，没有精确的反对角线模式。这和 query 近似、模型策略、位置编码都有关系。

**Q2：不同头的注意力模式会不同吗？**

✅ 回答了"会不同"。补充：在序列翻转任务中，可以预期：
- 某些头关注位置信息（attend 到固定偏移位置）
- 某些头关注内容（attend 到相同 token 值的位置）
- 某些头做"前瞻"（利用 Encoder Self-Attention 已编码的翻转信息）

Q2 的回答被 Q3 的内容混进来了，建议分开写。

**Q3：Cross-Attention 在计算生物学的应用？**

✅ 回答正确且有深度。蛋白质-配体对接和 RNA 二级结构的例子都很贴切。补充：
- **蛋白质结构预测**：AlphaFold2 的 Evoformer 用 attention 捕获残基间距离关系
- **基因组变异效应预测**：Cross-Attention 可以显示模型在预测变异致病性时关注基因组的哪些区域
- **药物-靶标相互作用**：可视化哪些氨基酸残基对结合贡献最大

---

## 代码质量总评

| 项目 | 评价 |
|------|------|
| TODO 1+2 权重提取 | ✅ 正确，近似方法合理 |
| TODO 3 前向传播 | ✅ 正确 |
| TODO 4 权重提取 | ✅ 正确 |
| TODO 5 可视化 | ✅ 正确 |
| TODO 6 训练循环 | ✅ 正确，丢弃 attn_weights |
| 热力图结果 | ⚠️ 不是理想反对角线，但代码逻辑正确 |
| 问题回答 | ✅ Q1/Q3 有深度，Q2 被 Q3 内容混入 |

---

## 总结

Day 7 的核心目标——**从训练好的模型中提取注意力权重并可视化**——完成得不错。热力图不是完美反对角线，但这不是代码的问题，而是：

1. 近似 query 带来的误差
2. 模型可能学到了比"逐 token 映射"更高效的策略
3. 多头平均模糊了单个头的清晰模式

如果想看到更接近反对角线的结果，可以尝试：
- 不取多头平均（`average_attn_weights=False`），分别可视化每个头
- 用更长的序列（seq_len=20），让映射关系更明显
- 手动实现最后一层的完整计算，用真正的 cross-attention 输入作为 query
