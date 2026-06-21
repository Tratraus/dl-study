# Week 10 Day 1 Review — MLM 原理 + 数据构造

## 代码结构分析

```
mlm_data.py
├── 词表定义（25 tokens：PAD/BOS/EOS/UNK/MASK + 20 氨基酸）
├── encode() / decode()  — 字符串 ↔ id 转换
├── mask_sequence()      — TODO 1：单条序列的 80-10-10 mask
├── make_mlm_batch()     — TODO 2：批量生成 MLM 训练数据
└── verify_batch()       — TODO 3：可视化验证
```

## 数据流 / 形状变化

```
random.choices(AA_IDS, k=seq_len)       → list[int], 长度 seq_len
    ↓
mask_sequence(seq)                       → (masked_ids: list[int], labels: list[int])
    ↓  × batch_size 次，append 到 all_masked / all_labels
    ↓
torch.tensor(all_masked, device=device)  → (batch, seq_len) long
torch.tensor(all_labels, device=device)  → (batch, seq_len) long
```

`verify_batch` 中额外返回 `original` tensor，方便三列对比验证。

## 关键知识点

### 1. MLM 的核心设计：只对 mask 位置算 loss
- `labels` 中未被选中的位置填 `-100`，`F.cross_entropy(ignore_index=-100)` 自动跳过
- 如果对全序列算 loss，模型可以直接"抄"输入，学不到上下文推理能力

### 2. 80-10-10 策略解决 distribution shift
- 训练时 80% → `[MASK]`，但推理时输入没有 `[MASK]`
- 10% 随机替换 + 10% 保持不变，让模型对所有位置都保持"需要预测"的警觉
- 这是 BERT 论文的关键 trick 之一

### 3. Encoder-only vs Seq2Seq 注意力方向
- Seq2Seq Decoder：单向（Causal Mask），只能看左边
- Encoder-only (BERT)：双向，每个位置能看到整个序列
- 蛋白质场景：功能/结构依赖远程残基交互，双向注意力天然更合适

## 踩坑 / 易错点

- **`return_original` 参数**：主人在 `make_mlm_batch` 中加了 `return_original` 开关，`verify_batch` 中设为 `True` 方便三列对比。这个扩展设计合理，但注意接口一致性——Day 2 的模型如果直接调用 `make_mlm_batch` 要确保传的参数对。
- **token 名已从 `***` 改为正常字符串**（`<PAD>`、`<MASK>` 等），比之前清晰。

## 输出验证

从贴出的输出看：
- `masked_src` 和 `labels` shape 均为 `(4, 20)` ✅
- 位置 1：`H → <MASK>`，label=11 ✅（H 在词表中确实是 id 11）
- 位置 15：`M → <MASK>`，label=15 ✅
- 位置 19：`I → Y`，label=12 ✅（10% 随机替换，label 仍为原始 I 的 id）
- mask 比例 3/20 = 15.0% ✅

## 输出问题 Review

### Q1 回答
> "如果对整个序列计算 loss，模型可能会过度关注非 mask 位置"

✅ 方向对。更精确的说法：对非 mask 位置算 loss 时，模型可以直接复制输入 token（identity shortcut），不需要学习上下文推理，MLM 的训练目标就失效了。

### Q2 回答
> "推理时输入中不会有 [MASK] token...通过随机替换和保持不变，模型可以学会更鲁棒地处理各种情况"

✅ 准确。这就是 **distribution shift** 问题——训练分布和推理分布不一致。80-10-10 让模型在三种输入状态下都能训练，缓解了这个问题。

### Q3 回答
> "Seq2Seq 模型（如 GPT）通常是单向的"

⚠️ 小修正：GPT 不是 Seq2Seq，是 **Decoder-only** 架构。Seq2Seq = Encoder + Decoder（如 T5、原始 Transformer）。分类应该是：
- **Encoder-only**：BERT，双向
- **Decoder-only**：GPT，单向（Causal）
- **Encoder-Decoder**：Seq2Seq，Encoder 双向 + Decoder 单向

其余内容准确。

---

**Day 1 完成** ✅ — MLM 数据构造 pipeline 跑通，三个 TODO 全部完成，Q1-Q2 到位，Q3 有一个小概念修正。
