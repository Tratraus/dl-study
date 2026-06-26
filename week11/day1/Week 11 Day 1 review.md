# Week 11 Day 1 Review：加载 ESM-2，提取真实蛋白质嵌入

## 代码结构分析

### 5 个函数的职责

| 函数 | 输入 | 输出 | 职责 |
|------|------|------|------|
| `load_esm2()` | model_name, device | tokenizer, model | 加载预训练权重，切换 eval |
| `embed_batch()` | tokenizer, model, sequences, device | (B, 320) tensor | 单批次：编码→前向→去特殊token→Mean Pool |
| `extract_embeddings()` | tokenizer, model, sequences, batch_size, device | (N, 320) numpy | 分批调用 embed_batch，拼接收集 |
| `print_model_info()` | model | None | 打印配置（层数/d_model/头数） |
| `main()` | — | — | 组装所有步骤，输出结果 |

### 调用链

```
main()
  ├─ load_esm2()           → tokenizer, model
  ├─ extract_embeddings()  → (10, 320) numpy
  │    └─ embed_batch() ×3  → 每批 (4, 320) 或 (2, 320)
  ├─ tokenizer(单条)       → 验证 input_ids shape
  └─ 余弦相似度计算        → 5×5 矩阵
```

## 数据流 / Shape 变化追踪

```
输入序列（字符串）
  │
  ▼ tokenizer(sequences, padding=True, truncation=True, max_length=512)
input_ids:      (B, L+2)    ← +2 是 <cls> 和 <eos>
attention_mask: (B, L+2)    ← 1=有效, 0=PAD
  │
  ▼ model(input_ids=..., attention_mask=...)
outputs.last_hidden_state:  (B, L+2, 320)
  │
  ▼ hidden[:, 1:-1, :]
(B, L, 320)                 ← 去掉首尾特殊 token
  │
  ▼ hidden.mean(dim=1)
(B, 320)                    ← 最终嵌入
```

## 关键知识点

### 1. HuggingFace tokenizer 的输出
- `input_ids`: 氨基酸 → token id 的整数序列，首尾自动加 `<cls>`(=0) 和 `<eos>`(=2)
- `attention_mask`: 与 input_ids 同 shape，1 表示真实 token，0 表示 PAD
- `padding=True`: 自动将 batch 内短序列补 PAD 到最长
- `truncation=True`: 超过 max_length 的截断

### 2. attention_mask 与 src_key_padding_mask 的关系
- **本质相同**：都是告诉模型"哪些位置是填充的，不要参与注意力计算"
- HuggingFace 的 `attention_mask`: 1=有效, 0=PAD
- PyTorch 的 `src_key_padding_mask`: True=被屏蔽（不参与）, False=有效
- **逻辑相反**！`src_key_padding_mask = 1 - attention_mask`（或 `~attention_mask.bool()`）
- 传给 `model(**inputs)` 时，HuggingFace 内部会自动转换

### 3. Mean Pooling 去掉特殊 token 的原因
- `<cls>` 和 `<eos>` 是模型用来标记序列边界的"标点符号"
- 它们的语义是"序列开始/结束"，不是氨基酸的特征
- 如果混入 Mean Pooling，会稀释真实氨基酸的信息
- 特别是短序列：特殊 token 占比高，污染更严重

### 4. ESM-2 vs ProteinBERT 定量对比

| 维度 | ProteinBERT | ESM-2 8M |
|------|-------------|----------|
| 参数量 | ~470K | 7,511,801 |
| 层数 | 3 | 6 |
| d_model | 128 | 320 |
| 注意力头数 | 4 | 20 |
| 嵌入维度 | 128 | 320 |
| 训练数据 | 合成随机序列 | 2.5 亿真实蛋白质 |

### 5. 余弦相似度分析
- 血红蛋白 vs 肌红蛋白：0.953（最高）— 两者都是珠蛋白家族，功能相似（氧气结合）
- 胰岛素 vs GFP：0.792（最低）— 功能完全不同
- 但所有相似度都在 0.79~0.95 之间，差距不大
- 原因：ESM-2 8M 是最小版本，嵌入区分度有限；更大的模型（150M/3B）会有更明显的差异

## 踩坑 / 易错点

1. **UNEXPECTED / MISSING 权重警告**：正常现象。ESM-2 预训练时有 `lm_head`（MLM 头）和 `pooler`，但 `AutoModel` 只加载主干，所以 lm_head 权重 "unexpected"，pooler 权重 "missing"。不影响嵌入提取。

2. **attention_mask 取出后要 `.to(device)`**：不能只移 input_ids 不移 mask，否则设备不匹配报错。

3. **`torch.no_grad()` 的作用**：推理时不需要梯度，关闭后节省显存和计算。和 Week 10 的 `model.eval()` 配合使用——eval 关闭 dropout，no_grad 关闭梯度追踪。

## 输出问题回顾

**Q1**: input_ids 是 token 索引序列，attention_mask 标记有效/PAD 位置。与 src_key_padding_mask 本质相同但逻辑相反（1=有效 vs True=屏蔽）。

**Q2**: 首尾是 `<cls>` 和 `<eos>`，不属于蛋白质序列。混入会稀释真实氨基酸信息。

**Q3**: 血红蛋白 vs 肌红蛋白 0.953，同属珠蛋白家族，功能相似（氧气运输/储存），所以嵌入最接近。
