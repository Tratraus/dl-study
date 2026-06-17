# Week 9 Review：序列到序列模型

## 本周全景

Week 9 从零实现了完整的 Seq2Seq Transformer，覆盖了从模型搭建到推理策略到可解释性的全链路。

```
Day 1: Encoder-Decoder 骨架
Day 2: Cross-Attention 原理
Day 3: 完整 Seq2Seq + Teacher Forcing 训练
Day 4: Greedy Decoding（自回归推理）
Day 5: Beam Search（多路径搜索）
Day 6: 蛋白质序列建模（真实任务迁移）
Day 6.5: 采样策略拓展（Temperature / Top-k / 长度归一化）
Day 7: 注意力可视化（Cross-Attention 热力图）
```

---

## 技术对比表

### 模型结构对比

| 组件 | Encoder | Decoder | Seq2Seq |
|------|---------|---------|---------|
| Embedding | ✅ | ✅ | — |
| Positional Encoding | ✅ | ✅ | — |
| Self-Attention | ✅ 双向 | ✅ 单向（causal mask） | — |
| Cross-Attention | ❌ | ✅（Q=Decoder, K/V=memory） | — |
| FFN | ✅ | ✅ | — |
| LayerNorm | ✅ | ✅ | — |
| 输出投影 | ❌ | ✅ Linear(d_model, vocab) | — |
| forward 输入 | src | tgt + memory | src + tgt_input |
| forward 输出 | memory | logits | logits |

### 解码策略对比

| 策略 | 确定性 | 搜索宽度 | 时间复杂度 | 适用场景 |
|------|--------|---------|-----------|---------|
| Greedy | ✅ 确定 | 1 | O(n) | 简单任务、实时推理 |
| Beam Search | ✅ 确定 | beam_width | O(n × beam_width) | 翻译、摘要（早期歧义大） |
| Beam + 长度归一化 | ✅ 确定 | beam_width | 同上 | 避免偏好短序列 |
| Temperature | ❌ 随机 | 1 | O(n) | 需要多样性（对话、创作） |
| Top-k | ❌ 随机 | k | O(n) | 截断长尾，比 Temperature 更可控 |

### 注意力类型对比

| 类型 | Q 来源 | K/V 来源 | 作用 | mask |
|------|--------|---------|------|------|
| Encoder Self-Attention | src | src | 理解输入内部结构 | padding mask |
| Decoder Self-Attention | tgt | tgt | 自回归依赖 | causal mask + padding mask |
| Cross-Attention | tgt | memory | 连接输入和输出 | padding mask |

---

## 关键收获

### 1. Encoder-Decoder 的信息流

```
src → [Encoder] → memory → [Decoder Cross-Attention] → logits
                                          ↑
tgt_input → [Decoder Self-Attention] ----┘
```

- Encoder 只跑一次，输出 memory
- Decoder 每层用 memory（通过 Cross-Attention）
- 输出长度由 Decoder 的 Q 决定，不受 src_len 限制

### 2. Teacher Forcing vs 自回归推理

| 维度 | Teacher Forcing（训练） | 自回归推理 |
|------|----------------------|-----------|
| Decoder 输入 | 真实 tgt（错开一位） | 模型自己的预测 |
| 并行性 | ✅ 全部位置并行计算 | ❌ 逐步生成 |
| 优势 | 训练稳定、收敛快 | 反映真实推理场景 |
| 劣势 | exposure bias（训练/推理不一致） | 误差累积 |

### 3. Padding Mask 的三层传递

```
src_key_padding_mask  → Encoder（忽略 src 中的 PAD）
tgt_key_padding_mask  → Decoder Self-Attention（忽略 tgt 中的 PAD）
memory_key_padding_mask → Decoder Cross-Attention（忽略 memory 中的 PAD）
```

Day 6 发现的 bug：`memory_key_padding_mask` 定义了但没传，导致 Decoder 会 attend 到 Encoder 的 PAD 位置。

### 4. 任务可学习性

Day 6 的核心发现：**N 端和 C 端独立随机生成 → 任务不可学习 → 模型退化为预测高频 token（全 E）**

确定性映射（序列翻转）：loss → 0.003，100% 准确率
随机映射（蛋白质）：loss → 2.9（≈ log(21)），输出全是 E

**教训**：可学习性是任务设计的前提。没有可学习的映射关系，模型再大也没用。

### 5. 近似提取注意力权重

Day 7 的方法：正常跑完所有层后，用最终输出 `x` 作为 query 重新跑最后一层的 `multihead_attn`。

- 优点：简单，不需要修改模型结构
- 缺点：query 是近似值（不是真正的 cross-attention 输入），热力图有误差

---

## 踩坑易错点汇总

| Day | 坑 | 原因 | 修复 |
|-----|-----|------|------|
| Day 3 | `view` vs `reshape` | 切片后 tensor 不连续 | 用 `reshape` |
| Day 4 | seq_len 8 vs 10 推理全错 | 训练/推理输入分布不一致 | 推理时用和训练相同的 seq_len |
| Day 6 | `tgt_padding_mask` 未定义 | 删掉多余代码 | 直接删除该行 |
| Day 6 | `memory_key_padding_mask` 没传 | forward 参数有了但没用 | 补上参数传递 |
| Day 6.5 | 采样策略输出和 Greedy 一样 | 模型太确定（loss 0.003），分布极端尖锐 | 在不确定任务上才有效 |
| Day 7 | 热力图不是反对角线 | 近似 query + 多头平均 + 模型捷径 | 分头可视化 / 更长序列 |

---

## 代码模板沉淀

Week 9 积累的可复用代码模块：

| 模块 | 文件 | 复用场景 |
|------|------|---------|
| `Encoder` 类 | Day 1~7 | Week 10 的 ProteinBERT 直接复用 |
| `Decoder` 类 | Day 1~5 | Seq2Seq 任务 |
| `make_causal_mask` | Day 3 | 任何需要因果 mask 的 Decoder |
| `make_batch`（序列翻转） | Day 3 | 快速验证模型正确性 |
| `make_protein_batch` | Day 6 | 变长序列 + padding + mask |
| `greedy_decode` | Day 4 | 任何自回归推理 |
| `beam_search` | Day 5 | 需要多路径搜索时 |
| `temperature_sample` / `topk_sample` | Day 6.5 | 需要多样性生成时 |
| `DecoderWithAttn` | Day 7 | 注意力权重提取 |
| `plot_attention` | Day 7 | 注意力可视化 |

---

## 技能树更新

### 新增技能（Week 9）

**模型结构**
- [x] Transformer Encoder（双向 Self-Attention）
- [x] Transformer Decoder（Causal Self-Attention + Cross-Attention）
- [x] Seq2Seq 组装（Encoder + Decoder + mask 传递）
- [x] Encoder-only 结构理解（BERT 式，无 Decoder）

**训练范式**
- [x] Teacher Forcing（tgt_input / tgt_output 错位切分）
- [x] `ignore_index=PAD` 的作用
- [x] view vs reshape 的内存连续性问题

**推理策略**
- [x] Greedy Decoding（自回归循环）
- [x] Beam Search（beam 扩展 + 剪枝 + done beam 保留）
- [x] 长度归一化 Beam Search
- [x] Temperature Sampling
- [x] Top-k Sampling

**生物序列处理**
- [x] 氨基酸词表设计（20 种 + 特殊 token）
- [x] 变长序列 padding + mask 生成
- [x] 任务可学习性判断

**可解释性**
- [x] Cross-Attention 权重提取
- [x] 热力图可视化与解读

---

## 与 Week 7/8 的衔接

| 周 | 核心能力 | 为 Week 9 铺垫了什么 |
|-----|---------|---------------------|
| W7 | ESM-2 迁移学习 | Encoder 结构、分层学习率、冻结策略 |
| W8 | Adapter / LoRA / 多任务 | 参数高效微调思想、loss 设计 |
| W9 | Seq2Seq 完整实现 | Decoder + Cross-Attention + 推理策略 |

**Week 10 衔接**：Day 6 的氨基酸词表 + Day 7 的 Encoder 直接复用到 ProteinBERT。Week 9 的 `Encoder` 类去掉 Decoder 就是 BERT 的骨架。

---

## 周评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 完成度 | ✅ 8/8 | Day 1~7 + Day 6.5 全部完成 |
| 代码质量 | ✅ | 结构清晰，shape 注释完整 |
| 知识理解 | ✅ | 输出问题回答准确率高，Q2 有时可更精确 |
| 主动探索 | ⭐ | Day 6.5 采样策略、主动修复 memory_key_padding_mask bug |
| 踩坑学习 | ✅ | view/reshape、seq_len 不匹配、mask 传递等都有深入分析 |
| 生物迁移 | ✅ | 从合成任务迁移到蛋白质序列，发现了可学习性问题 |
