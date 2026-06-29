# Week 12 · Day 7 Review：收官 + CLRS 复杂度分析

## 一、任务概述

Day 7 是纯理论收官日，不写代码，做三件事：
1. Self-Attention 复杂度推导（时间 + 空间）
2. 线性 Attention 的动机与方案对比
3. Week 12 总结 + 三模型对比表

---

## 二、复杂度推导评价

### 推导过程

任务里的推导是正确的，四个步骤清晰：

| 步骤 | 计算 | 复杂度 |
|------|------|--------|
| Q/K/V 投影 | 三次矩阵乘 $XW$ | $O(nd^2)$ |
| $QK^T$ | $n \times d_k$ 乘 $d_k \times n$ | $O(n^2 d_k)$ |
| Softmax | 逐行归一化 | $O(n^2)$ |
| $AV$ | $n \times n$ 乘 $n \times d_k$ | $O(n^2 d_k)$ |

**合计**：$O(n^2 d + nd^2)$

### 关键数字题

$n: 1024 \to 4096$（4 倍），$QK^T$ 计算量增加 **16 倍**。正确。

补充一个实际感受：ESM-2 8M 的 $d=320$，$n=512$ 时注意力矩阵大小为 $512^2 \times 320 \approx 83M$ 浮点数（~330MB FP32）。如果 $n=4096$，这个数字变成 $4096^2 \times 320 \approx 5.4B$（~21GB FP32），一张 24GB 的 4090 就放不下一个 batch 的注意力矩阵了。

---

## 三、线性 Attention 理解评价

### 任务表格理解

任务给出的对比表是 Week 11 的 Evo 论文背景知识延伸：

| 方法 | 时间 | 核心思路 |
|------|------|---------|
| 标准 Attention | $O(n^2 d)$ | 精确计算 |
| Linformer | $O(nkd)$ | 低秩投影 $K,V \to k$ 维 |
| Performer | $O(nd^2)$ | 随机特征近似 softmax 核 |
| FlashAttention | $O(n^2 d)$ | IO 优化，分块计算 |

**关键区分**：
- Linformer/Performer 是**算法级优化**（降低理论复杂度）
- FlashAttention 是**系统级优化**（不改复杂度，改 IO 调度）
- FlashAttention 的实际加速来自 **HBM → SRAM 的带宽差**（A100: HBM 2TB/s vs SRAM 19TB/s，~10× 差距）

---

## 四、输出问题回答评价

### Q1 评价

> "ESM2作为预训练模型，其分类头分工更为优秀，而自实现模型的各头注意力趋同。"

**太表面了**。问题问的是「根本原因」，这个回答只是重复了观察到的现象（分工 vs 趋同），没有解释**为什么**。

**更好的回答**：

ESM-2 的对角线模式 = **局部语法**（关注相邻残基的序列上下文），竖条纹模式 = **功能锚点**（特定位置被全局关注）。这种分化来自**预训练的进化压力**：

- ESM-2 在 2.5 亿蛋白质上做 MLM 预测，被迫学会：哪些残基经常共变（远距离相关）、哪些残基只和邻居相关（局部模式）
- 这种压力自然导致不同 head 分化出不同功能：有的负责局部上下文（对角线），有的负责远距离依赖（列型）
- 自实现模型只有 5871 条数据，没有足够的进化信号让 head 分化，所以所有 head 都收敛到同一个简单的「看邻居」策略

**根本原因**：数据量和训练目标的差异，导致注意力头是否发生功能分化。

### Q2 评价

> "FlashAttention通过分块，让计算集中在SRAM上，而SRAM的速度远大于HBM，从而加速了计算过程。"

**方向正确，但表述不够精确**。

更准确的说法：

FlashAttention 的核心是**分块 tiling + 在线 softmax**：
1. 将 $Q,K,V$ 分成小块，每块能放进 SRAM
2. 在 SRAM 内完成 $QK^T$ → softmax → $AV$ 的局部计算
3. 用在线 softmax 算法（Milakov & Gimelshein, 2018）合并各块结果，无需存储完整 $n \times n$ 矩阵

**为什么快 3-4 倍**：
- 不是「用更快的硬件」——SRAM 和 HBM 都是同一块 GPU 上的存储
- 而是**减少了 HBM 读写次数**。标准 Attention 需要把 $n \times n$ 矩阵写入 HBM 再读出来做 softmax 和 $AV$，FlashAttention 全程在 SRAM 内完成，避免了这些昂贵的 HBM 往返
- 瓶颈从**计算**变成了**IO**——GPU 的计算能力（~312 TFLOPS FP16）远超 HBM 带宽（~2TB/s），减少 IO 就能显著提速

### Q3 评价

> 1. 数据增强（保守氨基酸替换 / 随机截断）
> 2. 加强正则化（Dropout 0.1→0.3 + Label Smoothing）

**方向对，但遗漏了最重要的一个改动**。

按照优先级排序：

**最高优先级：Early Stopping**
- Day 5 的训练曲线显示 Val Loss 从 epoch 10 就不再下降，但跑了 30 epoch
- 加 Early Stopping (patience=5)，在 epoch 15 左右就停，best checkpoint 对应的 Val Acc 可能更高
- 这个改动零成本、零风险，预期收益 1-2%

**高优先级：类别采样策略**
- 数据集极度不平衡：Nucleus 2424 vs Peroxisome 93（26 倍）
- 当前用 class_weights 调整 loss，但每个 batch 里少数类可能一条都没有
- 改用 `WeightedRandomSampler`，让每个 batch 中各类别出现概率相等
- 这比数据增强更直接——不需要发明新的增强策略，只需要让模型见到更多少数类

**中优先级：主人提到的两个**
- 数据增强：保守替换是好主意，但需要氨基酸替换矩阵（BLOSUM62），实现成本较高
- Dropout + Label Smoothing：Dropout 加大可能有效（当前 0.1 确实偏小），Label Smoothing 对不均衡数据效果不确定

**额外建议：增加模型容量**
- 当前 d_model=128, 3 层，600K 参数
- 试 d_model=256, 4 层（~2.4M 参数），容量增加 4 倍
- 从头训练的模型可能容量不足，增大模型 + 更强正则化（Dropout 0.2 + Early Stopping）可能是一个好的平衡点

---

## 五、Week 12 总结

### 三模型对比表

任务中的对比表写得很好，但有一个数字需要修正：

> "用 1/13 参数量、零预训练，达到 ESM-2 Frozen 的 96% 性能"

$61.0\% / 63.4\% = 96.2\%$，数字正确。但 Macro F1 的差距更大：$0.485 / 0.551 = 88.0\%$。说明自实现模型在少数类上的表现差距更明显（准确率被多数类拉上去了）。

### 知识图谱

Day 7 的知识图谱画得清晰，完整展现了 Week 12 的递进关系：

```
Self-Attention → Multi-Head → Positional Encoding → Encoder Block → 分类器 → 热图 → 复杂度
```

这条线从**组件**到**组装**到**应用**到**分析**到**理论**，是一个完整的「从零实现→理解→优化」闭环。

### Week 12 整体评价

| Day | 内容 | 评价 |
|-----|------|------|
| 1 | Multi-Head Self-Attention 从零实现 | ✅ 基础扎实 |
| 2 | MultiHeadAttention 类（含 attn_mask） | ✅ 模块化好 |
| 3 | Positional Encoding + Encoder Block (Pre-LN) | ✅ 结构正确 |
| 4 | 完整 TransformerEncoder 组装 | ✅ 600K 参数验证 |
| 5 | 蛋白质分类训练 | ✅ 61.0%，验证了预训练价值 |
| 6 | 注意力可视化 + 生物学解读 | ✅ 直观看到 head 分化差异 |
| 7 | 复杂度分析 + 收官 | ✅ 理论闭环 |

**Week 12 核心收获**：
1. **动手验证了 Transformer 的每一个组件**——不再是黑箱，每个矩阵乘法都亲手写过
2. **用实验证明了预训练的不可替代性**——600K 从头训练 < 42K 冻结预训练
3. **看到了注意力头的分化现象**——预训练让 head 有了「分工」，从头训练的 head 趋同
4. **理解了 O(n²) 瓶颈的来源**——为后续学习高效 Attention 打下基础

---

## 六、Week 12 → Week 13 衔接

Week 13 主题是**多标签分类**。从 Day 5 的单标签（10 类选 1）扩展到多标签（一条蛋白质可能同时属于多个亚细胞位置）。

关键技术差异：
- 损失函数：`CrossEntropyLoss` → `BCEWithLogitsLoss`（每个类别独立的 sigmoid）
- 评估指标：Accuracy → Hamming Loss、Subset Accuracy、per-label F1
- 阈值选择：不再是 argmax，需要为每个类别选一个阈值（默认 0.5 或调优）

Week 12 的 Transformer Encoder 可以直接复用，只需要换分类头和损失函数。
