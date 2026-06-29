# Week 12 Review：注意力机制深入理解（从零实现 Transformer 全组件）

## 本周全景

Week 12 的目标是**从零实现 Transformer Encoder 的每一个组件**，然后用它做蛋白质分类，和 ESM-2 对比。

```
Day 1: Multi-Head Self-Attention（Q/K/V 投影、缩放点积、多头拆分）
Day 2: MultiHeadAttention 类（attn_mask、dropout、输出投影）
Day 3: SinusoidalPositionalEncoding + TransformerEncoderBlock（Pre-LN）
Day 4: 完整 TransformerEncoder（Embedding + PosEnc + N×Block + FinalLN）
Day 5: 蛋白质亚细胞定位分类（600K 参数，Test Acc 61.0%）
Day 6: 注意力可视化（自实现 vs ESM-2 热图对比）
Day 7: 复杂度推导 + O(n²) 瓶颈 + 线性 Attention 动机
```

## 核心实验结果

| 模型 | 参数量 | 预训练 | Test Acc | Macro F1 |
|------|--------|--------|----------|----------|
| 自实现 Encoder | 600K | ❌ | 61.0% | 0.485 |
| ESM-2 Frozen | 8M | ✅ (42K 可训练) | 63.4% | 0.551 |
| ESM-2 Fine-tuned | 8M | ✅ (2.5M 可训练) | 67.7% | 0.581 |

**核心结论**：600K 参数从头训练 < 42K 参数冻结预训练。预训练的「蛋白质先验知识」比模型大小更重要。

## 关键技术收获

### 1. Transformer 组件的逐层理解
- **Self-Attention**：Q/K/V 三个视角，缩放点积防止梯度消失
- **Multi-Head**：不同 head 关注不同特征（局部 vs 全局、不同位置）
- **Positional Encoding**：正弦编码的几何直觉（旋转矩阵、相对位置可表示为线性变换）
- **Pre-LN vs Post-LN**：Pre-LN 在子层前归一化，梯度流更稳定
- **残差连接**：恒等映射 + 微调，梯度直通

### 2. 注意力头的分化现象
- 从头训练：各头模式趋同（都学到了「看邻居」），Layer 1 强对角线 → Layer 3 均匀分散
- 预训练（ESM-2）：各头有明确分工——列型（锚点汇聚）、行型（信息广播）、块型（功能域内部）、对角线（局部上下文）
- 这种分化是**预训练的进化压力**（MLM 任务迫使不同 head 捕捉不同类型的共现模式）

### 3. O(n²) 复杂度的本质
- $QK^T$ 是 $n \times d_k$ 乘 $d_k \times n$，结果是 $n \times n$ 的注意力矩阵
- $n$ 增加 4 倍 → 计算量增加 16 倍，显存也增加 16 倍
- 这就是为什么处理全基因组（$n \sim 10^5$）时标准 Attention 不可行
- 解决方案：线性 Attention（$O(nd^2)$）、FlashAttention（IO 优化）、稀疏 Attention

## 工程经验

### 代码复用的价值
Day 5 没有照搬 task 模板的数据管线，而是复用 Week 11 的 `protein_dataset.py` + `esm2_embed.py`。这是正确的工程判断——核心目标是「自实现 Encoder 做分类」，不是「重写一遍 tokenize」。

### Mask 语义的双重处理
- **Attention 内部**：`attn_mask = ~mask.bool()`（True=屏蔽）
- **Pooling**：`mask`（1=有效）
- 两套语义相反，容易搞混，但代码里处理得清晰

### 过拟合的信号
- Day 5 的 Val Loss 从 epoch 10 就不再下降（1.49 → 1.60），但跑了 30 epoch
- 建议加 Early Stopping（patience=5-7）
- 从头训练的模型更容易过拟合（没有预训练的正则化效果）

## 技能树更新

```
Week 12 新增技能：
├── Transformer 从零实现
│   ├── Multi-Head Self-Attention（含 attn_mask）
│   ├── Sinusoidal Positional Encoding
│   ├── Pre-LN Encoder Block
│   └── 完整 TransformerEncoder 组装
├── 注意力可视化
│   ├── 提取注意力权重（自实现 + ESM-2）
│   ├── 层×头 网格热图
│   └── 全局平均注意力对比
├── 复杂度分析
│   ├── O(n²d + nd²) 推导
│   ├── 线性 Attention 原理（结合律绕过 n×n）
│   └── FlashAttention IO 优化原理
└── 工程实践
    ├── 模块复用（Week 11 数据管线）
    ├── 类别权重处理不均衡
    └── Mask 语义翻转处理
```

## 下周展望

Week 13：多标签分类。核心技术变化：
- `CrossEntropyLoss` → `BCEWithLogitsLoss`
- 单标签评估 → 多标签评估（Hamming Loss、per-label F1）
- 亚细胞定位本身就是多标签任务（一条蛋白质可能同时存在于多个位置）
- Week 12 的 TransformerEncoder 可以直接复用
