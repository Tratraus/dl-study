# Week 12 · Day 7：收官 + CLRS 复杂度分析

今天不写新代码，做两件事：**推导 + 总结**。

---

## Part 1：Self-Attention 复杂度推导

### 标准 Self-Attention 的计算步骤

给定输入序列长度 $n$，模型维度 $d$，单头：

**Step 1：投影 Q、K、V**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

- $X \in \mathbb{R}^{n \times d}$，$W \in \mathbb{R}^{d \times d_k}$
- 每次矩阵乘法：$O(n \cdot d \cdot d_k)$，三次共 $O(nd^2)$（当 $d_k = d/h$ 时）

**Step 2：计算注意力分数**

$$S = \frac{QK^T}{\sqrt{d_k}}, \quad Q \in \mathbb{R}^{n \times d_k},\ K \in \mathbb{R}^{n \times d_k}$$

- 矩阵乘法 $QK^T$：$O(n^2 d_k)$
- **这是瓶颈**：序列长度 $n$ 出现了平方项

**Step 3：Softmax**

$$A = \text{softmax}(S), \quad S \in \mathbb{R}^{n \times n}$$

- 逐行 softmax：$O(n^2)$

**Step 4：加权求和**

$$\text{out} = AV, \quad A \in \mathbb{R}^{n \times n},\ V \in \mathbb{R}^{n \times d_k}$$

- 矩阵乘法：$O(n^2 d_k)$

---

### 汇总

| 步骤 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Q/K/V 投影 | $O(nd^2)$ | $O(nd)$ |
| $QK^T$ | $O(n^2 d_k)$ | $O(n^2)$ ← 瓶颈 |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $AV$| $O(n^2 d_k)$ | $O(nd)$ |
| **合计** | $O(n^2 d + nd^2)$ | $O(n^2 + nd)$ |

当 $n \ll d$ 时（短序列、大模型），$nd^2$ 主导；
当 $n \gg d$ 时（长序列），$n^2 d$ 主导。

蛋白质场景通常 $n \sim 500$，$d \sim 320$（ESM-2），两项量级相近，**但随序列变长，$n^2 d$ 项增速更快**。

---

### 关键数字题

> 序列长度从 $n=1024$ 增加到 $n=4096$（增加 4 倍），$QK^T$ 的计算量变化？

$$\frac{(4096)^2}{(1024)^2} = \frac{4^2 \cdot 1024^2}{1024^2} = 16$$

**计算量增加 16 倍，显存占用也增加 16 倍**（注意力矩阵从 1M → 16M 个元素）。

这就是为什么处理全基因组序列（$n \sim 10^5$）时，标准 Self-Attention 完全不可行。

---

## Part 2：线性 Attention 的动机

### 核心思路

标准 Attention：先算 $n \times n$ 的注意力矩阵，再乘 $V$

$$\text{out}_i = \frac{\sum_j \exp(q_i \cdot k_j / \sqrt{d}) v_j}{\sum_j \exp(q_i \cdot k_j / \sqrt{d})}$$

**线性 Attention**：用核函数 $\phi$ 近似 $\exp$，利用矩阵乘法结合律**绕过** $n \times n$ 矩阵

$$\text{out}_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)}{\phi(q_i)^T \left(\sum_j \phi(k_j)\right)}$$

先算括号内的 $\sum_j \phi(k_j) v_j^T$（$O(nd^2)$），再乘 $\phi(q_i)$（$O(nd)$），**完全避免了 $O(n^2)$ 的矩阵**。

| 方法 | 时间复杂度 | 空间复杂度 | 代价 |
|------|-----------|-----------|------|
| 标准 Attention | $O(n^2 d)$ | $O(n^2)$ | 精确 |
| Linformer | $O(nkd)$ | $O(nk)$ | 低秩近似，精度损失 |
| Performer | $O(nd^2)$ | $O(nd)$ | 随机特征近似，无偏估计 |
| FlashAttention | $O(n^2 d)$ | $O(n)$ | IO 优化，不降低理论复杂度但实际快 3-4× |

> **FlashAttention 的巧妙之处**：时间复杂度没变，但通过分块计算避免了将完整 $n \times n$ 矩阵写入 HBM（高带宽显存），显存从 $O(n^2)$ 降到 $O(n)$ ——瓶颈从计算变成了 IO。

---

## Part 3：Week 12 完整收官

### 三模型对比表（最终版）

| 维度 | 自实现 Encoder | ESM-2 Frozen | ESM-2 Fine-tuned |
|------|--------------|-------------|-----------------|
| **参数量** | 600K | 8M | 8M |
| **预训练数据** | ❌ 无 | UR50D（2500亿AA） | UR50D |
| **测试准确率** | 61.0% | 63.4% | 67.7% |
| **Macro F1** | 0.485 | — | — |
| **注意力模式** | 稀疏列型（锚点汇聚） | 对角线带（局部语法）+ 功能分工 | 同左 |
| **过拟合程度** | 明显（Val Loss 早平台） | 无（Frozen） | 轻微 |
| **训练成本** | 低（30 epoch，分钟级） | 极高（预训练） | 中（Fine-tune） |

**核心结论**：用 1/13 参数量、零预训练，达到 ESM-2 Frozen 的 96% 性能。差距主要来自预训练知识（进化约束、结构偏好），而非模型结构本身。

---

### Week 12 知识图谱

```
Self-Attention（Day 1）
    ↓ 并行化
Multi-Head Attention（Day 2）
    ↓ 加位置信息
Positional Encoding（Day 3）
    ↓ 加 FFN + 残差 + LN
Transformer Encoder Block（Day 4）
    ↓ 堆叠 + 分类头
蛋白质分类器（Day 5）→ 61.0% Test Acc
    ↓ 打开黑盒
注意力热图（Day 6）→ 锚点 vs 局部语法
    ↓ 理论上界
复杂度分析（Day 7）→ O(n²d) 瓶颈 → 线性 Attention
```

---

## 输出问题（收官三问）

**Q1**：用自己的话，解释为什么 ESM-2 的注意力模式是"对角线"，而自实现模型是"竖条纹"——这个差异的根本原因是什么？

ESM2作为预训练模型，其分类头分工更为优秀，而自实现模型的各头注意力趋同。

**Q2**：FlashAttention 的显存从 $O(n^2)$ 降到 $O(n)$，但时间复杂度没变，为什么实际速度还是快了 3-4 倍？

因为FlashAttention通过分块，让计算集中在SRAM上，而SRAM的速度远大于HBM，从而加速了
计算过程。也就是用更快的硬件进行计算。

**Q3**：如果让你把自实现模型的准确率从 61% 提升到 65%，**不使用预训练**，你会优先尝试哪两个改动？说明理由。

1. 数据增强，增加数据多样性
2. 修改模型结构

Answer：
1. 改动 1：数据增强（最高性价比）
   * 蛋白质序列可以做保守氨基酸替换（如 Val→Ile，同族氨基酸互换），不改变功能但增加训练样本多样性。或者对序列做随机截断（Random Cropping），让模型见到更多局部片段。

2. 改动 2：加强正则化
   * 当前模型已经过拟合，应该加大 Dropout（0.1 → 0.3）+ 加入 Label Smoothing（将 one-hot 标签软化为 0.9/0.1），让模型不要对训练集过度自信。这两个改动通常能让 Val Acc 提升 2-4 个百分点，正好对应 61% → 65% 的目标。