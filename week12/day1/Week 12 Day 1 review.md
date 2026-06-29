# Week 12 Day 1 Review：Self-Attention 从零推导与实现

## 代码结构分析

```python
def scaled_dot_product_attention(Q, K, V, mask=None) -> (output, attn_weights)
```

四步流程，职责清晰：
1. **scores = Q @ Kᵀ / √d_k** — 计算原始注意力分数
2. **masked_fill** — 可选地屏蔽 padding 位置
3. **softmax(scores, dim=-1)** — 归一化为概率分布
4. **output = attn_weights @ V** — 加权求和

总代码量：~15 行有效代码。对比 Week 5 的同名函数，逻辑完全一致，但这次是**独立理解后写出来的**，不是填骨架。

## 数据流 / Shape 变化追踪

```
Q, K, V: (B, L, d_k) = (2, 10, 64)
    ↓
Q @ K.transpose(-2, -1)        → scores:      (2, 10, 10)   # 每个位置对每个位置的原始分数
    ↓ masked_fill (可选)
    ↓ softmax(dim=-1)
attn_weights:                   (2, 10, 10)   # 每行和为 1
    ↓ @ V
output:                         (2, 10, 64)   # 加权后的表示
```

**关键 shape 对比**：
- `attn_weights` 是 `(B, L, L)` 方阵 — 每个 token 对每个 token 的关注度
- Week 11 ESM-2 的注意力是 `(1, num_heads, L, L)` — 多了一个头维度
- 差异说明：本次是单头实现，Day 2 多头注意力会加上 `num_heads` 维度

## 关键知识点

### 1. 为什么需要三个独立的 W_Q / W_K / W_V？

Q、K、V 承担不同角色：
- **Q（查询）**：当前位置"想了解什么"
- **K（键）**：每个位置"能提供什么信息"
- **V（值）**：每个位置"实际携带的内容"

如果直接用 X·Xᵀ，查询和键共享同一个表示空间，模型无法学习"想问什么"和"能答什么"之间的非对称关系。三个独立投影矩阵让模型可以将同一输入映射到不同的语义空间。

### 2. 为什么要除以 √d_k？

当 d_k = 512 时，Q 和 K 的每个元素 ~ N(0,1)，点积 Q·Kᵀ 的方差约为 d_k = 512，标准差约 22.6。

这意味着 softmax 的输入值可能在 [-45, +45] 范围，softmax 会极度饱和（接近 one-hot），梯度几乎为零。

除以 √d_k = √512 ≈ 22.6 后，方差压回 1，softmax 在健康的非饱和区工作。

### 3. attn_weights[b, i, j] = 0.8 的含义

位置 i 的氨基酸"关注"位置 j 的氨基酸，权重为 0.8（80% 的信息来自位置 j）。

蛋白质语境下的直觉：如果 position 3 对 position 7 的注意力权重为 0.8，可能意味着这两个氨基酸在三维空间中相互靠近（结构接触），或在序列上有功能耦合（如二硫键、氢键、共进化信号）。

## 踩坑与易错点

### 1. np.sqrt vs torch.sqrt（潜在坑）

```python
scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # 用的 numpy
```

这里用 `np.sqrt(d_k)` 没有问题，因为 d_k 是 Python int，结果是 float，PyTorch 会自动处理标量除法。

但如果用 `torch.sqrt(d_k)` 需要注意 `d_k` 必须是 float tensor，否则会得到整数截断。更安全的写法：

```python
d_k = Q.shape[-1]
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # 纯 Python，最安全
```

Week 5 的版本用的就是 `d_k ** 0.5`，两种写法等价，但 `** 0.5` 更 PyTorch 惯用。

### 2. mask 的 shape 约定

Task 中 mask 定义为 `(B, L, L)`，mask=True 的位置填 -inf。这和 Week 5 的 mask 约定不同：

| 版本 | mask shape | mask 含义 |
|------|-----------|----------|
| Week 5 Day 2 | `(B, L)` | True = padding 位置，unsqueeze 到 (B, 1, L) |
| Week 12 Day 1 | `(B, L, L)` | True = 需要屏蔽的位置，直接用 |

Day 1 的 `(B, L, L)` 更通用，可以表达任意的注意力屏蔽模式（如因果 mask、局部窗口 mask）。Day 4 实现 Encoder Block 时需要注意统一。

### 3. 输出 shape 注释的笔误

Task 验证部分的注释写的是：
```
print(f"attn_weights shape: {attn_weights.shape}") # 期望: (2, 10, 64)
```

但 `attn_weights` 的 shape 应该是 `(2, 10, 10)`，不是 `(2, 10, 64)`。这是 task 文件的笔误，代码输出正确。

## 输出问题回答评估

**Q1**：✅ 回答正确。核心点抓住了——Q/K/V 角色不同，独立投影让模型可以学习非对称的查询-键关系。

**Q2**：✅ 回答正确。准确指出了 d_k=512 时点积量级约 512，softmax 接近 one-hot，梯度消失。

**Q3**：✅ 回答正确。"位置 3 的氨基酸对位置 7 的氨基酸影响较大"——可以更精确地说"位置 3 从位置 7 获取了 80% 的信息"，但本质理解到位。

## 与 Week 5 对比

| 维度 | Week 5 Day 2 | Week 12 Day 1 |
|------|-------------|--------------|
| 实现方式 | 骨架 + 提示填充 | 独立完成 |
| mask 约定 | `(B, L)` padding mask | `(B, L, L)` 通用 mask |
| √d_k 写法 | `d_k ** 0.5` | `np.sqrt(d_k)` |
| 验证方式 | smoke_test 函数 | 直接运行 + 打印 |
| 理解深度 | 跑通即可 | 能解释每个设计决策的原因 |

**结论**：Week 12 Day 1 达成目标。从"填骨架"升级到"理解后独立实现"，三个输出问题全部答对。下一步 Day 2 的多头注意力会在此基础上加入 `num_heads` 维度。
