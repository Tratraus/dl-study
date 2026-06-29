# Week 12 Day 2 Review：多头注意力实现

## 代码结构分析

```python
class MultiHeadAttention(nn.Module):
    __init__(d_model, num_heads)     # 4 个 Linear + 参数
    split_heads(x)                   # (B, L, d_model) → (B, H, L, d_k)
    forward(x, mask) → (output, attn_weights)
```

5 步 forward 流程：
1. **线性投影** — W_q/W_k/W_v 生成 Q/K/V
2. **拆分多头** — split_heads 把 d_model 切成 H 份
3. **合并 batch 维度** — (B, H, L, d_k) → (B*H, L, d_k)，复用 Day 1 的 scaled_dot_product_attention
4. **拼接多头** — reshape 回 (B, L, d_model)
5. **输出投影** — W_o 线性变换

核心设计：**通过 reshape 复用 Day 1 的单头注意力函数**，避免重复代码。

## 数据流 / Shape 变化追踪

```
输入 x:                        (B, L, d_model) = (2, 10, 64)
    ↓ W_q / W_k / W_v
Q, K, V:                       (2, 10, 64)        # 线性投影，shape 不变
    ↓ split_heads: view(2, 10, 4, 16) → transpose(1,2)
Q, K, V:                       (2, 4, 10, 16)     # num_heads=4, d_k=16
    ↓ reshape(2*4, 10, 16)
Q, K, V:                       (8, 10, 16)        # 合并 batch 和 heads
    ↓ scaled_dot_product_attention (Day 1 复用)
attn_out:                      (8, 10, 16)
attn_weights:                  (8, 10, 10)
    ↓ reshape(2, 4, 10, 16) → transpose(1,2) → reshape(2, 10, 64)
attn_out:                      (2, 10, 64)        # 拼接回 d_model
    ↓ W_o
output:                        (2, 10, 64)        # 最终输出，与输入 shape 一致
attn_weights:                  (2, 4, 10, 10)     # 每个头独立的注意力权重
```

**关键 shape 变化**：
- `split_heads` 中 view + transpose 的组合是核心技巧
- `B*H` 合并是复用单头函数的关键——把多头当成多个独立 batch

## 关键知识点

### 1. 多头 vs 单头的参数量

| 组件 | 单头 | 多头 (H=4) |
|------|------|-----------|
| W_q | d_model × d_model | d_model × d_model |
| W_k | d_model × d_model | d_model × d_model |
| W_v | d_model × d_model | d_model × d_model |
| W_o | — | d_model × d_model |
| **总计** | 3 × d² | 4 × d² |

多头比单头**多了 W_o**，但核心的 Q/K/V 投影参数量相同。多头的"多"体现在**子空间划分**，不是参数复制。

### 2. 为什么要 view + transpose，不能直接 view

`view` 按行优先（C-order）重新解释内存布局。假设 d_model=4, num_heads=2, d_k=2：

```
原始: [a0, a1, a2, a3]  ← 位置 0 的 4 维特征

view(B, L, H, d_k):  [a0, a1 | a2, a3]   ✓ 每个头拿到连续的 d_k 维
view(B, H, L, d_k):  [a0, a1 | a2, a3]   ✗ 特征被跨位置混在一起
```

直接 `view(B, H, L, d_k)` 会让 head_0 拿到位置 0 的前半和位置 1 的前半，特征来自不同位置，语义混乱。

**先 view(B, L, H, d_k) 保证每个位置的特征被正确切分，再 transpose(1,2) 把 head 维度提前。**

### 3. B*H 合并的技巧

```python
Q = Q.reshape(B * self.num_heads, L, self.d_k)
```

把 `(B, H, L, d_k)` reshape 成 `(B*H, L, d_k)`，这样 H 个头变成了 B*H 个独立的"batch"，可以直接传给 Day 1 的 `scaled_dot_product_attention`。

这是 PyTorch 中复用单头函数的标准技巧：**头 = 虚拟 batch 维度**。

## 踩坑与易错点

### 1. mask 的广播约定

Day 1 的 mask 是 `(B, L, L)`，但 Day 2 的 task 要求 mask 是 `(B, 1, 1, L)`。

当 mask 传给 `scaled_dot_product_attention` 时，需要先在 split_heads 之前处理好，或者在 reshape 成 `B*H` 之前做 `expand`。当前代码直接透传，依赖 Day 1 函数内部处理——这在 mask=None 时没问题，但有 mask 时需要额外注意 shape 兼容。

### 2. split_heads 的 contiguous 问题

```python
attn_out = attn_out.transpose(1, 2)          # (B, L, H, d_k) — 不连续
attn_out = attn_out.reshape(B, L, self.d_model)  # 需要 contiguous
```

`transpose` 不改变内存布局，只改变 stride。后续 `view/reshape` 需要连续内存。PyTorch 的 `reshape` 会自动处理（内部 copy if needed），但如果用 `view` 则必须先 `.contiguous()`。当前代码用 `reshape` 是安全的。

### 3. W_o 的作用

W_o 常被忽略，但它很重要：**混合来自不同头的信息**。没有 W_o，每个头的输出是独立的，拼接后各头信息不交互。W_o 让模型学习如何组合不同头的输出。

## 输出问题回答评估

**Q1**：✅ 核心正确——直接 view(B, H, L, d_k) 会破坏特征的空间结构。可以更精确地说"行优先重排会导致不同位置的特征混在一起"，但理解到位。

**Q2**：✅ 回答到位——子空间划分让每个头学习不同类型的注意力模式，参数量不变但表达能力更强。

**Q3**：✅ 正确——shape 一致，都是 (B, num_heads, L, L)。

## 与 Week 5 对比

| 维度 | Week 5 Day 3 | Week 12 Day 2 |
|------|-------------|--------------|
| 实现方式 | 填骨架（有提示） | 独立完成 |
| 拆分方式 | view + transpose | view + transpose（一致） |
| mask 处理 | unsqueeze(1).unsqueeze(2) | 透传给 Day 1 函数 |
| W_o | bias=False | bias=True（默认） |
| 参数验证 | 无 | ✅ 验证了 16640 = 4×64×64 + 4×64 |

**Day 2 达成目标**。多头注意力的核心技巧（view+transpose 拆分、B*H 合并复用单头函数、W_o 混合头信息）全部理解和实现到位。
