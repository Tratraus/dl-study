# Week 9 Day 2 Review — Cross-Attention 原理与实现

## 代码结构分析

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        # 4 个 Linear 投影（全部 bias=False）
        self.q_proj  ← query 投影
        self.k_proj  ← context 投影 (Key)
        self.v_proj  ← context 投影 (Value)
        self.out_proj ← 输出投影
        self.scale = sqrt(d_model)

    def forward(self, query, context):
        Q = q_proj(query)           # 来自 Decoder
        K = k_proj(context)         # 来自 Encoder memory
        V = v_proj(context)         # 来自 Encoder memory
        scores = Q @ K^T / scale    # 点积 + 缩放
        weights = softmax(scores)   # 注意力权重
        output = weights @ V        # 加权求和
        output = out_proj(output)   # 输出投影
```

结构清晰，4 步流水线完整。

---

## 数据流 / Shape 变化追踪

| 阶段 | 操作 | Shape |
|------|------|-------|
| 输入 query | — | (2, 6, 64) |
| 输入 context | — | (2, 10, 64) |
| Q = q_proj(query) | Linear(64→64) | (2, 6, 64) |
| K = k_proj(context) | Linear(64→64) | (2, 10, 64) |
| V = v_proj(context) | Linear(64→64) | (2, 10, 64) |
| Q @ K^T | matmul | (2, 6, 10) |
| / scale | 除以 8.0 | (2, 6, 10) |
| softmax(dim=-1) | 沿 src_len 归一化 | (2, 6, 10) |
| weights @ V | 加权求和 | (2, 6, 64) |
| out_proj | Linear(64→64) | (2, 6, 64) ✅ |

**关键**：输出 shape 始终由 `tgt_len=6` 决定，`src_len=10` 只出现在注意力权重矩阵的宽度方向，最终被"求和掉"了。

---

## 关键知识点

### 1. Cross-Attention vs Self-Attention 的唯一区别：Q/K/V 的来源

- **Self-Attention**：Q、K、V 全来自同一个输入（"自己问自己"）
- **Cross-Attention**：Q 来自 Decoder，K/V 来自 Encoder（"拿着问题去查资料"）

计算过程完全相同，区别只在于 `query` 和 `context` 是不是同一个东西。

### 2. `K.transpose(-2, -1)` vs `K.T`

这是本代码的一个易错点，主人做对了：

```python
# ❌ K.T 会反转所有维度 → (d_model, src_len, batch) — 完全错误
# ✅ K.transpose(-2, -1) 只交换最后两维 → (batch, d_model, src_len) — 正确
```

3D tensor 不能用 `.T`，必须指定转置哪两个维度。

### 3. Softmax 沿 `dim=-1` 的含义

```python
weights = F.softmax(scores, dim=-1)  # 沿 src_len 方向归一化
```

对于每个 query 位置（tgt_len 中的每一行），注意力权重在所有 src_len 个 key 上求和为 1。这意味着：**每个 query 位置独立地决定"关注哪些 encoder 位置"**。

### 4. `bias=False` 的选择

全部投影层都没用 bias。这是合理的——在 Transformer 中，投影后通常跟着 LayerNorm，bias 是冗余的。主流实现（如 PyTorch 的 `nn.MultiheadAttention`）默认也是 `bias=True`，但 `bias=False` 完全合法。

---

## 踩坑易错点

### 易错 1：转置写错

```python
# ❌ 常见错误
scores = Q @ K.T           # .T 对 3D tensor 会出错
scores = Q @ K.permute(0, 2, 1)  # 对，但不如 transpose 简洁
# ✅ 正确
scores = Q @ K.transpose(-2, -1)
```

### 易错 2：softmax 的 dim 参数

```python
# ❌ 如果写 dim=0 或 dim=1，归一化方向就错了
F.softmax(scores, dim=0)   # 沿 batch 方向归一化 — 错！
F.softmax(scores, dim=1)   # 沿 tgt_len 方向归一化 — 错！
# ✅ 必须沿 src_len（最后一维）归一化
F.softmax(scores, dim=-1)
```

### 易错 3：缩放位置

```python
# ❌ 缩放放在 softmax 之后
weights = F.softmax(scores, dim=-1)
output = weights @ V / self.scale  # 错！缩放应在 softmax 之前

# ✅ 缩放放在 softmax 之前
scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
weights = F.softmax(scores, dim=-1)
```

主人做对了——缩放在 matmul 之后、softmax 之前。

---

## 输出问题回答评价

### Q1：K^T 在 PyTorch 里怎么写？

**主人回答**：`K.transpose(-2, -1)` 或 `K.transpose(1, 2)`，都可以。

**评价**：✅ 正确。补充一点：`K.transpose(1, 2)` 和 `K.transpose(-2, -1)` 在 3D tensor 中完全等价（因为 -2=1, -1=2），但写 `-2, -1` 更通用——如果以后变成 4D（比如多头注意力），`-2, -1` 依然正确，而 `1, 2` 就错了。建议养成用负索引的习惯。

### Q2：为什么要除以 sqrt(d_model)？

**主人回答**：防止 scores 数值过大导致 softmax 后梯度消失。

**评价**：✅ 方向正确，但可以更精确。标准表述是：

> 当 d_model 较大时，Q 和 K 的点积的方差约为 d_model（假设每个分量均值 0、方差 1）。点积值越大，softmax 的输出越接近 one-hot（趋近 0 或 1），梯度越小。除以 √d_k 将方差稳定在 1 附近，让 softmax 保持在梯度敏感的区域。

简单说：**大 d_model → 大点积 → softmax 饱和 → 梯度消失**。缩放让点积的方差与 d_model 无关。

### Q3：输出形状为什么和 src_len 无关？

**主人回答**：因为加权求和的结果维度是 d_model，与 src_len 无关。

**评价**：✅ 完全正确。从矩阵乘法的角度：`(tgt_len, src_len) @ (src_len, d_model) = (tgt_len, d_model)`，src_len 是被"求和掉"的内维。这就是注意力机制的核心——**不管有多少个 source token，每个 query 位置的输出都是一个 d_model 维的向量**。

---

## 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| TODO 完成度 | ✅ 6/6 + 验证 | 全部完成 |
| Shape 推理 | ✅ | 所有 shape 正确 |
| 转置写法 | ✅ | 用了 `transpose(-2, -1)`，正确 |
| 缩放位置 | ✅ | 在 softmax 之前，正确 |
| 理解深度 | ✅ | Q1-Q3 回答准确，Q2 可更精确 |

**一句话总结**：Cross-Attention 的代码和理解都到位了。和 Day 1 的 Self-Attention 相比，唯一变化就是 K/V 的来源从"自己"变成了"context"——计算过程完全相同。这就是为什么 Transformer 里 Self-Attention 和 Cross-Attention 可以共用同一套实现。
