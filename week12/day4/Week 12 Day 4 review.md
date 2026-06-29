# Week 12 Day 4 Review：完整 Transformer Encoder Block

## 代码结构分析

```
FeedForward(d_model, d_ff, dropout)
    └── nn.Sequential: Linear → ReLU → Dropout → Linear → Dropout

TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
    ├── attn: MultiHeadAttention (Day 2)
    ├── ffn: FeedForward
    ├── norm1, norm2: LayerNorm
    └── forward: Pre-LN × 2 子层

TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
    ├── embedding: nn.Embedding
    ├── pos_enc: SinusoidalPositionalEncoding (Day 3)
    ├── layers: ModuleList[TransformerEncoderBlock] × N
    └── norm: 最终 LayerNorm
```

三层嵌套结构，每层职责清晰。Day 1~3 的模块全部复用，没有重复代码。

## 数据流 / Shape 变化追踪

```
TransformerEncoder.forward:
    input_ids:           (B, L)           = (2, 50)
    ↓ embedding
    x:                   (2, 50, 64)
    ↓ pos_enc
    x:                   (2, 50, 64)      # 加上位置编码

    ↓ 遍历 3 层 TransformerEncoderBlock:

      Block.forward:
        x:               (2, 50, 64)
        ↓ norm1
        x_norm:          (2, 50, 64)      # Pre-LN
        ↓ MultiHeadAttention
        attn_out:        (2, 50, 64)
        attn_weights:    (2, 4, 50, 50)   # 4 个头
        ↓ dropout + 残差
        x:               (2, 50, 64)      # 残差连接

        ↓ norm2
        x_norm:          (2, 50, 64)
        ↓ FeedForward
        ffn_out:         (2, 50, 64)      # 64→256→64
        ↓ dropout + 残差
        x:               (2, 50, 64)

    ↓ 最终 LayerNorm
    x:                   (2, 50, 64)      # 最终隐状态
    all_attn:            [(2,4,50,50)] × 3
```

**输入输出 shape 一致**：`(B, L, d_model)` 进，`(B, L, d_model)` 出。这是 Encoder 的基本契约。

## 关键知识点

### 1. Pre-LN vs Post-LN

```
Post-LN（原始论文）:  x = LayerNorm(x + SubLayer(x))
Pre-LN（ESM-2 用的）: x = x + SubLayer(LayerNorm(x))
```

**梯度流动差异**：
- Post-LN：梯度必须穿过 LayerNorm 才能到达残差连接。深层网络中，LayerNorm 的归一化操作会压缩梯度量级，导致深层梯度衰减。
- Pre-LN：残差连接是"直通"的，梯度可以不经过 LayerNorm 直接回传。LayerNorm 只作用于子层的输入，不影响残差路径的梯度。

这就是为什么深层 Transformer（如 ESM-2 33 层）几乎都用 Pre-LN。

### 2. FFN 的逐位置独立性

FFN 的参数是 `(d_model, d_ff)` 和 `(d_ff, d_model)`，对每个位置**独立**应用。位置 0 和位置 49 用完全相同的 FFN 参数，互不影响。

这和 Attention 形成互补：
- **Attention**：跨位置信息聚合（位置 i 关注位置 j）
- **FFN**：逐位置特征变换（每个位置独立"思考"）

### 3. ModuleList vs 普通 list

```python
# ✅ 正确：ModuleList 注册子模块
self.layers = nn.ModuleList([Block(...) for _ in range(N)])

# ❌ 错误：普通 list，parameters() 找不到子模块
self.layers = [Block(...) for _ in range(N)]
```

`nn.ModuleList` 会把子模块注册到父模块的 `_modules` 字典中。`model.parameters()` 递归遍历 `_modules`，普通 list 不在遍历范围内，子模块的参数会被"遗忘"——看起来模型在训练，但实际上只有 embedding 和 norm 在更新，layers 的梯度全为零。

### 4. 参数量验证

```
单层 Block: 49,984
  ├── MHA:  W_q + W_k + W_v + W_o = 4 × (64×64 + 64) = 16,640
  ├── FFN:  (64×256 + 256) + (256×64 + 64) = 33,088
  ├── LN1:  64 + 64 = 128
  └── LN2:  64 + 64 = 128

总计:
  Embedding:  23 × 64 = 1,472
  3 × Block:  49,984 × 3 = 149,952
  Final LN:   128
  ───────────────────────
  Total:      151,552 ✅
```

### 5. 与 Week 5 / Week 10 的架构对比

| 维度 | Week 5 Day 5 | Week 10 ProteinBERT | Week 12 Day 4 |
|------|-------------|--------------------|----|
| MHA | 手写 | `nn.TransformerEncoderLayer` | 手写（Day 2 复用） |
| 位置编码 | 正弦（手写） | 可学习 `nn.Embedding` | 正弦（Day 3 复用） |
| LN 位置 | Pre-LN | PyTorch 默认 Post-LN | Pre-LN |
| FFN 激活 | ReLU | GELU（PyTorch 默认） | ReLU |
| d_model | 64 | 128 | 64 |
| 参数量 | ~50K | 469,657 | 151,552 |
| 复用方式 | 全部自包含 | PyTorch 内置 | Day 1-3 模块复用 |

Week 12 的实现方式**最模块化**——每个组件独立文件、独立测试、通过 import 复用。这是工程上的进步。

## 踩坑与易错点

### 1. FFN 中 Dropout 的位置

```python
self.net = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Dropout(dropout),    # ← ReLU 之后、第二个 Linear 之前
    nn.Linear(d_ff, d_model),
    nn.Dropout(dropout),    # ← 输出之后
)
```

两个 Dropout 的作用不同：
- 第一个：正则化隐藏层，防止 FFN 过拟合
- 第二个：正则化输出，与残差连接前的 dropout 统一

有些实现只在 ReLU 后放一个 Dropout，效果差异不大。

### 2. Pre-LN 的最终 LayerNorm

```python
self.norm = nn.LayerNorm(d_model)  # TransformerEncoder 最后的 LN
```

Pre-LN 架构中，每个 Block 内部的 LN 在子层**之前**。但最后一层 Block 输出后没有 LN，所以需要在 Encoder 最后加一个额外的 LayerNorm。这是 Pre-LN 的惯例。

### 3. mask 透传

当前实现中 mask 从 `TransformerEncoder.forward` 透传到每个 `TransformerEncoderBlock`，再透传到 `MultiHeadAttention`。这是正确的，但要注意 mask 的 shape 必须与 `MultiHeadAttention` 期望的 `(B, 1, 1, L)` 一致。Day 5 在蛋白质分类任务中使用时需要处理好 padding mask 的 shape。

## 输出问题回答评估

**Q1**：✅ 正确。ModuleList 注册子模块到 `_modules`，普通 list 不注册，`parameters()` 遗漏子模块参数。

**Q2**：✅ 基本正确——经验值，来自原始论文。可以补充：4 倍扩展因子的直觉是"先升维增加表达能力，再降维回去"，类似 bottleneck 但方向相反（先扩后缩）。

**Q3**：✅ 正确。Pre-LN 的残差路径是"干净"的，梯度可以不经过 LayerNorm 直接回传；Post-LN 的梯度必须穿过 LayerNorm，深层时容易衰减。

## 与 Week 5 对比

| 维度 | Week 5 Day 5 | Week 12 Day 4 |
|------|-------------|--------------|
| 实现方式 | 填骨架 | 独立完成 |
| 模块化 | 全部自包含 | Day 1-3 import 复用 |
| Pre-LN 理解 | 有实现但未必理解 | 能解释梯度流动差异 |
| FFN 激活 | ReLU | ReLU |
| 参数验证 | 无 | ✅ 逐模块分解验证 |

**Day 4 达成目标**。前三天的零件全部组装完成，一个完整的 Transformer Encoder 可以工作了。Day 5 将用它替换 ESM-2 做蛋白质分类——这是 Week 12 的核心实验。
