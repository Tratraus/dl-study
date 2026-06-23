# Week 10 Day 4 Review — 序列嵌入提取

## 完成状态

| 项目 | 状态 | 备注 |
|---|---|---|
| Step 1: `encode` 方法 | ✅ | protein_bert.py 已新增 |
| TODO 1: `cls_pooling` | ✅ | 4 行，拼 BOS → encode → 取位置 0 |
| TODO 2: `mean_pooling` | ✅ | 4 行，构造 mask → encode → 加权平均 |
| TODO 3: `verify_embeddings` | ✅ | 加载 checkpoint + 两种池化对比 + 一致性验证 |
| Q1-Q3 | ✅ | 全部回答 |

---

## 代码结构分析

### Step 1：ProteinBERT.encode（改 day2/protein_bert.py）

```python
def encode(self, src, src_key_padding_mask=None):
    x = self.embedding(src)
    x = self.pe(x)
    x = self.dropout(x)
    for layer in self.layers:
        x = layer(x, src_key_padding_mask=src_key_padding_mask)
    x = self.norm(x)
    return x   # 不走 mlm_head
```

**与 forward 的区别：**
```
forward:  ... → LayerNorm → MLM Head → logits   (batch, seq_len, vocab_size)
encode:   ... → LayerNorm → return              (batch, seq_len, d_model)
```

只差最后两行——`encode` 在 `self.norm(x)` 后直接返回，`forward` 多过一个 `self.mlm_head`。

---

### TODO 1：cls_pooling

```python
bos_col = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
src_with_bos = torch.cat([bos_col, src], dim=1)
src_key_padding_mask = (src_with_bos == PAD)
hidden = model.encode(src_with_bos, src_key_padding_mask=src_key_padding_mask)
cls_embedding = hidden[:, 0, :]
return cls_embedding
```

**数据流 / shape 追踪：**
```
src:                  (4, 20)        ← 原始序列
bos_col:              (4, 1)         ← 全是 BOS id
src_with_bos:         (4, 21)        ← 拼接后
src_key_padding_mask: (4, 21) bool   ← 全 False
encode → hidden:      (4, 21, 128)   ← 每个位置一个向量
hidden[:, 0, :]:      (4, 128)       ← 取 BOS 位置
```

**关键理解：** BOS 位置没有对应氨基酸，它的输出完全来自 attention 聚合，天然适合作为全局表示。

---

### TODO 2：mean_pooling

```python
src_key_padding_mask = (src == PAD)
hidden = model.encode(src, src_key_padding_mask=src_key_padding_mask)
mask = (~src_key_padding_mask).float().unsqueeze(-1)
embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
return embedding
```

**数据流 / shape 追踪：**
```
src:                  (4, 20)
src_key_padding_mask: (4, 20) bool    ← 全 False（等长序列无 PAD）
encode → hidden:      (4, 20, 128)
mask:                 (4, 20, 1)      ← 全 1.0（~False = True → 1.0）
hidden * mask:        (4, 20, 128)    ← PAD 位置乘 0，非 PAD 乘 1
.sum(dim=1):          (4, 128)        ← 对 seq_len 维求和
mask.sum(dim=1):      (4, 1)          ← 非 PAD 个数 = 20
最终 embedding:       (4, 128)        ← 除以有效长度得到均值
```

**关键理解：** `mask.sum(dim=1)` 做除数而非 `seq_len`，确保变长序列下 PAD 位置不稀释均值。

---

### TODO 3：verify_embeddings

```python
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()   # 关闭 Dropout

src = torch.tensor([random.choices(AA_IDS, k=20) for _ in range(4)], ...)

with torch.no_grad():
    cls_emb  = cls_pooling(model, src)
    mean_emb = mean_pooling(model, src)
```

**验证结果：**

| 检查项 | 预期 | 实际 | 状态 |
|---|---|---|---|
| CLS shape | (4, 128) | (4, 128) | ✅ |
| Mean shape | (4, 128) | (4, 128) | ✅ |
| CLS ≠ Mean | diff > 0 | diff ≈ 7.7 ~ 8.5 | ✅ |
| 两次推理一致 | diff ≈ 0 | diff = 0.000000 | ✅ |

---

## 输出分析

### L2 范数对比

| 序列 | CLS 范数 | Mean 范数 | CLS/Mean 比值 |
|---|---|---|---|
| 0 | 10.6471 | 6.5460 | 1.63 |
| 1 | 10.6500 | 6.7355 | 1.58 |
| 2 | 10.6713 | 6.6096 | 1.61 |
| 3 | 10.6675 | 6.7698 | 1.58 |

**观察：**
- CLS 范数 (~10.65) 比 Mean 范数 (~6.6) 大约 1.6 倍
- CLS 位置通过 attention 聚合了整条序列的信息，向量幅度更大
- Mean 是 20 个位置的平均，幅度被"稀释"了
- 两种方法的差异 (L2 ≈ 7.7~8.5) 远大于范数本身，说明方向也不同，不仅仅是尺度差异

### 两次推理一致性

`diff = 0.000000` — `model.eval()` 下 Dropout 关闭，同一输入严格一致。如果忘记 `model.eval()`，Dropout 的随机性会导致两次结果不同（diff > 0）。

---

## 关键知识点

1. **encode vs forward** — 同一个 Transformer 主干，区别只在最后是否过 MLM Head。encode 返回的是"真正的语义表示"，forward 返回的是"token 预测分布"
2. **CLS 池化的原理** — BOS token 没有语义内容，它的输出完全来自对其他位置的 attention 聚合，是天然的全局表示
3. **Mean 池化的 mask 机制** — 用 `(~pad_mask).float()` 做加权，`mask.sum()` 做归一化，确保 PAD 位置不影响结果
4. **model.eval() 的重要性** — 关闭 Dropout/BatchNorm 的随机性，保证推理确定性
5. **两种池化结果不同** — CLS 是"一个代表总结所有人"，Mean 是"集体意见的平均"，信息聚合方式不同，方向和幅度都不同

---

## 踩坑易错点

| 问题 | 原因 | 修正 |
|---|---|---|
| 忘记 `model.eval()` | Dropout 未关闭，两次推理结果不一致 | 加 `model.eval()` |
| Mean 池化用 `seq_len` 做除数 | PAD 位置被计入平均，结果偏低 | 用 `mask.sum(dim=1)` |
| 直接用 `forward` 输出做池化 | MLM Head 的投影改变了向量空间 | 用 `encode` 输出 |
| CLS 池化未拼 BOS | 缺少 BOS token，取不到位置 0 的全局表示 | 先 `torch.cat` 再 encode |

---

## Q1-Q3 回答

**Q1**：`encode` 方法和 `forward` 方法的区别是什么？为什么不直接用 `forward` 的输出来做池化？

> encode 方法只经过 Transformer 编码层，输出每个位置的隐藏状态；forward 方法在 encode 基础上还经过 mlm_head，输出每个位置的 token 预测分布。池化应该基于 encode 的输出，因为我们需要的是隐藏状态，而不是预测分布。

✅ 完全正确。补充一点：MLM Head 是一个 `Linear(d_model, vocab_size)` 的投影层，会把向量从语义空间（d_model=128）映射到预测空间（vocab_size=25），维度都变了，不适合做下游任务的表示。

**Q2**：`model.eval()` 对嵌入提取有什么影响？如果忘记写 `model.eval()`，同一输入两次推理的结果会一样吗？

> model.eval() 会关闭 dropout 和其他训练时特有的行为，确保同一输入得到相同的输出。

✅ 正确。`model.eval()` 影响 `Dropout` 层（训练时随机置零，推理时关闭）和 `BatchNorm` 层（训练时用 batch 统计量，推理时用全局统计量）。我们的模型只有 Dropout 没有 BatchNorm，所以核心影响就是 Dropout。

**Q3**：平均池化时，为什么要用 `mask.sum(dim=1)` 作为除数，而不是直接用 `seq_len`？

> 因为不同序列的有效长度可能不同，直接用 seq_len 会导致 PAD 位置的向量也被平均进去，影响结果。使用 mask.sum(dim=1) 可以正确处理变长序列，只对有效位置的向量求平均。

✅ 准确。具体来说：PAD 位置的 hidden 向量不是零（经过 Transformer 后有非零值），如果直接除以 seq_len，这些"垃圾值"会被混入平均，拉低有效位置的贡献。
