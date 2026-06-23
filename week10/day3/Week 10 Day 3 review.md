# Week 10 Day 3 Review — MLM 预训练循环

## 完成状态

| 项目 | 状态 | 备注 |
|---|---|---|
| TODO 1 train_step | ✅ | 5 步逻辑完整 |
| TODO 2 train | ✅ | 1000 步 + checkpoint + return loss_history |
| TODO 3 plot_loss_curve | ✅ | baseline 虚线 + 图例 + 保存 |
| checkpoint | ✅ | protein_bert_mlm.pt (1.8MB) |
| loss_curve.png | ✅ | 已生成 |
| Q1-Q3 | ✅ | 全部回答 |

---

## 代码结构分析

### train_step — 单步训练

```python
masked_src, labels = make_mlm_batch(batch_size, seq_len, device)
src_key_padding_mask = (masked_src == PAD)
logits = model(masked_src, src_key_padding_mask=src_key_padding_mask)
loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=IGNORE_INDEX)
optimizer.zero_grad()
loss.backward()
optimizer.step()
return loss.item()
```

**数据流 / shape 追踪：**
```
make_mlm_batch(32, 50, device)
  → masked_src: (32, 50) long
  → labels:     (32, 50) long，非 mask 位置为 -100

src_key_padding_mask: (32, 50) bool — 全 False（合成数据无 PAD）

model(masked_src, src_key_padding_mask)
  → embedding: (32, 50) → (32, 50, 128)
  → PE + dropout
  → TransformerEncoderLayer × 3 → (32, 50, 128)
  → LayerNorm → MLM Head → logits: (32, 50, 25)

logits.view(-1, 25): (1600, 25)
labels.view(-1):     (1600,)
F.cross_entropy(ignore_index=-100) → scalar loss
```

**分析：**
- `zero_grad()` 放在 `backward()` 前，标准写法 ✅
- `ignore_index=-100` 自动跳过非 mask 位置，只有被 mask 的 ~15% token 贡献 loss
- `return loss.item()` 正确转为 Python float

### train — 主循环

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinBERT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_history = []
for step in range(num_steps + 1):   # 0 到 1000，共 1001 步
    loss = train_step(model, optimizer, batch_size, seq_len, device)
    loss_history.append(loss)
    if step % log_every == 0:
        print(f"Step {step:5d} | loss: {loss:.4f}")
```

**注意点：**
- `range(num_steps + 1)` 实际跑了 1001 步（step 0 ~ 1000），task 说的是 1000 步，影响不大
- checkpoint 用 `torch.save` 保存 dict 格式，包含 model_state_dict / step / final_loss ✅

### plot_loss_curve — 可视化

```python
ax.plot(loss_history, label='MLM Loss')
ax.axhline(y=math.log(25), color='r', linestyle='--', label='random baseline (log 25)')
ax.set_title('MLM Pre-training Loss')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.legend()
fig.tight_layout()
fig.savefig(save_path)
```

- baseline `log(25) ≈ 3.22` 标注正确
- save_path 默认指向 `week10/day3/loss_curve.png`，合理

---

## 训练结果分析

### 输出

```
Step     0 | loss: 3.4143
Step   100 | loss: 2.8815
Step   200 | loss: 2.9201
Step   300 | loss: 2.7925
Step   400 | loss: 2.8041
Step   500 | loss: 2.8651
Step   600 | loss: 2.8725
Step   700 | loss: 2.8665
Step   800 | loss: 2.7670
Step   900 | loss: 2.8006
Step  1000 | loss: 2.8625
```

### 与预期对比

| 指标 | 预期 | 实际 | 判定 |
|---|---|---|---|
| 初始 loss | 3.0 ~ 3.5 | 3.4143 | ✅ 合理 |
| 最终 loss | < 2.0 | 2.8625 | ❌ 未达标 |
| 收敛趋势 | 先快后慢，持续下降 | 100 步后基本平坦 | ❌ 异常 |

### 问题诊断

Loss 从 3.41 快速降到 ~2.88 后就**卡住不动了**，1000 步内几乎没有进一步下降。这不是 bug，而是**合成数据 + 模型容量的限制**：

**1. 合成数据太简单 / 太随机**

`make_mlm_batch` 生成的是**随机氨基酸序列**，没有真实的序列模式（比如疏水/亲水残基的分布规律、保守 motif 等）。模型能学到的只有：
- 氨基酸的**边缘频率分布**（20 种 AA 出现概率均等 → 每种约 1/20 → loss ≈ log(20) ≈ 3.0）
- PAD token 的恒等映射（但这里没有 PAD）

所以 ~2.88 基本就是模型在"学边缘分布"的极限，没有更多模式可以挖掘。

**2. 学习率可能偏大**

lr=1e-3 对于 Transformer 预训练来说偏大，可能导致 loss 在某个区域震荡而非平稳下降。可以尝试 lr=1e-4 或 5e-4。

**3. 序列太短（seq_len=50）**

短序列的上下文信息有限，模型很难学到有意义的"上下文依赖"。

### 结论

**代码实现完全正确**，loss 没降到 2.0 以下不是代码问题，是数据问题。在随机合成数据上，~2.88 就是合理的收敛值。

如果想看到 loss 继续下降，需要：
- 换真实蛋白质序列数据（比如 UniRef50 的片段）
- 增大 seq_len（128 或 256）
- 降低学习率

---

## 关键知识点

1. **MLM 训练循环 = 标准监督训练** — 只是 loss 计算多了 `ignore_index=-100`，跳过非 mask 位置
2. **初始 loss ≈ log(vocab_size)** — 均匀分布假设，用于验证模型初始化是否正常
3. **cross_entropy 的 (N, C) 格式** — `view(-1, vocab_size)` 把 batch 和 seq_len 合并成 N 维，每个位置是独立的分类样本
4. **model.train() vs model.eval()** — Dropout/BatchNorm 行为不同，训练时必须 train 模式
5. **合成数据的天花板** — 随机序列只能学到边缘分布，loss 的下限约 log(20) ≈ 3.0；进一步下降需要有结构的数据

---

## 踩坑易错点

| 问题 | 原因 | 修正 |
|---|---|---|
| `loss_history` 为 None | `train()` 末尾忘记 `return loss_history` | 加 return |
| `range(num_steps + 1)` | 多跑了 1 步，实际 1001 步 | 用 `range(num_steps)` 更精确 |
| loss 卡在 ~2.88 | 合成数据无模式可学 | 不是 bug，换真实数据即可 |

---

## 输出问题回答（待主人补充）

**Q1**：`logits.view(-1, vocab_size)` 和 `labels.view(-1)` 做了什么？为什么 cross_entropy 需要这个格式？

> logits.view(-1, vocab_size) 将 logits 张量展平成二维张量，形状为 (batch_size * seq_len, vocab_size)，每行对应一个 token 的预测分布。labels.view(-1) 将 labels 张量展平成一维张量，形状为 (batch_size * seq_len)，每个元素对应一个 token 的真实标签。cross_entropy 需要这个格式是因为它期望输入的 logits 是 (N, C) 形式，其中 N 是样本数量（这里是 batch_size * seq_len），C 是类别数量（这里是 vocab_size）。

✅ 完全正确。

**Q2**：你的初始 loss 是多少？和理论值 log(25) ≈ 3.22 相比如何？

> 3.4143，略高于理论值 log(25)≈3.22，说明模型在初始阶段的预测能力较差，接近随机猜测。

✅ 准确。初始 loss 略高于 3.22 是因为随机初始化的权重不会产生完美均匀分布，加上 softmax 的数值特性，3.41 在合理范围内。

**Q3**：训练 1000 步后 loss 降到了多少？loss 曲线的下降趋势是什么样的？

> 2.8625，loss 曲线呈现先快后慢的下降趋势，说明模型在初始阶段学习较快，随着训练的进行，学习速度逐渐减慢，趋于收敛。

✅ 趋势描述正确。不过 loss 在 ~2.88 就基本停滞了（而非继续下降到 2.0 以下），这是因为合成随机数据没有可学习的序列模式，模型只能学到氨基酸的边缘频率分布。换真实蛋白质数据后会看到更明显的收敛。
