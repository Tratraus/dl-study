# Week 9 Day 6.5 Review：长度惩罚 + 温度采样 / Top-k Sampling

## 代码结构总览

```
sampling.py
├── 模型定义（Encoder / Decoder / Seq2Seq）   ← 从 Day 5 复制
├── make_causal_mask / make_batch             ← 从 Day 5 复制
├── greedy_decode / beam_search / evaluate    ← 从 Day 5 复制
├── beam_search_with_length_penalty           ← TODO 1: 长度归一化
├── temperature_sample                        ← TODO 2: 温度采样
├── topk_sample                               ← TODO 3: Top-k 采样
├── compare_all()                             ← 五种策略对比
└── train() → return model, device            ← 训练 + 返回模型
```

---

## TODO 1：beam_search_with_length_penalty ✅ 正确

```python
beams = sorted(
    all_candidates,
    key=lambda b: b["score"] / (max(len(b["tokens"]) - 1, 1) ** alpha),
    reverse=True
)[:beam_width]
```

**分析**：
- `len(tokens) - 1`：去掉 BOS，得到实际生成长度
- `max(..., 1)`：防止除零（虽然理论上不会出现长度为 0 的 beam）
- `alpha=0` 时分母 = 1，退化为原始 beam search ✅
- 最终选 best 也用同样的归一化公式 ✅
- 累计 score 本身保持原始 log 概率（不归一化），只在排序时归一化——这正是 task 要求的做法 ✅

---

## TODO 2：temperature_sample ✅ 正确

```python
scaled_logits = logits[:, -1, :] / temperature
probs = F.softmax(scaled_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**分析**：
- 结构和 greedy_decode 完全一致，唯一区别：`argmax` → `multinomial`
- 温度缩放在 softmax 之前，逻辑正确
- `torch.multinomial` 从概率分布中采样，返回 `(1, 1)` 形状的 tensor，直接 `cat` ✅

**温度效果**：
- T < 1：分布更尖锐 → 更接近 greedy
- T = 1：原始分布
- T > 1：分布更平坦 → 更随机

---

## TODO 3：topk_sample ✅ 正确

```python
topk_values, _ = torch.topk(last_logits, k)
threshold = topk_values[0, -1]                    # 第 k 大的值
last_logits = last_logits.masked_fill(last_logits < threshold, float('-inf'))
scaled_logits = last_logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**分析**：
- `torch.topk` 返回的 values 已排序，`[0, -1]` 取最后一个 = 第 k 大的值 ✅
- `masked_fill` 比手动赋值更 PyTorch 风格 ✅
- `< threshold` 而不是 `<= threshold`：保留等于阈值的 token（如果多个 token 并列第 k 大，都保留）✅
- 截断后再 temperature + softmax + 采样，顺序正确 ✅

**注意**：`last_logits` 先 `.unsqueeze(0)` 变成 `(1, vocab_size)`，是为了后面 `torch.multinomial` 能正常工作。最后 `next_token` 需要 `.unsqueeze(0).unsqueeze(0)` 才能和 `generated` 拼接，稍显繁琐但正确。

---

## compare_all 输出分析

### T=0.8：五种策略输出完全一致

```
expected:      [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
greedy      ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
beam        ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
beam+len   ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
temp(0.8)     : [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]  ← 随机，不评对错
topk(5,0.8)   : [5, 10, 13, 5, 8, 19, 9, 4, 12, 15] ← 随机，不评对错
```

**所有样本、所有策略输出完全相同。**

### T=1.2：几乎一致，仅一处差异

```
topk(5,1.2)   : [19, 7, 3, 18, 18, 16, 14, 7, 16, 5]  ← 第 9 位 16 vs 18
expected:      [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]
```

Top-k 在 T=1.2 时终于出现了一次不同输出。

### 为什么采样策略没有"随机性"？

**根本原因：模型把序列翻转学得太好了，logits 分布极端尖锐。**

序列翻转是确定性任务——同一个输入永远对应同一个输出。训练 500 步后 loss 降到 0.003，模型对正确 token 给出接近 1.0 的概率，其他 token 概率接近 0。

在这种情况下：
- **T=0.8**：缩放后分布更尖锐，正确 token 概率 → 1.0+，采样几乎 100% 选它
- **T=1.2**：缩放后分布稍微变平，但正确 token 仍然占绝对主导
- **Top-k(k=5)**：即使只保留 top 5，第 1 名概率远超其他 4 个之和

**类比**：就像掷一个六个面都写着 6 的骰子——不管你怎么"随机"掷，结果都是 6。

### 要看到真正的随机性差异

需要一个**一对多映射**的任务（比如 Day 6 的随机蛋白质生成），或者故意降低训练质量（比如只训 50 步）。在确定性任务上，训练越完美，采样策略越没用武之地。

---

## 代码质量

| 项目 | 评价 |
|------|------|
| TODO 1 长度归一化 | ✅ 正确，alpha=0 退化逻辑完整 |
| TODO 2 温度采样 | ✅ 正确，三行核心逻辑清晰 |
| TODO 3 Top-k 采样 | ✅ 正确，masked_fill 风格好 |
| compare_all | ✅ 正确，T 参数化 + 两种温度对比 |
| train 返回值 | ✅ 改为 return model, device |
| __main__ 入口 | ✅ 正确调用 |

---

## 输出问题回答

**Q1: T=0.8 和 T=1.2 的区别？**

✅ 回答基本正确。补充：在序列翻转任务上区别几乎不可见（模型太确定了）。T=1.2 时 top-k 出现了一次不同，说明高温确实增加了采样随机性，只是这个任务里模型太自信，随机性没有空间施展。

**Q2: k 的 trade-off？**

✅ 回答正确。k=1 退化 greedy，k=词表大小退化完全随机。通常 k=5~50，取决于任务和模型置信度。

**Q3: 蛋白质生成用哪种策略？**

✅ 回答方向正确。补充：蛋白质生成场景下，如果要探索新序列（蛋白质设计），Top-k + 中等温度（T=0.8~1.0, k=10~20）比较合适——既保持生物学合理性，又有一定多样性。如果要生成高置信度序列（结构预测），Greedy 或 Beam Search 更稳。

---

## 总结

| 项目 | 评价 |
|------|------|
| 三个 TODO | ✅ 全部正确 |
| 代码结构 | ✅ 从 Day 5 复制 + 新增函数，完整可运行 |
| 输出现象 | 符合预期——确定性任务 + 完美训练 = 采样无用武之地 |
| 问题回答 | ✅ 全部正确，Q3 有思考 |
| 建议 | 如果想真正体验采样效果，可以在 Day 6 的蛋白质任务上尝试 Top-k/Temperature |
