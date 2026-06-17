# Week 9 Day 5 Review — Beam Search 实现

## 代码结构分析

```python
beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

for _ in range(max_len):
    if all done: break                    # TODO 2 ✅
    for beam in beams:
        if beam["done"]: skip             # TODO 3 ✅
        tgt = tensor(beam["tokens"])      # TODO 4 ✅
        logits = decoder(tgt, memory)     # TODO 5 ✅
        log_probs = log_softmax(last)     # TODO 6 ✅
        topk → new candidates             # TODO 7 ✅
    sort by score, keep top beam_width    # TODO 8 ✅

best = max(beams)                         # TODO 9 ✅
```

9 个 TODO 全部完成，逻辑清晰。beam 管理（done 跳过、候选收集、排序筛选）都正确。

---

## 数据流追踪

以 `beam_width=3, src_len=10, vocab_size=20` 为例：

| 阶段 | beams 数量 | 每条展开 | all_candidates |
|------|-----------|---------|----------------|
| 初始 | 1 | — | — |
| Step 1 | 1→3 | 20 个候选，取 top 3 | 20 |
| Step 2 | 3→3 | 每条 20 个，共 60，取 top 3 | 60 |
| Step 3 | 3→3 | 60 取 3 | 60 |
| ... | 3 | ... | ... |
| EOS 后 | done beam 直接保留 | 不再展开 | 减少计算 |

---

## 关键知识点

### 1. Log 概率累加 vs 概率相乘

```python
# ❌ 概率相乘：0.9 × 0.8 × 0.7 × ... → 快速趋近 0，浮点下溢
# ✅ log 累加：log(0.9) + log(0.8) + log(0.7) + ... → 稳定的负数
score = beam["score"] + log_prob.item()
```

主人回答正确——数值稳定性。补充：log 是单调函数，`argmax(p1×p2) = argmax(log(p1)+log(p2))`，排序结果不变。

### 2. Beam Search 的时间复杂度

主人回答：Greedy 的 beam_width 倍。

**评价**：⚠️ 不完全准确。更精确的分析：

```
Greedy 每步：  1 次 Decoder forward（batch=1）
Beam 每步：    beam_width 次 Decoder forward（每条 beam 各一次）

总时间 = beam_width × Greedy 时间
```

但在实际工程中，Beam Search 可以把 beam_width 条 beam **batch 在一起**做一次 forward（batch=beam_width），这样 GPU 利用率更高，实际时间可能 < beam_width × Greedy。不过当前实现是逐条跑的，所以确实是 beam_width 倍。

### 3. Beam Search 优势场景

主人回答：机器翻译中 Greedy 可能选错局部最优。

**评价**：✅ 方向正确。补充更具体的例子：

```
翻译 "bank"：
Greedy: "银行" (0.6) → 后续句子不通
Beam:   保留 "河岸" (0.4) → 后续句子通顺，总分更高
```

关键：**当早期选择的"正确答案"有多个，且后续影响很大时**，Beam Search 优势明显。序列翻转任务太简单（每步只有一个正确答案），所以两者没区别。

---

## 踩坑易错点

### 易错 1：done beam 的处理

```python
# ❌ 错误：跳过 done beam，不加入 candidates
if beam["done"]:
    continue  # done beam 丢失了！

# ✅ 正确：done beam 直接加入 candidates，不展开
if beam["done"]:
    all_candidates.append(beam)
    continue
```

主人做对了。done beam 不能丢——它的 score 可能是最高的。

### 易错 2：tokens 是 list 不是 tensor

```python
candidate = {
    "tokens": beam["tokens"] + [idx.item()],  # list 拼接 ✅
    ...
}
```

注意 `beam["tokens"]` 是 Python list，`idx.item()` 是 int。用 `+` 拼接 list，不是 `torch.cat`。

---

## 输出问题回答评价

### Q1：为什么用 log 概率？

**主人回答**：log 后值线性化，不容易出现极大值极小值。

**评价**：✅ 方向对，表述不够精确。标准说法：概率相乘会导致浮点下溢（多个 <1 的数相乘趋近 0），log 把乘法变加法，数值稳定。且 log 是单调函数，不影响排序。

### Q2：时间复杂度？

**主人回答**：beam_width 倍。

**评价**：⚠️ 基本正确，但应提到"逐条 Decoder forward"这个前提。如果 batch 处理，可以更快。

### Q3：Beam Search 优势场景？

**主人回答**：机器翻译中 Greedy 可能选局部次优。

**评价**：✅ 正确，是 Beam Search 最经典的应用场景。

---

## 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| TODO 完成度 | ✅ 9/9 | 全部完成 |
| beam 管理逻辑 | ✅ | done 跳过、候选收集、排序都对 |
| 结果 | ✅ | 5/5 全对 |
| Q1-Q3 | ⚠️ | 方向正确，Q2 可更精确 |
| 训练 loss | ✅ | 0.0050，正常收敛 |

**核心收获**：Beam Search 本质是"维护 k 条候选路径的贪心"。在这类简单任务上和 Greedy 没区别，但在早期选择有歧义的任务（翻译、摘要）上优势明显。
