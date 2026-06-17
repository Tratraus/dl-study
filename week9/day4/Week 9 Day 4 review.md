# Week 9 Day 4 Review — 自回归推理（Greedy Decoding）

## 代码结构分析

```python
# 推理流程：
memory = encoder(src)                    # 编码源序列
generated = [BOS]                        # 从 BOS 开始
for _ in range(max_len):
    logits = decoder(generated, memory)  # 喂入已生成序列
    next_token = argmax(logits[:, -1])   # 取最后一个位置
    generated.append(next_token)         # 拼接
    if all have EOS: break
return generated[1:]                     # 去掉 BOS
```

greedy_decode 实现完整，逻辑清晰。evaluate 函数正确打印 src / expected / pred。

---

## 踩坑：seq_len 不匹配导致推理全错

### 现象

首次运行时 5 条全部 ❌，pred 比 expected 多 1~2 个重复前缀 token：

```
src:      [11, 18, 12, 7, 12, 10, 17, 18]   ← 长度 8
expected: [18, 17, 10, 12, 7, 12, 18, 11]   ← 长度 8
pred:     [18, 18, 18, 17, 10, 12, 7, 12, 18, 11]  ← 长度 10
```

### 根因

**训练时 `SEQ_LEN = 10`，推理时 `make_batch(n_samples, 8)` 用了长度 8。**

```
训练：src 长度 10 → tgt 长度 12 (含 BOS/EOS) → tgt_input 长度 11
推理：src 长度 8  → tgt 长度 10 → tgt_input 长度 9
```

虽然 learnable positional embedding 覆盖了所有位置（0~9 都训练过），但模型整体在**特定长度的翻转模式**上学习得更好——长度 10 翻转和长度 8 翻转是不同的分布，模型对后者的泛化不如前者稳定。

### 修复

`evaluate` 中 `make_batch(n_samples, 10, device)` → 5/5 ✅

### 教训

**推理时的输入分布必须和训练时一致。** 任何维度的差异（seq_len、vocab 范围、数据分布）都可能导致模型表现下降。这是机器学习工程中最常见也最容易忽视的 bug。

---

## 关键知识点

### 1. Greedy Decoding 的自回归循环

每步只看最后一个位置的 logits：

```python
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

为什么是 `-1`？因为 `logits` 的 shape 是 `(batch, current_len, vocab_size)`，最后一个位置的输出就是"预测下一个 token"。

### 2. 因果 mask 在推理时的作用

推理时 Decoder 的输入只有已生成的 token（没有"未来"），因果 mask **理论上不是必须的**。但加上它有两个好处：

1. **和训练行为保持一致**：训练时有 mask，推理时没有，模型看到的注意力模式不同，可能导致分布偏移
2. **防御性编程**：统一加 mask，避免未来修改代码时出错

### 3. Greedy vs Beam Search

Greedy 每步取 argmax，不能保证全局最优。例：

```
Step 1: A 概率 0.51, B 概率 0.49 → 选 A
Step 2: 给定 A，最优后续 = [A, C, D]，总概率 0.51×0.8 = 0.408
        给定 B，最优后续 = [B, E, F]，总概率 0.49×0.95 = 0.4655
                                 ↑ 全局最优，但 Greedy 错过了
```

---

## 输出问题回答评价

### Q1：推理时为什么还需要因果 mask？

**主人回答**：推理时也需要 mask，防止 attend 到未来位置。

**评价**：⚠️ 半对半错。推理时 Decoder 的输入**本身就没有未来 token**（输入就是已生成的全部），所以"防止 attend 到未来"这个理由不成立。真正原因是**和训练保持一致**——训练时有 mask，推理时没有，注意力模式不同会导致分布偏移。你的 Answer 部分修正得不错。

### Q2：为什么去掉 BOS？

**主人回答**：BOS 是输入标记，不是生成内容的一部分。

**评价**：✅ 正确。BOS 的作用是告诉 Decoder"开始生成"，它本身不是输出序列的一部分。就像写文章时"开始写"这个指令不算文章的第一段。

### Q3：Greedy 的缺陷？

**主人回答**：贪心只看当前最优，可能错过全局最优。

**评价**：✅ 方向正确，但例子不够具体。更好的例子：

```
翻译 "I love you"
Step 1: "我" (0.5) vs "俺" (0.4) → 选 "我"
Step 2: 给定 "我"，"爱" (0.3) → "我 爱"
        给定 "俺"，"喜欢" (0.9) → "俺 喜欢"  ← 概率更高但 Greedy 错过了
```

---

## 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| TODO 完成度 | ✅ 3/3 | greedy_decode + evaluate 完整 |
| 推理逻辑 | ✅ | 自回归循环正确 |
| 初始 bug | ⚠️ | seq_len 8 vs 10 不匹配，已修复 |
| Q1 回答 | ⚠️ | 理由不准确，已自我修正 |
| Q2-Q3 回答 | ✅ | 正确 |
| 最终结果 | ✅ | 5/5 全对 |

**核心收获**：推理时的输入分布必须和训练时一致。seq_len、vocab 范围、数据分布——任何维度的差异都可能导致模型表现下降。这是 ML 工程中最常见也最容易忽视的 bug。
