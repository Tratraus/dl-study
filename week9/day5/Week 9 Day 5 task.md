# Week 9 Day 5：Beam Search 实现

---

## 任务目标

在 Greedy Decoding 的基础上，实现 **Beam Search**，理解它如何通过维护多条候选路径来找到更优的序列。

---

## 最小必要理论

### 1. Beam Search 的核心思想

Greedy 每步只保留 1 个最优 token，Beam Search 保留 **k 条最优路径**（k = beam_width）。

```text
beam_width = 3，词表大小 = 5

初始：
  候选路径：[ [<BOS>] ]  得分：[ 0.0 ]

第 1 步，展开所有候选，每条路径生成 5 个可能的下一个 token：
  [<BOS>, A]  得分 = log P(A|<BOS>)
  [<BOS>, B]  得分 = log P(B|<BOS>)
  [<BOS>, C]  得分 = log P(C|<BOS>)
  [<BOS>, D]  得分 = log P(D|<BOS>)
  [<BOS>, E]  得分 = log P(E|<BOS>)
  保留得分最高的 3 条：假设是 A、C、E

第 2 步，对 3 条路径各展开 5 个，共 15 条候选：
  [<BOS>, A, ?]  × 5
  [<BOS>, C, ?]  × 5
  [<BOS>, E, ?]  × 5
  保留得分最高的 3 条

...直到所有路径都生成了 <EOS>
```

得分用 **log 概率累加**（而不是概率相乘），避免数值下溢：

$$\text{score}(y_1, ..., y_t) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, \text{memory})$$

---

### 2. 长度惩罚

Beam Search 倾向于选短序列（因为每步 log 概率都是负数，步数越多得分越低）。

实际工程中会加**长度归一化**：

$$\text{score\_normalized} = \frac{\text{score}}{L^\alpha}$$

其中 $$L$$ 是序列长度，$$\alpha$$ 通常取 0.6~0.7。

今天先不实现长度惩罚，了解概念即可。

---

### 3. 实现思路（单条序列）

为了简单，今天先实现**单条序列**的 Beam Search（batch_size=1）：

```text
beams = [
    {"tokens": [BOS],         "score": 0.0, "done": False},
    {"tokens": [BOS],         "score": 0.0, "done": False},
    {"tokens": [BOS],         "score": 0.0, "done": False},
]

每步：
  1. 对所有未完成的 beam，跑 Decoder，得到下一步的 log 概率分布
  2. 每个 beam 展开 vocab_size 个候选
  3. 从所有候选中选出得分最高的 beam_width 个
  4. 更新 beams
  5. 如果某个 beam 生成了 EOS，标记为 done
  6. 所有 beam 都 done，退出
```

---

## 代码任务

新建文件：`week9/day5/beam_search.py`

复用 Day 3/4 的模型，只需要实现推理函数：

```python
import torch
import torch.nn.functional as F

# 复用之前的模型定义和训练好的权重
# （把 Day 4 的完整代码复制过来，在 train() 末尾保存模型）

# ── 在 train() 末尾添加 ───────────────────────────────────────
# torch.save(model.state_dict(), "seq2seq.pt")

# ── Beam Search ──────────────────────────────────────────────
@torch.no_grad()
def beam_search(model, src, beam_width, max_len, device):
    """
    单条序列的 Beam Search。

    输入：
      src:        (1, src_len)   ← 注意是单条
      beam_width: int
      max_len:    int
    输出：
      best_tokens: List[int]，最优序列（不含 BOS，含 EOS）
    """
    model.eval()

    # TODO 1：Encoder 编码
    memory = model.encoder(src)   # (1, src_len, d_model)

    # 初始化 beams
    # 每个 beam 是一个 dict：
    #   "tokens": List[int]，当前已生成的 token（含 BOS）
    #   "score":  float，累计 log 概率
    #   "done":   bool，是否已生成 EOS
    beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

    for _ in range(max_len):

        # TODO 2：如果所有 beam 都 done，退出
        ...

        all_candidates = []

        for beam in beams:
            # TODO 3：跳过已完成的 beam（直接加入 candidates，不展开）
            ...

            # TODO 4：把当前 beam 的 tokens 转成 tensor
            # shape: (1, current_len)
            tgt = torch.tensor([beam["tokens"]], device=device)

            # TODO 5：生成因果 mask，Decoder 前向
            # logits shape: (1, current_len, vocab_size)
            ...

            # TODO 6：取最后一个位置，计算 log softmax
            # log_probs shape: (vocab_size,)
            ...

            # TODO 7：展开 beam_width 个最优候选
            # 提示：用 torch.topk(log_probs, beam_width)
            # 对每个候选，新 score = beam["score"] + log_prob
            ...

        # TODO 8：从 all_candidates 中选出得分最高的 beam_width 个
        # 提示：按 score 排序，取前 beam_width 个
        ...

        beams = ...

    # TODO 9：从所有 beam 中选出得分最高的，返回其 tokens（去掉 BOS）
    best = max(beams, key=lambda b: b["score"])
    return best["tokens"][1:]   # 去掉 BOS


# ── 对比验证 ─────────────────────────────────────────────────
def compare(model, device, n_samples=5):
    """
    对比 Greedy 和 Beam Search 的结果。
    """
    model.eval()
    src_batch, tgt_batch = make_batch(n_samples, 10, device)

    print("\n── Greedy vs Beam Search ──")
    for i in range(n_samples):
        src = src_batch[i].unsqueeze(0)   # (1, src_len)
        expected = tgt_batch[i][1:-1].tolist()

        # Greedy
        greedy_out = greedy_decode(model, src, max_len=20, device=device)
        greedy_pred = greedy_out[0].tolist()
        if EOS in greedy_pred:
            greedy_pred = greedy_pred[:greedy_pred.index(EOS)]

        # Beam Search
        beam_pred = beam_search(model, src, beam_width=3, max_len=20, device=device)
        if EOS in beam_pred:
            beam_pred = beam_pred[:beam_pred.index(EOS)]

        g_mark = "✅" if greedy_pred == expected else "❌"
        b_mark = "✅" if beam_pred   == expected else "❌"

        print(f"src:      {src_batch[i].tolist()}")
        print(f"expected: {expected}")
        print(f"greedy  {g_mark}: {greedy_pred}")
        print(f"beam    {b_mark}: {beam_pred}")
        print()
```

---

## 完成标准

1. `beam_search` 实现完整，无报错
2. Beam Search 结果正确率 ≥ Greedy（在这个简单任务上两者应该都接近 100%）
3. 能回答下面三个问题

---

## 输出问题

**Q1**：为什么用 **log 概率累加** 而不是**概率直接相乘**？

**Q2**：Beam Search 的时间复杂度是 Greedy 的多少倍？（从每步的计算量角度分析）

**Q3**：在这个序列翻转任务上，Beam Search 和 Greedy 的结果几乎一样。什么样的任务上 Beam Search 会有明显优势？（提示：想想什么情况下"局部最优 ≠ 全局最优"）

---

准备好后提交代码、终端输出和三个问题的回答。