# Week 9 Day 6.5（拓展）：长度惩罚 + 温度采样 / Top-k Sampling

---

## 任务目标

在 Day 5 的 Beam Search 基础上，实现三个生成策略的改进：
1. **长度归一化**：修复 Beam Search 偏好短序列的问题
2. **Temperature Sampling**：控制生成的"随机性"
3. **Top-k Sampling**：截断长尾分布，避免采样到低概率 token

---

## 最小必要理论

### 1. 长度归一化

Beam Search 的得分是 log 概率的累加，序列越长得分越低（因为每个 log prob 都是负数）。

$$\text{score\_raw} = \sum_{i=1}^{L} \log P(y_i)$$

加入长度惩罚后：

$$\text{score\_normalized} = \frac{\sum_{i=1}^{L} \log P(y_i)}{L^\alpha}$$

$\alpha = 0$ 时退化为原始 Beam Search，$\alpha = 1$ 时完全按平均 log 概率排序。通常取 $\alpha = 0.6$。

---

### 2. Temperature Sampling

Greedy 和 Beam Search 都是**确定性**的，同样的输入永远输出同样的结果。

Temperature 通过缩放 logits 来控制分布的"尖锐程度"：

$$P(y_i) = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

```text
T → 0：分布趋向 one-hot，退化为 Greedy（确定性）
T = 1：原始分布
T → ∞：分布趋向均匀分布（完全随机）
```

然后从这个分布中**采样**（而不是 argmax）。

---

### 3. Top-k Sampling

Temperature Sampling 的问题：即使 T 很小，仍然可能采样到概率极低的 token（长尾问题）。

Top-k：只保留概率最高的 k 个 token，其余设为 `-inf`，再 softmax + 采样：

```text
原始分布：[0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
k=3 后：  [0.4, 0.3, 0.2,  -∞,   -∞,   -∞ ]
重新归一：[0.44, 0.33, 0.22,  0,    0,    0 ]
```

---

## 代码任务

新建文件：`week9/day6_5/sampling.py`

复用 Day 5 的完整代码（模型 + 训练），在此基础上添加：

```python
# ── 1. 带长度归一化的 Beam Search ────────────────────────────
@torch.no_grad()
def beam_search_with_length_penalty(model, src, beam_width, max_len, device, alpha=0.6):
    """
    在 Day 5 的 beam_search 基础上，排序时使用长度归一化得分。

    唯一的修改点：
      选出最优 beam 时，用 score / (len ** alpha) 排序
      但累计 score 本身仍然是原始 log 概率之和（不归一化）
    """
    model.eval()
    memory = model.encoder(src)
    beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

    for _ in range(max_len):
        if all(b["done"] for b in beams):
            break

        all_candidates = []
        for beam in beams:
            if beam["done"]:
                all_candidates.append(beam)
                continue

            tgt = torch.tensor([beam["tokens"]], device=device)
            tgt_mask = make_causal_mask(tgt.size(1), device)
            logits = model.decoder(tgt, memory, tgt_mask=tgt_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_tokens = beam["tokens"] + [idx.item()]
                all_candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + log_prob.item(),
                    "done": idx.item() == EOS
                })

        # TODO 1：排序时使用长度归一化
        # 提示：排序 key = score / (len(tokens) ** alpha)
        # 注意：len(tokens) 包含 BOS，所以实际生成长度 = len(tokens) - 1
        # 注意：alpha=0 时应退化为原始 beam search
        ...

        beams = all_candidates[:beam_width]

    best = max(beams, key=lambda b: b["score"] / (max(len(b["tokens"]) - 1, 1) ** alpha))
    return best["tokens"][1:]


# ── 2. Temperature Sampling ───────────────────────────────────
@torch.no_grad()
def temperature_sample(model, src, max_len, device, temperature=1.0):
    """
    用 Temperature Sampling 生成序列。
    结构和 greedy_decode 几乎一样，只改一行。

    输入：
      src:         (1, src_len)   ← 单条序列
      temperature: float，控制随机性
    输出：
      generated:   List[int]，不含 BOS，含 EOS
    """
    model.eval()
    memory = model.encoder(src)
    generated = torch.full((1, 1), BOS, device=device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(generated.size(1), device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)

        # TODO 2：用 temperature 缩放 logits，然后采样（不是 argmax）
        # 提示：torch.multinomial(probs, num_samples=1)
        ...

        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == EOS:
            break

    return generated[0, 1:].tolist()


# ── 3. Top-k Sampling ────────────────────────────────────────
@torch.no_grad()
def topk_sample(model, src, max_len, device, k=10, temperature=1.0):
    """
    Top-k Sampling：只从概率最高的 k 个 token 中采样。

    输入：
      src:         (1, src_len)
      k:           int，保留的 token 数量
      temperature: float
    输出：
      generated:   List[int]
    """
    model.eval()
    memory = model.encoder(src)
    generated = torch.full((1, 1), BOS, device=device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(generated.size(1), device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)

        last_logits = logits[0, -1, :]   # (vocab_size,)

        # TODO 3：实现 Top-k 截断
        # step 1: 找到第 k 大的 logit 值（阈值）
        # step 2: 把低于阈值的位置设为 -inf
        # step 3: 用 temperature 缩放，softmax，采样
        # 提示：torch.topk(last_logits, k) 返回 (values, indices)
        #       阈值 = values[-1]（第 k 大的值）
        ...

        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_token.item() == EOS:
            break

    return generated[0, 1:].tolist()


# ── 对比实验 ─────────────────────────────────────────────────
def compare_all(model, device, n_samples=3):
    model.eval()
    src_batch, tgt_batch = make_batch(n_samples, 10, device)

    print("\n── 四种解码策略对比 ──")
    for i in range(n_samples):
        src = src_batch[i].unsqueeze(0)
        expected = tgt_batch[i][1:-1].tolist()

        # Greedy
        g = greedy_decode(model, src, 20, device)[0].tolist()
        if EOS in g: g = g[:g.index(EOS)]

        # Beam Search（无长度惩罚）
        b = beam_search(model, src, 3, 20, device)
        if EOS in b: b = b[:b.index(EOS)]

        # Beam Search（有长度惩罚）
        bl = beam_search_with_length_penalty(model, src, 3, 20, device, alpha=0.6)
        if EOS in bl: bl = bl[:bl.index(EOS)]

        # Temperature Sampling（T=0.8）
        t = temperature_sample(model, src, 20, device, temperature=0.8)
        if EOS in t: t = t[:t.index(EOS)]

        # Top-k Sampling（k=5, T=0.8）
        tk = topk_sample(model, src, 20, device, k=5, temperature=0.8)
        if EOS in tk: tk = tk[:tk.index(EOS)]

        g_mark  = "✅" if g  == expected else "❌"
        b_mark  = "✅" if b  == expected else "❌"
        bl_mark = "✅" if bl == expected else "❌"

        print(f"expected:      {expected}")
        print(f"greedy      {g_mark}: {g}")
        print(f"beam        {b_mark}: {b}")
        print(f"beam+len   {bl_mark}: {bl}")
        print(f"temp(0.8)     : {t}  ← 随机，不评对错")
        print(f"topk(5,0.8)   : {tk} ← 随机，不评对错")
        print()
```

---

## 完成标准

1. 三个函数均实现完整，无报错
2. Greedy / Beam / Beam+length 在翻转任务上全部 ✅
3. Temperature 和 Top-k 输出合法 token（在词表范围内）
4. 能回答下面三个问题

---

## 输出问题

**Q1**：Temperature = 0.8 和 Temperature = 1.2 生成的序列有什么区别？运行几次，观察随机性的变化。

**Q2**：Top-k 中，k 的选择有什么 trade-off？k 太小和 k 太大分别会导致什么问题？

**Q3**：在蛋白质序列生成场景中，你会选择哪种解码策略？为什么？（没有标准答案，说出你的思考即可）

---

准备好后提交代码、终端输出和三个问题的回答。