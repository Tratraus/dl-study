# Week 9 Day 4：自回归推理（Greedy Decoding）

---

## 任务目标

在 Day 3 训练好的模型上，实现推理阶段的**自回归解码**，并验证模型真的学会了序列翻转。

---

## 最小必要理论

### 1. 训练 vs 推理的本质区别

```text
训练时（Teacher Forcing）：
  一次性把整个 tgt_input 喂给 Decoder
  并行计算所有位置的 logits
  快，但依赖真实标签

推理时（自回归）：
  没有真实标签
  从 <BOS> 开始，每次只生成一个 token
  把生成的 token 拼回输入，再预测下一个
  慢，但这才是真实使用场景
```

### 2. 自回归解码的步骤

```text
初始状态：
  generated = [<BOS>]

第 1 步：
  Decoder 输入 [<BOS>]
  取最后一个位置的 logits → argmax → token_1
  generated = [<BOS>, token_1]

第 2 步：
  Decoder 输入 [<BOS>, token_1]
  取最后一个位置的 logits → argmax → token_2
  generated = [<BOS>, token_1, token_2]

...直到生成 <EOS> 或达到最大长度
```

关键点：**每步只看最后一个位置的输出**，因为那是"下一个 token"的预测。

### 3. Greedy vs Beam Search

| 方法 | 策略 | 质量 | 速度 |
|------|------|------|------|
| Greedy | 每步取概率最大的 token | 一般 | 最快 |
| Beam Search | 维护 k 条候选路径 | 更好 | 慢 k 倍 |
| Sampling | 按概率分布采样 | 多样性好 | 快 |

今天实现最简单的 **Greedy Decoding**。

---

## 代码任务

在 `week9/day3/train_seq2seq.py` 末尾添加推理函数，并在训练结束后调用：

```python
# ── 自回归推理 ────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """
    对单条或一批 src 序列进行 Greedy Decoding。

    输入：
      src:     (batch, src_len)
      max_len: 最大生成长度（防止无限循环）
    输出：
      generated: (batch, generated_len)  不含 <BOS>，含 <EOS>
    """
    model.eval()

    # TODO 1：用 Encoder 编码 src，得到 memory
    # memory shape: (batch, src_len, d_model)
    ...

    batch_size = src.size(0)

    # TODO 2：初始化 generated，shape = (batch, 1)，全部填 BOS
    ...

    # TODO 3：自回归循环
    for _ in range(max_len):
        # step 1: 生成因果 mask
        ...

        # step 2: Decoder 前向
        # logits shape: (batch, current_len, vocab_size)
        ...

        # step 3: 取最后一个位置的 logits，argmax 得到下一个 token
        # next_token shape: (batch, 1)
        ...

        # step 4: 拼接到 generated
        ...

        # step 5: 如果所有样本都生成了 <EOS>，提前退出
        # 提示：检查 generated 中是否每行都含有 EOS
        ...

    # 去掉开头的 <BOS>，返回
    return generated[:, 1:]


# ── 验证函数 ─────────────────────────────────────────────────
def evaluate(model, device, n_samples=5):
    """
    随机生成 n_samples 条样本，打印 src、期望输出、模型输出。
    """
    model.eval()
    src, tgt = make_batch(n_samples, 8, device)

    generated = greedy_decode(model, src, max_len=20, device=device)

    print("\n── 推理结果 ──")
    for i in range(n_samples):
        src_list = src[i].tolist()
        # tgt 是 [<BOS>, ...翻转..., <EOS>]，去掉首尾
        expected = tgt[i][1:-1].tolist()
        # generated 可能含 <EOS>，截断到第一个 <EOS>
        pred = generated[i].tolist()
        if EOS in pred:
            pred = pred[:pred.index(EOS)]

        match = "✅" if pred == expected else "❌"
        print(f"{match} src:      {src_list}")
        print(f"   expected: {expected}")
        print(f"   pred:     {pred}")
        print()
```

然后在 `train()` 函数末尾加上：

```python
evaluate(model, device)
```

---

## 完成标准

1. `greedy_decode` 实现完整，无报错
2. 5 条样本中至少 4 条 ✅（模型已经训练得很好，应该全对）
3. 能回答下面三个问题

---

## 输出问题

**Q1**：推理时为什么还需要因果 mask？训练时加 mask 是为了防止"看到未来"，推理时 Decoder 的输入本来就只有"已生成的部分"，为什么还要加？

**Q2**：`generated[:, 1:]` 去掉了 `<BOS>`。为什么最终输出不需要 `<BOS>`？

**Q3**：Greedy Decoding 有一个著名的缺陷：它不能保证找到全局最优序列。举一个简单的例子说明为什么。

---

准备好后提交代码、终端输出和三个问题的回答。