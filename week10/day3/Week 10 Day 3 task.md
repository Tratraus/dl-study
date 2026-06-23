# Week 10 Day 3：MLM 预训练循环

## 最小必要理论（10 分钟）

### 1. 今天要拼的东西

Day 1 有数据，Day 2 有模型，今天把它们连起来：

```text
make_mlm_batch()
      ↓
masked_src  ──→  ProteinBERT  ──→  logits (batch, seq_len, vocab_size)
labels      ──→  cross_entropy(logits, labels, ignore_index=-100)  ──→  loss
                      ↓
               loss.backward()
               optimizer.step()
```

这个循环和 Week 1 的 MLP 训练循环本质上没有区别，只是 loss 的计算方式多了一个 `ignore_index`。

---

### 2. MLM Loss 的计算细节

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (batch * seq_len, vocab_size)
    labels.view(-1),               # (batch * seq_len,)
    ignore_index=-100
)
```

为什么要 `.view(-1, vocab_size)`？

`cross_entropy` 期望输入是 `(N, C)` 的格式，N 是样本数，C 是类别数。我们把 `(batch, seq_len, vocab_size)` 展平成 `(batch * seq_len, vocab_size)`，每个位置都当成一个独立的分类样本。

`ignore_index=-100` 的效果：

```text
labels = [12, -100, -100, 10, -100, ...]
                ↑
         这些位置不参与 loss 计算
```

只有 label ≠ -100 的位置会贡献 loss，其余位置的 loss 被自动置零。

---

### 3. 理论初始 Loss

模型刚初始化时，对每个 token 的预测是均匀分布（接近均匀）。

理论初始 loss：

$$\mathcal{L}_0 = \log(\text{vocab\_size}) = \log(25) \approx 3.22$$

如果你的初始 loss 远高于 3.22（比如 > 5），说明模型初始化有问题。

如果初始 loss 接近 3.22，说明模型正常。

---

### 4. 收敛的判断标准

| loss 范围 | 含义 |
|---|---|
| ~3.22 | 模型在随机猜，没有学到任何信息 |
| 2.0 ~ 2.5 | 模型开始学到氨基酸的频率分布 |
| 1.5 ~ 2.0 | 模型开始利用上下文信息 |
| < 1.0 | 模型对合成数据过拟合（正常，合成数据简单） |

今天的目标：**loss 从 ~3.22 下降到 2.0 以下**。

---

### 5. 一个需要注意的细节：训练模式 vs 推理模式

`Dropout` 和 `BatchNorm` 在训练和推理时行为不同：

```python
model.train()   # 训练时：Dropout 生效
model.eval()    # 推理时：Dropout 关闭
```

训练循环里要确保 `model.train()` 在前，保存 checkpoint 前不需要切换。

---

## 代码任务

新建文件：`week10/day3/train_mlm.py`

```python
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))

from mlm_data import (
    VOCAB_SIZE, PAD, IGNORE_INDEX,
    make_mlm_batch
)
from protein_bert import ProteinBERT


# ── TODO 1：实现单步训练函数 ──────────────────────────────────
def train_step(
    model: ProteinBERT,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> float:
    """
    执行一步 MLM 训练。

    步骤：
      1. 生成一批 MLM 数据（make_mlm_batch）
      2. 构造 src_key_padding_mask
      3. 前向传播，得到 logits
      4. 计算 MLM loss（cross_entropy，ignore_index=IGNORE_INDEX）
      5. 反向传播 + optimizer.step() + 梯度清零

    返回：
      loss.item()（float）

    注意：
      - logits shape: (batch, seq_len, vocab_size)
      - labels shape: (batch, seq_len)
      - cross_entropy 需要 (N, C) 格式，记得 .view()
    """
    ...


# ── TODO 2：实现训练主循环 ────────────────────────────────────
def train(
    num_steps:  int   = 1000,
    batch_size: int   = 32,
    seq_len:    int   = 50,
    lr:         float = 1e-3,
    log_every:  int   = 100,
    save_path:  str   = 'protein_bert_mlm.pt',
):
    """
    MLM 预训练主循环。

    步骤：
      1. 初始化设备、模型、optimizer（用 Adam）
      2. 循环 num_steps 步，每步调用 train_step
      3. 每 log_every 步打印当前 step 和 loss
      4. 训练结束后保存 checkpoint

    checkpoint 格式（用 torch.save 保存 dict）：
      {
        'model_state_dict': model.state_dict(),
        'step': num_steps,
        'final_loss': last_loss,
      }

    打印格式示例：
      Step    0 | loss: 3.2154
      Step  100 | loss: 2.8732
      Step  200 | loss: 2.4501
      ...
      Step 1000 | loss: 1.3204
      ✅ 训练完成，checkpoint 已保存至 protein_bert_mlm.pt
    """
    ...


# ── TODO 3：loss 曲线可视化 ───────────────────────────────────
def plot_loss_curve(loss_history: list[float], save_path: str = 'loss_curve.png'):
    """
    绘制 loss 曲线并保存。

    要求：
      - x 轴：训练步数
      - y 轴：loss
      - 标题：MLM Pre-training Loss
      - 加一条水平虚线标注 log(25) ≈ 3.22（理论初始 loss）
      - 保存为 loss_curve.png
    """
    import matplotlib.pyplot as plt
    import math

    fig, ax = plt.subplots(figsize=(8, 4))

    # TODO：
    # 1. 画 loss 曲线
    # 2. 画 y = log(25) 的水平虚线，标注 "random baseline (log 25)"
    # 3. 设置标题、x/y 轴标签
    # 4. 保存图片
    ...


if __name__ == "__main__":
    # 先跑训练，收集 loss history
    # 再画曲线

    # 提示：可以修改 train() 让它返回 loss_history
    train()
```

---

## 完成标准

1. `train_step` 能正确计算 MLM loss 并完成一步梯度更新
2. 初始 loss 在 **3.0 ~ 3.5** 之间
3. 1000 步后 loss 降到 **2.0 以下**
4. checkpoint 保存成功（`protein_bert_mlm.pt`）
5. `loss_curve.png` 生成，曲线可见下降趋势，包含 baseline 虚线

---

## 输出问题

**Q1**：`logits.view(-1, vocab_size)` 和 `labels.view(-1)` 做了什么？为什么 cross_entropy 需要这个格式？

**Q2**：你的初始 loss 是多少？和理论值 $$\log(25) \approx 3.22$$ 相比如何？

**Q3**：训练 1000 步后 loss 降到了多少？loss 曲线的下降趋势是什么样的（匀速下降、先快后慢、还是其他）？

---

准备好后提交代码、终端输出、loss 曲线图和三个问题的回答。