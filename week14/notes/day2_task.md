# Week 14 Day 2：BCEWithLogitsLoss 与多标签损失函数设计

## 1. 为什么多标签任务不能用 Softmax + CrossEntropyLoss

`Softmax` 的数学定义：

$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

分母是对所有K个类别logits求和，这个结构**天然假设类别之间互斥**——即所有类别的概率加起来必须等于1：

$\sum_{i=1}^{K} \text{softmax}(z_i) = 1$

但你的数据现实是：一个蛋白平均有2.25个Top-50标签同时为真。如果用softmax，模型会被迫在"金属离子结合"和"ATP结合"这两个本该同时成立的标签之间**互相抢概率**，这是对任务本质的错误建模。

多标签任务的正确假设是：**每个标签的存在与否是一个独立的二分类问题**，K个标签之间没有"归一化到1"的约束。

## 2. Sigmoid + BCE 的数学含义

对第 $i$ 个标签，模型输出原始logit $z_i$（未经过任何激活函数），通过sigmoid转成概率：

$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$

每个 $p_i$ 独立地表示"这个标签成立的概率"，不需要和其他标签的概率相加为1。

对单个标签的二元交叉熵损失：

$\ell_i = -\left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$

其中 $y_i \in \{0, 1\}$ 是真实标签。整个样本的总损失是K个标签损失的求和（或平均）：

$L = \sum_{i=1}^{K} \ell_i$

这正是"每个标签独立二分类"思想的直接体现——50个标签，就是50个并行的逻辑回归。

## 3. 为什么 `BCEWithLogitsLoss` 比 `Sigmoid()+BCELoss()` 更稳定

这是Day2最容易被忽略但最重要的工程细节。

### 3.1 问题根源：sigmoid的浮点下溢

当 $z_i$ 是一个绝对值很大的负数（比如 $z_i = -100$）时：

$\sigma(-100) = \frac{1}{1+e^{100}}$

```python

import numpy as np

def manual_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def naive_bce(z, y, eps=0.0):
    """先手动sigmoid，再算log(p)，模拟 Sigmoid()+BCELoss() 的组合方式"""
    p = manual_sigmoid(z)
    p = np.clip(p, eps, 1 - eps) if eps > 0 else p
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def stable_bce_with_logits(z, y):
    """
    这是 BCEWithLogitsLoss 内部真正使用的数值稳定公式:
    L = max(z,0) - z*y + log(1 + exp(-|z|))
    等价于原始BCE，但避免了exp(z)在z很大时溢出、以及log(0)的问题
    """
    return np.maximum(z, 0) - z * y + np.log(1 + np.exp(-np.abs(z)))

# 测试极端情况：logit绝对值很大
test_logits = np.array([-100.0, -30.0, -10.0, 0.0, 10.0, 30.0, 100.0])
y_true = 1.0  # 假设真实标签为1，此时 z=-100 意味着模型"极度自信地"给出错误预测

print(f"{'logit z':>10} | {'naive sigmoid+BCE (无clip)':>28} | {'BCEWithLogits(稳定公式)':>22}")
print("-" * 68)
for z in test_logits:
    naive_result = naive_bce(z, y_true, eps=0.0)
    stable_result = stable_bce_with_logits(z, y_true)
    print(f"{z:>10.1f} | {naive_result:>28} | {stable_result:>22.4f}")

```
```
结果
   logit z |    naive sigmoid+BCE (无clip) |    BCEWithLogits(稳定公式)
--------------------------------------------------------------------
    -100.0 |                        100.0 |               100.0000
     -30.0 |           30.000000000000092 |                30.0000
     -10.0 |           10.000045398899216 |                10.0000
       0.0 |           0.6931471805599453 |                 0.6931
      10.0 |        4.539889921682063e-05 |                 0.0000
      30.0 |        9.348077867344255e-14 |                 0.0000
     100.0 |                          nan |                 0.0000
/tmp/ipykernel_504/2312576992.py:14: RuntimeWarning: divide by zero encountered in log
  return -(y * np.log(p) + (1 - y) * np.log(1 - p))
/tmp/ipykernel_504/2312576992.py:14: RuntimeWarning: invalid value encountered in scalar multiply
  return -(y * np.log(p) + (1 - y) * np.log(1 - p))
```
看到了吗？在 $z=100$ 这一行，"先sigmoid再取log"的写法直接产出了 **`nan`**，还伴随着 `RuntimeWarning: divide by zero`。而右侧稳定公式全程正常。

### 3.2 问题出在哪里

当 $y=1$、$z=100$ 时（模型极度自信地预测"应该是1"，恰好预测对了方向），朴素实现的计算路径是：

1. 先算 $\sigma(100) = 1/(1+e^{-100})$，由于 $e^{-100}$ 在浮点下溢为0，结果 $p$ 被四舍五入成恰好 **1.0**
2. 再算 $(1-y)\log(1-p) = 0 \times \log(0)$
3. $\log(0) = -\infty$，而 $0 \times (-\infty) = \text{nan}$（不是0，是未定义）

即使模型这次预测方向是对的，梯度计算依然会因为这个 `nan` 而彻底崩溃，反向传播会污染整个batch的梯度。

### 3.3 数值稳定的解决方案

PyTorch的 `BCEWithLogitsLoss` 内部使用的是这个等价但数值安全的公式：

$L(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$

这个公式的关键技巧在于：**指数运算的输入永远是 $-|z|$，也就是永远是负数或0**，所以 $e^{-|z|}$ 永远落在 $(0, 1]$ 区间，绝不会溢出成 `inf`，也绝不会导致后续的 `log(0)`。

从上面的验证数据能看到，稳定公式在整个范围 $z \in [-100, 100]$ 内都给出了合理的输出，没有一次NaN或警告。

**工程结论：永远使用 `nn.BCEWithLogitsLoss()`，永远不要手动写 `Sigmoid() + BCELoss()` 这种组合。** 模型最后一层不要加sigmoid，直接输出logits交给损失函数处理。

## 4. 从Top-50标签频次构造 `pos_weight`

`BCEWithLogitsLoss` 支持传入 `pos_weight` 参数，公式变为：

$\ell_i = -\left[ w_i \cdot y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$

其中 $w_i$ 是第 $i$ 个标签的正样本权重，常见取法：

$w_i = \frac{N - n_{pos,i}}{n_{pos,i}}$

即"负样本数/正样本数"。这样高频标签的权重小（模型本来就见得多，不需要额外加权），低频标签的权重大（强迫模型更重视这些稀少的正例，否则模型会学到"永远预测0"这个懒惰但loss很低的策略）。

刚才代码验证了你真实数据算出来的结果：

| 标签 | 正样本数 | pos_weight |
|---|---:|---:|
| GO:0042802（最高频） | 238 | **4.10** |
| GO:0051087（最低频） | 21 | **56.76** |

不平衡跨度达到 **13.86倍**。这说明如果不加pos_weight，模型很可能会完全忽略像GO:0051087这种低频标签——因为把它全部预测为0，loss只会有轻微上升，但梯度信号极弱。

## 5. 今日代码任务：`scripts/day2_loss_setup.py`

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

def compute_pos_weight(label_counts: dict, N: int, cap: float = None) -> torch.Tensor:
    """
    根据Top-K标签的正样本频次，计算BCEWithLogitsLoss所需的pos_weight向量。

    Args:
        label_counts: {label_name: 正样本出现次数}，按固定顺序排列
        N: 总样本数（过滤后的样本数，你的场景中是1213）
        cap: 可选，对pos_weight设置上限，防止极端稀有标签权重过大导致训练不稳定
    """
    weights = []
    for label, n_pos in label_counts.items():
        n_neg = N - n_pos
        w = n_neg / n_pos
        if cap is not None:
            w = min(w, cap)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(pos_weight: torch.Tensor) -> nn.Module:
    """
    正确的多标签损失函数构造方式：
    - 不在模型最后一层加sigmoid
    - 直接用BCEWithLogitsLoss处理原始logits
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


if __name__ == "__main__":
    # 用你Day1的真实数据做sanity check
    N = 1213
    top50_counts = {
        "GO:0042802": 238, "GO:0051087": 21,  # 示例：取最高频和最低频做对比
    }
    pos_weight = compute_pos_weight(top50_counts, N, cap=50.0)
    print("pos_weight:", pos_weight)

    loss_fn = build_loss_fn(pos_weight)

    # 模拟一个batch：batch_size=4, num_labels=2
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])

    loss = loss_fn(logits, targets)
    loss.backward()

    print("loss:", loss.item())
    print("grad存在且非NaN:", not torch.isnan(logits.grad).any().item())
```

## 6. 单元测试要求（`tests/test_day2_loss.py`）

请在本地补充以下断言：

```python
def test_pos_weight_shape():
    # pos_weight的长度必须等于标签数K
    assert pos_weight.shape[0] == K

def test_loss_no_nan_extreme_logits():
    # 用极端logits（例如±100）测试BCEWithLogitsLoss不产生NaN
    extreme_logits = torch.tensor([[100.0, -100.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0]])
    loss = loss_fn(extreme_logits, targets)
    assert not torch.isnan(loss).any()
    loss.backward()
    assert not torch.isnan(extreme_logits.grad).any()

def test_pos_weight_direction():
    # 低频标签的pos_weight必须大于高频标签
    assert pos_weight[low_freq_idx] > pos_weight[high_freq_idx]
```

---

## 今日反思问题

1. **cap的取舍**：上面代码里我给 `pos_weight` 设置了 `cap=50.0` 上限。如果你的最低频标签算出pos_weight是56.76，被cap截断到50后，这个标签的训练会发生什么变化？这种截断是否会引入新的偏差？
2. **macro vs micro**：如果你后续用 `pos_weight` 训练后发现，高频标签（如GO:0042802）的F1很高，但低频标签（如GO:0051087）F1接近0，这说明什么问题？pos_weight调大就能解决吗？
3. **Loss不是全部**：假设两组超参数训出的validation loss完全相同，但一组的低频标签召回率明显更高。这时候单看loss选模型，会不会做出错误决策？这对你日后设计早停（early stopping）指标有什么启发？

---

## 下一步行动

1. 本地运行 `day2_loss_setup.py`，确认没有NaN、梯度正常。
2. 用你真实的Top-50全部50个标签频次（不是我示例里的2个）计算出完整的 `pos_weight` 向量，跑一次sanity check。
3. 完成后回复：**"Day2 loss设计完成，准备进入Day3"**，我们将进入 **多标签分类模型架构设计**（这时候我们要具体讨论：输入是序列还是特征？用什么backbone？输出层怎么接这50维logits？）。