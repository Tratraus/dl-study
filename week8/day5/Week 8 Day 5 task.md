# Week 8 Day 5：LoRA 原理与 `LoRALinear` 实现

今天进入 Week 8 的第二个核心模块：**LoRA，Low-Rank Adaptation**。

Adapter 是：

```text
在 Transformer 层后面额外插入一个小模块。
```

LoRA 是：

```text
不额外插入完整模块，而是在原始 Linear 层里加入一个低秩的可训练权重更新。
```

---

# 1. 今天的任务目标

你今天要完成三件事：

1. 理解 LoRA 的核心公式：

$$W' = W_0 + \Delta W = W_0 + BA$$

2. 手写一个 `LoRALinear` 类，用它包装原始 `nn.Linear`

3. 验证 LoRA 初始化后，输出与原始 Linear 完全一致

---

# 2. LoRA 的最小必要理论

## 2.1 普通 Linear 层在做什么？

PyTorch 的 `nn.Linear(in_features, out_features)` 本质是：

$$y = xW_0^T + b$$

其中：

```text
x:      (..., in_features)
W0:     (out_features, in_features)
b:      (out_features,)
y:      (..., out_features)
```

比如 ESM-2 8M 中注意力层的 Q/V 投影：

```text
in_features  = 320
out_features = 320
W0 shape     = (320, 320)
参数量        = 102,400 + 320
```

---

## 2.2 全量微调怎么做？

全量微调直接更新原始权重：

$$W_0 \rightarrow W_0 + \Delta W$$

也就是说，训练时原始的 $$W_0$$ 会被修改。

问题是：

```text
如果 W0 是 320×320：
  ΔW 也是 320×320
  需要训练 102,400 个权重

如果模型很大、线性层很多：
  可训练参数会迅速变得巨大
```

---

## 2.3 LoRA 怎么做？

LoRA 认为：

> 微调时真正需要的权重变化 $$\Delta W$$ 不一定是满秩的大矩阵，它可能可以用一个低秩矩阵近似。

所以不直接训练完整的 $$\Delta W$$，而是让：

$$\Delta W = BA$$

其中：

$$A \in \mathbb{R}^{r \times d_{in}}$$

$$B \in \mathbb{R}^{d_{out} \times r}$$

于是：

$$W' = W_0 + BA$$

其中 $$r$$ 是 rank，通常很小，比如 4、8、16。

如果：

```text
d_in  = 320
d_out = 320
r     = 8
```

那么 LoRA 参数量是：

$$r \times d_{in} + d_{out} \times r = 8 \times 320 + 320 \times 8 = 5120$$

而完整 Linear 权重是：

$$320 \times 320 = 102400$$

参数减少：

$$102400 / 5120 = 20$$

也就是说，一个 Linear 层里 LoRA 的参数量只有原始权重的 **1/20**。

---

## 2.4 为什么初始化时输出要和原始 Linear 一致？

LoRA 通常这样初始化：

```text
A：随机初始化
B：全零初始化
```

这样一开始：

$$BA = 0$$

所以：

$$W' = W_0 + 0 = W_0$$

模型刚开始时完全等价于原始预训练模型，不会破坏已有表示。

这和 Adapter 里把 `up_proj` 初始化为零是同一个思想：

```text
训练初期不扰动预训练模型；
训练过程中逐渐学习任务特定变化。
```

---

## 2.5 为什么不能 A 和 B 都初始化为零？

如果：

```text
A = 0
B = 0
```

那么：

$$\Delta W = BA = 0$$

看起来也满足初始化不扰动模型，但反向传播有问题。

对 $$B$$ 的梯度依赖 $$A$$：

$$\frac{\partial L}{\partial B} \propto A$$

对 $$A$$ 的梯度依赖 $$B$$：

$$\frac{\partial L}{\partial A} \propto B$$

如果二者都为 0，那么两个梯度都可能为 0，LoRA 分支无法启动训练。

所以一般做法是：

```text
A 随机
B 置零
```

这样：

```text
初始输出不变；
第一步 B 有梯度；
B 更新后，A 也开始获得有效梯度。
```

这和 Day 2 里 Adapter 的现象很像：第一步主要是上游/下游某一侧参数先动起来。

---

# 3. 今天要实现的 `LoRALinear`

新建文件：

```text
week8/day5/lora_linear.py
```

完成下面代码。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    用 LoRA 包装一个已有的 nn.Linear。

    原始 Linear:
        y = x W0^T + b

    LoRA:
        y = x W0^T + b + scale * x (B A)^T

    其中：
        A: (r, in_features)
        B: (out_features, r)
        BA: (out_features, in_features)
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()

        assert isinstance(base_linear, nn.Linear), "base_linear 必须是 nn.Linear"

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # TODO 1：冻结原始 Linear 参数
        for param in self.base_linear.parameters():
            ...

        # TODO 2：定义 LoRA A 和 B
        # A shape: (rank, in_features)
        # B shape: (out_features, rank)
        self.lora_A = nn.Parameter(...)
        self.lora_B = nn.Parameter(...)

        # TODO 3：初始化
        # A 用 Kaiming uniform 或正态小随机数
        # B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        ...

    def forward(self, x):
        # 原始 frozen linear 输出
        base_out = self.base_linear(x)

        # TODO 4：LoRA 分支
        # 方法一：
        #   lora_hidden = F.linear(x, self.lora_A)
        #   lora_out    = F.linear(lora_hidden, self.lora_B)
        #   return base_out + self.scale * lora_out
        ...

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merge_weights(self):
        """
        推理时可以把 LoRA 权重合并进 base_linear.weight：
            W_merged = W0 + scale * B @ A

        注意：调用后 LoRA 分支理论上不应再重复加，否则会加两次。
        今天只实现计算，不真正修改 forward 逻辑。
        """
        with torch.no_grad():
            delta_w = self.scale * (self.lora_B @ self.lora_A)
            merged_weight = self.base_linear.weight + delta_w
        return merged_weight


def verify_lora_linear():
    torch.manual_seed(42)

    in_features = 320
    out_features = 320
    rank = 8
    alpha = 16

    # 原始 Linear
    base = nn.Linear(in_features, out_features)

    # 保存一份原始输出
    x = torch.randn(4, 100, in_features)
    with torch.no_grad():
        base_out = base(x)

    # 包装成 LoRA Linear
    lora = LoRALinear(base, rank=rank, alpha=alpha)

    # 验证 1：原始权重已冻结
    assert lora.base_linear.weight.requires_grad == False
    assert lora.base_linear.bias.requires_grad == False
    print("✅ 原始 Linear 参数已冻结")

    # 验证 2：输出形状
    out = lora(x)
    assert out.shape == base_out.shape
    print(f"✅ 输出形状正确：{out.shape}")

    # 验证 3：初始化时输出应与原始 Linear 完全一致
    max_diff = (out - base_out).abs().max().item()
    print(f"✅ 初始化最大输出偏差：{max_diff:.2e}")

    # 验证 4：参数量
    trainable = lora.count_trainable_parameters()
    expected = rank * in_features + out_features * rank
    print(f"✅ LoRA 可训练参数量：{trainable}，预期：{expected}")

    # 验证 5：梯度
    loss = out.sum()
    loss.backward()

    print(f"✅ base weight grad: {lora.base_linear.weight.grad}")
    print(f"✅ LoRA A grad norm: {lora.lora_A.grad.norm():.4f}")
    print(f"✅ LoRA B grad norm: {lora.lora_B.grad.norm():.4f}")

    # 验证 6：merge 权重形状
    merged = lora.merge_weights()
    assert merged.shape == lora.base_linear.weight.shape
    print(f"✅ merged weight shape 正确：{merged.shape}")

    print("\n🎉 LoRALinear 所有验证完成！")


if __name__ == "__main__":
    verify_lora_linear()
```

---

# 4. 你需要补全的关键代码

为了避免你卡在语法上，这里给出关键提示。

## TODO 1：冻结原始 Linear

```python
for param in self.base_linear.parameters():
    param.requires_grad = False
```

---

## TODO 2：定义 A/B

```python
self.lora_A = nn.Parameter(torch.empty(rank, in_features))
self.lora_B = nn.Parameter(torch.empty(out_features, rank))
```

---

## TODO 3：初始化 B

```python
nn.init.zeros_(self.lora_B)
```

---

## TODO 4：forward

```python
lora_hidden = F.linear(x, self.lora_A)
lora_out = F.linear(lora_hidden, self.lora_B)
return base_out + self.scale * lora_out
```

解释一下形状：

```text
x:            (batch, seq_len, 320)

self.lora_A:  (rank, 320)
F.linear(x, A):
  输出 shape = (batch, seq_len, rank)

self.lora_B:  (320, rank)
F.linear(hidden, B):
  输出 shape = (batch, seq_len, 320)

最后和 base_out 相加。
```

---

# 5. 预期输出

运行后你应该看到类似：

```text
✅ 原始 Linear 参数已冻结
✅ 输出形状正确：torch.Size([4, 100, 320])
✅ 初始化最大输出偏差：0.00e+00
✅ LoRA 可训练参数量：5120，预期：5120
✅ base weight grad: None
✅ LoRA A grad norm: 0.0000
✅ LoRA B grad norm: 1234.5678
✅ merged weight shape 正确：torch.Size([320, 320])

🎉 LoRALinear 所有验证完成！
```

注意：

```text
LoRA A grad norm 初始可能是 0
LoRA B grad norm 非 0
```

这是因为：

```text
B 初始化为 0
所以梯度一开始无法通过 B 传回 A
但 B 自己能收到梯度
B 更新一步后，A 就会有梯度
```

和 Day 2 的 Adapter 现象是同构的。

---

# 6. 今日输出问题

完成代码和运行后，请回答：

## Q1

为什么 LoRA 要把 `B` 初始化为零，而不是把 `A` 和 `B` 都随机初始化？

---

## Q2

如果 `rank=8, in_features=320, out_features=320`，LoRA 参数量是多少？相比原始 Linear 的权重参数量减少了多少倍？

---

## Q3

`merge_weights()` 的作用是什么？为什么 LoRA 在推理时可以做到几乎不增加延迟？

---

## Q4

你观察到的梯度现象是什么？`lora_A.grad` 和 `lora_B.grad` 谁一开始为 0？为什么？

---

完成后把代码、运行结果、四个问题的回答发给我，我来验收。