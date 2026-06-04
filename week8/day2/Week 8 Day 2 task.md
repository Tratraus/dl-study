# Week 8 Day 2：Adapter 模块实现

## 今天的任务目标

**手写一个 `AdapterLayer` 类，理解其"降维→激活→升维→残差"的结构，并验证近恒等初始化的效果。**

---

## 最小必要理论

### 1. Adapter 的结构

```
输入 x (shape: batch, seq_len, d_model)
 │
 ├─── 残差分支（直接保留）
 │
 └─── Adapter 分支：
 Linear(d_model → bottleneck) ← 降维
 激活函数（GELU）
 Linear(bottleneck → d_model) ← 升维
 │
 └─── 两路相加 → 输出
```

用公式写：

$$\text{output} = x + W_{up} \cdot \text{GELU}(W_{down} \cdot x)$$

其中 $$W_{down} \in \mathbb{R}^{b \times d}$$，$$W_{up} \in \mathbb{R}^{d \times b}$$，$$b$$ 是 bottleneck 维度。

---

### 2. 近恒等初始化（Near-Identity Initialization）

这是 Adapter 能稳定训练的关键：

```
如果 W_up 初始化为全零：
 Adapter 分支输出 = W_up · GELU(W_down · x) = 0
 总输出 = x + 0 = x

效果：
 训练刚开始时，Adapter 对模型行为没有任何影响
 等价于"插入了一个什么都不做的模块"
 随着训练，Adapter 才逐渐学习有用的变换

对比如果不这样初始化：
 随机初始化的 W_up 会立刻破坏预训练模型的输出分布
 导致训练初期不稳定，甚至发散
```

**实现方式**：把 `W_up`（升维层）的权重初始化为零即可。

---

### 3. 为什么用 GELU 而不是 ReLU

```
GELU 在 x=0 附近是平滑的，梯度不会突然截断
Transformer 系列模型（包括 ESM-2）普遍使用 GELU
保持一致性，避免引入额外的分布差异
```

---

## 代码任务

新建文件：`week8/day2/adapter.py`

请完成以下骨架中的 TODO：

```python
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
 def __init__(self, d_model: int, bottleneck: int):
 super().__init__()
 # TODO 1：定义降维线性层
 # 输入维度 d_model，输出维度 bottleneck
 self.down_proj = ...

 # TODO 2：定义激活函数（使用 GELU）
 self.act = ...

 # TODO 3：定义升维线性层
 # 输入维度 bottleneck，输出维度 d_model
 self.up_proj = ...

 # TODO 4：近恒等初始化
 # 将 up_proj 的权重初始化为全零
 # 提示：nn.init.zeros_(tensor)
 ...

 def forward(self, x: torch.Tensor) -> torch.Tensor:
 # TODO 5：实现前向传播
 # 注意：需要残差连接
 ...

 def count_parameters(self) -> int:
 # TODO 6：返回该模块的可训练参数数量
 ...


# ════════════════════════════════════════════════════════════
# 验证代码（不需要修改，完成上面的 TODO 后直接运行）
# ════════════════════════════════════════════════════════════

def verify_adapter():
 d_model = 320
 bottleneck = 64
 batch_size = 4
 seq_len = 100

 adapter = AdapterLayer(d_model, bottleneck)

 x = torch.randn(batch_size, seq_len, d_model)

 # 验证 1：输出形状是否正确
 out = adapter(x)
 assert out.shape == x.shape, f"形状错误：{out.shape} != {x.shape}"
 print(f"✅ 输出形状正确：{out.shape}")

 # 验证 2：近恒等初始化——初始输出应与输入几乎相同
 max_diff = (out - x).abs().max().item()
 print(f"✅ 初始最大偏差：{max_diff:.2e}（应接近 0）")

 # 验证 3：参数量
 params = adapter.count_parameters()
 expected = 2 * d_model * bottleneck + d_model + bottleneck
 print(f"✅ 可训练参数量：{params}（预期约 {expected}）")

 # 验证 4：梯度能否正常流动
 loss = out.sum()
 loss.backward()
 grad_down = adapter.down_proj.weight.grad
 grad_up = adapter.up_proj.weight.grad
 assert grad_down is not None, "down_proj 没有梯度！"
 assert grad_up is not None, "up_proj 没有梯度！"
 print(f"✅ 梯度正常：down_proj grad norm = {grad_down.norm():.4f}")
 print(f"✅ 梯度正常：up_proj grad norm = {grad_up.norm():.4f}")

 print("\n🎉 所有验证通过！")

if __name__ == "__main__":
 verify_adapter()
```

---

## 完成标准

运行 `verify_adapter()`，四项验证全部通过：

```
✅ 输出形状正确：torch.Size([4, 100, 320])
✅ 初始最大偏差：0.00e+00（应接近 0）
✅ 可训练参数量：41344（预期约 41344）
✅ 梯度正常：down_proj grad norm = ...
✅ 梯度正常：up_proj grad norm = ...

🎉 所有验证通过！
```

---

## 输出问题

**Q1**：验证 2 检查"初始最大偏差接近 0"，你的输出是多少？为什么不是严格的 0？（提示：`down_proj` 的初始化是什么？）

**Q2**：如果把 `TODO 4` 的初始化去掉（让 `up_proj` 保持默认的 Kaiming 初始化），验证 2 的结果会变成什么？这对训练有什么影响？

**Q3**：`count_parameters` 里，`bias` 也是参数，你有没有把它算进去？`bias` 的维度是多少？
