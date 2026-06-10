# Week 8 Day 2 复盘：Adapter 模块实现

## 代码结构

```
adapter.py
├── AdapterLayer(nn.Module)
│   ├── __init__()     # down_proj + GELU + up_proj + 近恒等初始化
│   ├── forward()      # x + up_proj(GELU(down_proj(x)))  残差连接
│   └── count_parameters()
└── verify_adapter()   # 四项验证：形状 / 近恒等 / 参数量 / 梯度
```

---

## 数据流 & 形状变化

```
输入 x: (batch=4, seq_len=100, d_model=320)
         │
         ├── 残差分支（直接保留 x）
         │
         └── Adapter 分支：
              down_proj: Linear(320 → 64)
              │  (4, 100, 64)
              ▼
              GELU
              │  (4, 100, 64)
              ▼
              up_proj: Linear(64 → 320)
              │  (4, 100, 320)
              ▼
         x + adapter_out → (4, 100, 320)
```

---

## 关键知识点

### 1. 近恒等初始化（Near-Identity Initialization）

```python
nn.init.zeros_(self.up_proj.weight)
nn.init.zeros_(self.up_proj.bias)
```

- `up_proj` 权重和 bias 全零 → Adapter 分支输出恒为 0
- 训练初期：`output = x + 0 = x`，Adapter 等价于不存在
- 意义：不破坏预训练模型的输出分布，训练稳定

### 2. 参数量计算

```
down_proj: weight (320×64) + bias (64)   = 20,480 + 64 = 20,544
up_proj:   weight (64×320) + bias (320)  = 20,480 + 320 = 20,800
单层 Adapter 总计: 41,344
两层 + classifier(963): 83,651
```

### 3. 为什么用 GELU 而非 ReLU

- GELU 在 x=0 附近平滑，梯度不会突然截断
- 与 ESM-2 内部激活函数保持一致，避免引入分布差异

---

## 踩坑 & 易错点

| 易错点 | 说明 |
|--------|------|
| 忘记近恒等初始化 | `up_proj` 用默认 Kaiming 初始化 → 初始输出不为零 → 破坏预训练分布 |
| bias 维度容易混淆 | `down_proj` bias 是 bottleneck(64)，`up_proj` bias 是 d_model(320) |

---

## 输出问题回顾

### Q1：初始最大偏差

输出 `0.00e+00`（严格为零）。因为 `up_proj` 的 weight 和 bias 都被 `zeros_` 初始化，Adapter 分支输出恒为零。

### Q2：去掉近恒等初始化

Kaiming 初始化会让 `up_proj` 输出非零随机值，初始偏差会显著增大，训练初期可能不稳定甚至发散。

### Q3：bias 的维度

- `down_proj` bias: bottleneck = 64
- `up_proj` bias: d_model = 320
- 都已包含在 `count_parameters()` 中

---

## 总结

Day 2 实现了 Adapter 的核心模块。近恒等初始化是 Adapter 能稳定训练的关键——它让新插入的模块在训练初期"隐形"，不干扰预训练模型的输出。这个设计思想贯穿所有 PEFT 方法。
