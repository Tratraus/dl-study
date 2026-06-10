# Week 8 Day 5 复盘：LoRA 原理与 LoRALinear 实现

## 代码结构

```
lora_linear.py
├── LoRALinear(nn.Module)
│   ├── __init__()         # 冻结 base_linear + 定义 A/B + 初始化
│   ├── forward()          # base_out + scale * B(Ax)
│   ├── count_trainable_parameters()
│   └── merge_weights()    # W_merged = W₀ + scale * B @ A
└── verify_lora_linear()   # 六项验证
```

---

## 数据流 & 形状变化

```
输入 x: (batch=4, seq_len=100, in_features=320)
         │
         ├── 原始 Linear 分支（冻结）
         │   base_out = x @ W₀^T + b
         │   (4, 100, 320)
         │
         └── LoRA 分支（可训练）
              lora_hidden = F.linear(x, A)
              │  A: (8, 320) → 输出 (4, 100, 8)     ← 降维到 rank
              ▼
              lora_out = F.linear(lora_hidden, B)
              │  B: (320, 8) → 输出 (4, 100, 320)    ← 升回原维度
              ▼
              × scale (alpha/rank = 16/8 = 2.0)
              ▼
         base_out + scale * lora_out → (4, 100, 320)
```

---

## 关键知识点

### 1. LoRA 的核心公式

```
W' = W₀ + ΔW = W₀ + scale · B @ A
```

- A: (r, d_in) — 降维矩阵
- B: (d_out, r) — 升维矩阵
- scale = alpha / rank — 缩放因子，控制 LoRA 的"学习强度"

### 2. 初始化策略

```python
A: Kaiming uniform (随机非零)
B: zeros
```

- B=0 → BA=0 → 初始输出 = base_out，不扰动预训练模型
- A 非零 → 梯度链能启动（B 更新后 A 才有梯度）
- 如果 A 和 B 都为零 → 两个矩阵都收不到梯度，无法训练

### 3. 推理零开销：merge_weights()

```python
delta_w = scale * (B @ A)          # (320, 320)
merged_weight = W₀ + delta_w       # 合并后只需一次矩阵乘法
```

推理时不需要额外的 LoRA 分支计算，延迟与原始 Linear 完全一致。

### 4. 为什么是 B=0、A=Kaiming，而不是反过来？

数学上两种方案都能实现 warm-start（BA=0），但训练动态不同：

| | A=Kaiming, B=0（标准） | A=0, B=Kaiming（反过来） |
|---|---|---|
| 第一步谁动 | B 先动 | A 先动 |
| 第一步 A 学到什么 | 不更新 | 在随机 B 方向上学习（效率低） |
| 第一步 B 学到什么 | 用 A 的随机方向作为锚点学习输出映射 | 不更新 |

标准做法让**输出侧先学习**，输入侧的随机初始化提供固定的低维空间作为起点，训练动态更干净。

### 5. LoRA vs Adapter 的结构区别

| | Adapter | LoRA |
|---|---|---|
| 插入方式 | 层间额外模块 | 直接修改权重矩阵 |
| 参数位置 | down_proj + up_proj | A + B |
| 推理开销 | 始终有额外计算 | 可 merge 为零开销 |
| warm-start | up_proj 置零 | B 置零 |

---

## 踩坑 & 易错点

| 易错点 | 说明 |
|--------|------|
| A 和 B 都置零 | 梯度链断裂，两个矩阵都无法更新 |
| scale 忘记 | alpha/rank 控制 LoRA 的影响强度，不加 scale 会过强 |
| F.linear 参数顺序 | `F.linear(x, weight)` 中 weight 的 shape 是 (out, in)，和 nn.Linear 一致 |
| merge 后重复计算 | merge 后 forward 还会再加一次 LoRA 分支，推理时需去掉 |
| A/B 都置零 | 梯度链断裂，两个矩阵都无法更新；反过来(A=0,B=Kaiming)虽能训练但第一步效率低 |

---

## 输出问题回顾

### Q1：为什么 B 初始化为零而不是 A 和 B 都随机？

B=0 保证初始输出不变；A 非零保证梯度链能启动。都置零会导致两个矩阵都收不到梯度。

### Q2：参数量

- LoRA: 8×320 + 320×8 = 5,120
- 原始 Linear: 320×320 = 102,400
- 减少 20 倍

### Q3：merge_weights 的作用

将 B@A 合并进 W₀，推理时只需一次矩阵乘法，不增加延迟。

### Q4：梯度现象

- `lora_A.grad.norm() = 0.0000` — 严格为零，因为 B=0 阻断了梯度回传
- `lora_B.grad.norm() = 1179.7676` — B 直接收到梯度
- 训练第一步：只有 B 更新 → B 非零 → 第二步 A 开始有梯度

---

## 与 Adapter 的 warm-start 对比

| 维度 | Adapter | LoRA |
|------|---------|------|
| 零初始化位置 | up_proj (升维层) | B (输出侧矩阵) |
| 非零初始化位置 | down_proj (Kaiming) | A (Kaiming) |
| 初始效果 | x + 0 = x | W₀ + 0 = W₀ |
| 梯度启动 | up_proj 先动 → down_proj 后动 | B 先动 → A 后动 |
| 本质 | 同一种思想：输出侧置零，输入侧保留梯度链 |

---

## 总结

Day 5 实现了 LoRA 的核心组件。LoRA 和 Adapter 的 warm-start 是同构的——都是"输出侧置零"保证初始不扰动模型，"输入侧非零"保证梯度链能启动。关键区别在于 LoRA 通过矩阵乘法直接修改权重，推理时可 merge 为零开销，而 Adapter 作为独立模块始终有额外计算。
