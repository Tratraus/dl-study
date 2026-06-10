# Week 8 Day 3 复盘：将 Adapter 插入 ESM-2

## 代码结构

```
esm_with_adapter.py
├── AdapterLayer          # 复用 Day 2 的 Adapter 模块
├── ESMWithAdapter        # 主模型：ESM-2 + hook + classifier
├── generate_data()       # 合成数据（复用 Week 7）
├── ProteinDataset        # Dataset 封装
├── train_epoch()         # 训练一个 epoch
├── eval_epoch()          # 验证
└── main()                # 入口：数据 → 模型 → 训练 → 验证
```

---

## 数据流 & 形状变化

```
输入: seq (str), 长度 L ∈ [50, 150]
         │
         ▼
batch_converter
         │  tokens: (batch, L+2)        ← 自动加 <cls>/<eos>，padding 到 batch 内最长
         ▼
ESM-2 Embedding
         │  (batch, L+2, 320)
         ▼
Layer 0-3 (直接透传，无 Adapter)
         │
         ▼
Layer 4 → forward hook 触发
         │  output[0]: (batch, L+2, 320)
         │  → Adapter 0: down(320→64) → GELU → up(64→320) + 残差
         │  → (batch, L+2, 320)
         ▼
Layer 5 → forward hook 触发
         │  → Adapter 1 (同上)
         │  → (batch, L+2, 320)
         ▼
classifier (Linear 320→3)
         │  (batch, L+2, 3)
         ▼
切片 [:, 1:-1, :]
         │  (batch, L, 3)               ← 去掉 <cls>/<eos>，对齐标签
         ▼
CrossEntropyLoss
         │  logits: (batch×L, 3)
         │  labels: (batch×L)           ← padding 位为 -100，被 ignore
         ▼
loss.backward() → 只更新 Adapter + classifier
```

---

## 关键知识点

### 1. `register_forward_hook` — 不改源码的插桩方式

```python
def hook(module, input, output, adapter=adapter):
    hidden = adapter(output[0])        # output 是元组，[0] 才是 hidden_states
    return (hidden,) + output[1:]      # 替换 hidden_states，保留其余

layer.register_forward_hook(hook)
```

- hook 的返回值**替换**该模块的输出
- ESM-2 TransformerLayer 输出是元组，需要拆开处理

### 2. 闭包捕获的坑

```python
# ❌ 错误写法：循环变量延迟绑定
for i, adapter in enumerate(self.adapters):
    def hook(module, input, output):
        hidden = self.adapters[i](output[0])  # i 最终 = 2，所有 hook 都指向同一个 adapter
    layer.register_forward_hook(hook)

# ✅ 正确写法：用默认参数捕获当前值
for i, adapter in enumerate(self.adapters):
    def hook(module, input, output, adapter=adapter):  # adapter 在定义时绑定
        hidden = adapter(output[0])
    layer.register_forward_hook(hook)
```

Python 闭包捕获的是**变量引用**，不是值。循环结束后 `i` 的值是最后一次迭代的值。

### 3. `nn.ModuleList` vs 普通 `list`

```python
# ❌ 普通 list：参数不会被注册到模型，optimizer 找不到
self.adapters = [AdapterLayer(320, 64) for _ in range(2)]

# ✅ ModuleList：参数自动注册，model.parameters() 能遍历到
self.adapters = nn.ModuleList([AdapterLayer(320, 64) for _ in range(2)])
```

### 4. `state_dict()` 返回的是引用，不是拷贝

```python
# ❌ 潜在风险：weight_before 和 weight_after 指向同一块内存
weight_before = model.esm.state_dict()   # 引用
# ... 训练 ...
weight_after = model.esm.state_dict()    # 同一块内存
# 如果 ESM 参数被意外修改，两者同时变，torch.equal 永远 True

# ✅ 正确做法
import copy
weight_before = copy.deepcopy(model.esm.state_dict())

# ✅ 更轻量：只检查一层
weight_before = model.esm.layers[-1].self_attn.q_proj.weight.data.clone()
```

当前代码碰巧能通过验证，是因为 ESM 参数 `requires_grad=False` 没被修改，不是验证逻辑本身正确。

---

## 踩坑 & 易错点

| 易错点 | 说明 |
|--------|------|
| `output` 不是 tensor | ESM-2 每层输出是元组，`output[0]` 才是 hidden_states |
| 闭包延迟绑定 | 循环里定义函数要注意变量捕获方式 |
| `list` vs `ModuleList` | 普通 list 的参数不会被 optimizer 看到 |
| `state_dict()` 是引用 | 验证权重变化需要 `.clone()` 或 `deepcopy()` |
| 标签对齐 | logits 去掉首尾 `[:, 1:-1, :]` 后才能和标签对齐 |

---

## 输出问题回顾

### Q1：可训练参数量

| 模块 | 参数量 |
|------|--------|
| Adapter 0 (down+up) | 41,344 |
| Adapter 1 (down+up) | 41,344 |
| classifier | 963 |
| **总计** | **83,651** |

Day 1 估算：82,688 (Adapter) + 963 (classifier) = 83,651 ✅ 完全吻合

### Q2：闭包捕获

Python 闭包捕获变量引用而非值。直接写 `self.adapters[i]` 会导致所有 hook 共享循环结束后的 `i` 值，全部指向最后一个 adapter。

### Q3：训练结果

```
Epoch 10 | Val Acc: 93.26%
Week 7 解冻2层: ~92%
```

**用更少的可训练参数（83K vs 2.4M）达到了更好的效果** — Adapter 的 bottleneck (320→64→320) 起到了隐式正则化作用。

---

## 总结

Day 3 的核心收获是**如何在不修改预训练模型源码的情况下插入新模块**。`register_forward_hook` 是 PyTorch 提供的非侵入式插桩机制，配合 PEFT 的冻结策略，实现了"只训练新增参数"的目标。

同时发现了一个代码隐患：`state_dict()` 返回引用导致权重验证可能失效，后续应使用 `.clone()` 或 `deepcopy()`。
