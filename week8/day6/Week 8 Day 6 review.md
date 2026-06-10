# Week 8 Day 6 复盘：LoRA 注入 ESM-2 训练实验

## 代码结构

```
lora_esm.py
├── LoRALinear          # Day 5 复用，低秩适配器
├── ESMWithLoRA         # 核心模型类
│   ├── 加载 ESM-2 8M + 全冻结
│   ├── TODO 1: 替换最后 2 层 Q/V → LoRALinear  ← 今日唯一代码任务
│   └── classifier (Linear 320→3)
├── generate_data       # 合成蛋白质序列 + 二级结构标签（复用）
├── train_epoch / eval_epoch  # 训练/评估循环（复用）
└── main                # 完整训练 + 汇总对比表
```

## 数据流 / 形状变化

```
输入序列 → batch_converter → tokens [B, L+2]
  → ESM-2 embed → [B, L+2, 320]
  → TransformerLayer × 6（最后 2 层 Q/V 走 LoRA 前向）
    → LoRALinear.forward:
        base_out = W·x           # 原始投影，冻结
        lora_out = B·(A·x)       # 低秩路径，可训练
        output = base_out + (α/r) × lora_out  # 残差融合
  → 最后一层 repr [B, L+2, 320]
  → classifier → [B, L+2, 3]
  → slice [B, L, 3]（去掉 CLS/SEP）
  → CrossEntropyLoss（ignore -100 padding）
```

## 关键知识点

### 1. LoRA 的"注入"机制 — 如何挂载到原模型

**核心理解**：LoRA 的本质是用两个低秩矩阵 A、B 近似 ΔW 来做微调。我们要做的是把这两个矩阵"挂载"到原模型中。

**挂载方式：Python 属性替换 + 多态调用**

```python
# 原始状态：q_proj 是 nn.Linear
layer.self_attn.q_proj = nn.Linear(320, 320)  # 权重 W

# 替换后：q_proj 变成 LoRALinear
layer.self_attn.q_proj = LoRALinear(original_q)  # 包含冻结的 W + 可训练的 A、B
```

ESM-2 内部调用 `self.q_proj(x)` 时，Python 自动分派到 LoRALinear.forward：
```python
def forward(self, x):
    base_out    = self.base_linear(x)      # W·x（冻结）
    lora_hidden = F.linear(x, self.lora_A) # A·x（可训练）
    lora_out    = F.linear(lora_hidden, self.lora_B) # B·A·x（可训练）
    return base_out + self.scale * lora_out  # W·x + (α/r)·B·A·x
```

**没有 hook，没有注册表，没有特殊机制**——就是对象属性替换，Python 多态的经典玩法。

**与 Adapter 的本质区别**：
- **Adapter**：在管道中间加一个新阀门。水流经过原管道 → 被拦截 → 进阀门处理 → 流回原管道。管道结构变了。
- **LoRA**：把原管道偷偷换成了双层管。外层管（原始 W）照常走，内层管（A·B）额外分流一部分，最后合并。从外面看管道没变，接口也没变。

用一句话总结：**Adapter 是拦截后单独处理，LoRA 是偷换实现、保留接口。**

### 2. 参数量精确计算
- ESM-2 8M: d_model=320, 6 层 Transformer
- 每个 LoRALinear: A(8×320) + B(320×8) = 2,560 + 2,560 = 5,120
- 替换 2 层 × Q+V = 4 个 LoRALinear = 20,480
- Classifier: 320×3 + 3 = 963
- **总计: 21,443** ✓

### 3. 训练曲线特征
- Epoch 1-6: F1-H 完全为 0，模型"看不到"H 类
- Epoch 7-10: F1-H 从 0.068 → 0.315，缓慢上升
- 原因：LoRA 参数量极少（21K vs Adapter 的 83K），低秩空间需要更多迭代才能收敛到有用方向
- Val Acc 从 Epoch 2 就稳定在 ~0.92，因为 C 类占 77%，整体 Acc 被主导

### 4. 与 Adapter 的对比（参数效率）
| 策略 | 参数量 | Val Acc | F1-H | 参数效率 |
|------|--------|---------|------|----------|
| Adapter b=64 | 83,651 | 0.9296 | 0.381 | baseline |
| LoRA r=8 | 21,443 | 0.9263 | 0.315 | ~4× 压缩，Acc 差 0.33% |

LoRA 以 1/4 参数逼近 Adapter，但在最难的 H 类上仍有 gap。

## 踩坑易错点

1. **替换 vs hook**：LoRA 是直接替换属性（`layer.self_attn.q_proj = ...`），不是用 register_forward_hook。两种方式效果一样，但替换更简洁
2. **只替换 Q/V，不替换 K**：这是 LoRA 原论文的经验结论——Q/V 修改影响输出内容，K 修改影响注意力分布模式，风险更大
3. **scale = alpha/rank**：alpha=16, rank=8 时 scale=2，相当于把 LoRA 分支的贡献放大 2 倍，帮助低秩路径在训练初期更快生效

## 输出问题回答

**Q1**：实际参数量 21,443，与估算完全一致。
- 4 × (8×320 + 320×8) = 20,480 (LoRA)
- 320×3 + 3 = 963 (classifier)
- 合计 21,443

**Q2**：Adapter b=64 vs LoRA r=8
- Val Acc 差 0.0033（0.33%）
- F1-H 差 0.066（6.6%）
- LoRA 参数效率高（4× 压缩比），但在样本最少的 H 类上仍有可观察的 gap

**Q3**：为什么不替换 K？
- Q 决定"我在找什么"，K 决定"我有什么"，V 决定"给你什么"
- Q/V 修改直接影响输出内容（下游任务最需要调整的维度）
- K 修改会改变注意力分布模式（谁关注谁），这些模式是预训练学到的通用结构，改动风险大
- 原论文实验验证：只改 Q/V 是参数效率和性能的最佳平衡点
