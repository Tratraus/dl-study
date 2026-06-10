# Week 8 Day 7 复盘：多任务学习初探

## 代码结构

```
multitask.py
├── 标签定义          # AA, SS_CLS, HYDRO（疏水集合）
├── generate_data     # 生成序列 + SS3 标签 + 疏水标签
│   └── TODO 1: hydro_labels = [1 if aa in HYDRO else 0 for aa in seq]
├── MultiTaskESM      # 共享主干 + 双任务头
│   ├── Adapter（hook 方式，复用 Day 3）
│   ├── TODO 2: ss3_head(320→3) + hydro_head(320→2)
│   └── TODO 3: forward 返回两个 logits
├── train_epoch
│   ├── TODO 4: 构建 hydro_tensor（和 ss_tensor 同构，padding=-100）
│   └── TODO 5: loss = loss_ss3 + loss_hydro
├── eval_epoch        # 双任务评估
└── main              # TODO 6 已填，对比单任务结果
```

## 数据流 / 形状变化

```
输入序列 → batch_converter → tokens [B, L+2]
  → ESM-2 embed → TransformerLayer × 6（最后 2 层有 Adapter hook）
  → 最后一层 repr [B, L+2, 320]
  → slice [B, L, 320]（去掉 CLS/SEP）
  → ss3_head(x)   → ss3_logits   [B, L, 3]
  → hydro_head(x) → hydro_logits [B, L, 2]
  → L_total = CE(ss3) + CE(hydro)  # λ₁=λ₂=1
  → backward → 梯度同时流向两个头 + Adapter + 共享主干
```

## 关键知识点

### 1. 多任务学习的梯度流
- L_total = L_SS3 + L_Hydro
- 两个 loss 的梯度在共享主干上**叠加**
- 共享主干被两个任务的梯度共同塑造 → 正则化效果
- 一个任务的过拟合方向会被另一个任务抑制

### 2. 参数量分析
- Adapter 参数：83,651（和 Day 4 单任务相同）
- 新增 hydro_head：320×2 + 2 = 642
- **总计：84,293**（比单任务多 642，仅来自第二个任务头）

### 3. 训练结果
- **SS3 Val Acc：0.9296 → 0.9347（↑0.0051）**
- **SS3 F1-H：0.381 → 0.499（↑0.118）**
- 多任务训练显著提升了最难的 H 类预测
- Hydro Val Acc = 1.0000，F1 = 1.000（任务本身是平凡的）

### 4. Hydro 任务为什么是"平凡任务"
- 疏水标签直接由氨基酸身份决定（`aa in HYDRO`），是确定性规则
- 模型本质上只需要学会读取输入 token 的 identity → 做一个 lookup table
- 没有噪声、没有模糊边界，所以 100% 准确率
- 但即使任务简单，它的梯度仍然对共享主干有正则化作用

## 踩坑易错点

1. **hydro_tensor 的构建**：标签已经是整数列表，不需要 SS2IDX 映射，直接 `torch.tensor(lb)` 即可
2. **padding 用 -100**：两个任务的 tensor 都用 -100 padding，`CrossEntropyLoss` 默认 `ignore_index=-100`
3. **loss 加权**：今天用 λ₁=λ₂=1，实际项目中可能需要根据任务难度调整权重

## 输出问题回答

**Q1**：可训练参数量 84,293，比单任务 Adapter b=64（83,651）多 642。多出来的参数来自新增的 `hydro_head = Linear(320, 2)`（权重 320×2 + 偏置 2 = 642）。

**Q2**：SS3 Val Acc 从 0.9296 升到 0.9347（↑0.0051），F1-H 从 0.381 升到 0.499（↑0.118）。原因：多任务学习的正则化效应——两个任务共享主干，Hydro 的梯度帮助主干学到更通用的序列表示，抑制了 SS3 的过拟合方向。H 类提升最大，说明多任务对样本少的困难类别帮助最明显。

**Q3**：Hydro Val Acc = 1.0000，任务本身是平凡的。疏水标签由氨基酸身份直接决定（确定性规则），模型只需学会 lookup。但即使是"简单任务"，其梯度仍然对共享主干有正则化价值——这就是多任务学习的核心洞察。
