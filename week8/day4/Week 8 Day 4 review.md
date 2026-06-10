# Week 8 Day 4 复盘：Adapter vs 解冻微调对比实验

## 代码结构

```
compare.py
├── AdapterLayer           # 复用 Day 2 的 Adapter 模块
├── ESMClassifier          # 通用分类器基类（ESM 全冻结 + classifier）
├── build_frozen()         # 策略 A：全冻结
├── build_unfreeze()       # 策略 B：解冻最后 N 层
├── build_adapter()        # 策略 C/D：插入 Adapter + hook
├── generate_data()        # 合成数据（复用 Week 7）
├── ProteinDataset         # Dataset 封装
├── train_epoch()          # 训练一个 epoch
├── eval_epoch()           # 验证 + F1 计算
├── set_seed()             # 随机种子统一设置
├── run_experiment()       # 单次实验封装（训练 + 计时 + 返回结果）
└── main()                 # 入口：四策略依次运行 → 汇总表格
```

---

## 数据流 & 形状变化

与 Day 3 完全一致（共用 ESMClassifier 的 forward 路径）：

```
序列 → batch_converter → tokens (batch, L+2)
  → ESM-2 Embedding → (batch, L+2, 320)
  → Layer 0-5（全冻结/解冻/有 hook，视策略而定）
  → classifier → (batch, L+2, 3)
  → 切片 [:, 1:-1, :] → (batch, L, 3)
  → CrossEntropyLoss
```

**eval_epoch 新增的数据收集：**
```
preds[mask]         → 一维张量，有效位置的预测
label_tensor[mask]  → 一维张量，有效位置的标签
→ .extend() 到 all_preds / all_labels
→ f1_score(all_labels, all_preds, average=None, labels=[0,1,2])
```

---

## 关键知识点

### 1. 公平对比的实现

```python
set_seed(42)  # 每次实验前重置所有随机种子
```

确保四种策略在相同数据、相同初始化、相同训练配置下对比。

### 2. ESMClassifier 作为基类的设计

```python
class ESMClassifier(nn.Module):
    def __init__(self):
        # ESM 全部冻结（默认状态）
        # classifier 默认可训练
        self.adapters = nn.ModuleList()  # 空的，按需填充
```

策略通过外部的 `build_*` 函数控制哪些参数可训练，而不是在模型内部硬编码——这是**策略模式**的体现。

### 3. `filter(lambda p: p.requires_grad, model.parameters())`

```python
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
)
```

只对 `requires_grad=True` 的参数创建优化器状态，节省显存。

### 4. F1 的意义

Val Acc 是整体准确率，但标签分布严重不平衡（C 占 ~77%）。F1 per class 揭示了：
- **F1-H 最低**：α螺旋样本最少，边界模糊
- **F1-C 最高**：coil 样本最多，最容易学
- **F1-E 居中**：β折叠

---

## 实验结果汇总

| 策略 | 参数量 | Val Acc | F1-H | F1-E | F1-C | 秒/epoch |
|------|--------|---------|------|------|------|----------|
| A. 全冻结 | 963 | 0.9199 | 0.000 | 0.880 | 0.954 | 0.2 |
| B. 解冻2层 | 2,466,883 | 0.9122 | 0.376 | 0.855 | 0.949 | 0.2 |
| C. Adapter b=64 | 83,651 | 0.9296 | 0.381 | 0.880 | 0.960 | 0.2 |
| D. Adapter b=256 | 329,795 | 0.9326 | 0.454 | 0.878 | 0.962 | 0.2 |

---

## 踩坑 & 易错点

| 易错点 | 说明 |
|--------|------|
| 全冻结策略的 F1-H=0 | 只有 963 个可训练参数，分类头不足以学习 H 类的特征模式 |
| 解冻2层的过拟合 | Train 99.29% vs Val 91.22%，Epoch 3 后 Val 持续下降 |
| Adapter 学习速度慢 | Epoch 1-3 几乎没变化（F1-H=0），需要更多 epoch 才能收敛 |
| 时间差异不明显 | 模型太小（8M），GPU 利用率低，掩盖了实际计算量差异 |

---

## 输出问题回顾

### Q1：Adapter b=64 vs 解冻2层

- **参数量**：83,651 vs 2,466,883 → **减少约 29.5 倍**
- **Val Acc**：92.96% vs 91.22% → **Adapter 反而高 1.74 个百分点**
- 结论：Adapter 用 1/30 的参数达到了更好的效果，PEFT 的参数效率优势明显

### Q2：bottleneck 64 → 256 的性价比

- **Val Acc**：93.26% - 92.96% = **+0.3 个百分点**
- **参数量**：329,795 / 83,651 ≈ **4 倍**
- 性价比很低。参数量翻了 4 倍，性能提升不到 0.5%。在资源受限场景下，b=64 是更优选择。

### Q3：秒/epoch 规律

四种策略的秒/epoch 都在 0.2s 左右，几乎无差异。原因：

1. **模型太小**：ESM-2 8M 只有 6 层，即使解冻 2 层，额外的梯度计算量相对于 GPU 算力可以忽略
2. **GPU 利用率低**：batch_size=8 + 短序列（50-150），GPU 大部分时间在等数据，不在计算
3. **显存充裕**：RTX 4090 24GB 跑 8M 模型，无论哪种策略都绑绑有余

如果换成 ESM-2 150M 或更大的模型，时间差异会显著体现——解冻 2 层的梯度计算量远大于只更新 Adapter 参数。

---

## 核心结论

| 维度 | 全冻结 | 解冻2层 | Adapter b=64 | Adapter b=256 |
|------|--------|---------|-------------|---------------|
| 参数效率 | 最高 | 最低 | 高 | 中 |
| 性能 | 差（F1-H=0） | 过拟合 | 好 | 最好 |
| 稳定性 | 稳定但欠拟合 | 过拟合风险 | 稳定 | 稳定 |

**关键洞见**：在这个小数据集（240 条）场景下，Adapter 是最优策略——参数少、不过拟合、性能最佳。解冻2层参数太多导致过拟合，全冻结参数太少导致欠拟合。Adapter 的 bottleneck 起到了隐式正则化的作用。
