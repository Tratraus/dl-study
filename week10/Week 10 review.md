# Week 10 Review — BERT-style MLM 预训练

## 全景概览

Week 10 用 7 天完成了 BERT-style 蛋白质语言模型的完整流程：

```
Day 1: MLM 原理 + 数据构造（mask_sequence, make_mlm_batch）
Day 2: Encoder-only ProteinBERT 搭建（Transformer × 3 + MLM Head）
Day 3: MLM 预训练循环（train_step, loss 曲线）
Day 4: 序列嵌入提取（encode 方法, CLS/Mean 池化）
Day 5: 下游分类任务（冻结 Encoder, K/R 分类）
Day 6: Fine-tuning 与对比实验（三组对比, 分组学习率）
Day 7: 嵌入可视化（PCA/t-SNE, 定量评估）
```

**核心主题：** 预训练 → 提取表示 → 下游应用 → 验证预训练价值

---

## 技术栈

| 组件 | 实现 |
|---|---|
| 模型架构 | Encoder-only Transformer (d_model=128, 3层, 4头) |
| 预训练任务 | MLM (Masked Language Model), 80-10-10 策略 |
| 池化方法 | CLS 池化 (BOS token) + Mean 池化 (非PAD平均) |
| 下游任务 | 二分类 (正电荷蛋白 / Motif 检测) |
| 可视化 | PCA (手动实现) + t-SNE (sklearn) |
| 总参数量 | 469,657 |

---

## 每日关键收获

### Day 1：MLM 数据构造
- **80-10-10 策略**：80% MASK / 10% 随机替换 / 10% 保持不变
- `ignore_index=-100` 让 cross_entropy 跳过非 mask 位置
- 合成数据的验证：mask 比例、label 对应关系

### Day 2：ProteinBERT 搭建
- **Encoder-only** = 去掉 Decoder + 加 MLM Head
- `nn.ModuleList` vs 普通 list（for 循环必须用 ModuleList）
- `batch_first=True` 改变 Transformer 的输入格式
- `padding_idx=PAD` 让 PAD embedding 恒为零

### Day 3：MLM 预训练
- **初始 loss ≈ log(vocab_size)** ≈ 3.22（均匀分布假设）
- `logits.view(-1, vocab_size)` 展平为 (N, C) 格式
- 合成随机数据的天花板：loss 卡在 ~2.88（只能学到边缘分布）
- 换真实数据才能看到进一步下降

### Day 4：序列嵌入提取
- **encode vs forward**：只差最后 MLM Head 一步
- CLS 池化：BOS 位置通过 attention 聚合全局信息
- Mean 池化：`mask.sum(dim=1)` 做除数而非 `seq_len`
- `model.eval()` 保证推理确定性（Dropout 关闭）

### Day 5：下游分类（冻结 Encoder）
- 冻结 Encoder 只训练分类头（8,386 参数）
- `torch.no_grad()` 提取嵌入，节省显存
- K/R 频率任务 50 步内收敛，Test Acc 99.92%
- 预训练嵌入质量验证通过

### Day 6：Fine-tuning 与对比实验
- **分组学习率**：`Adam([{'params': ..., 'lr': 1e-4}, ...])`
- 灾难性遗忘：encoder 用小学习率保护预训练知识
- 三组对比：随机+冻结 / 预训练+冻结 / 预训练+FT
- 任务太简单时三组无差异——实验设计的重要性

### Day 7：嵌入可视化
- **PCA 手动实现**：中心化→协方差→eigh→top2 投影
- **t-SNE**：非线性降维，保留局部邻居关系
- 定量评估：类内/类间距离比
- 预训练 Encoder 聚类质量优于随机（比值 0.635 vs 0.577）

---

## 关键技术对比表

### 池化方法

| | CLS 池化 | Mean 池化 |
|---|---|---|
| 实现 | 拼 BOS → 取位置 0 | 构造 mask → 加权平均 |
| 范数 | ~10.65 | ~6.6 |
| 适用 | 序列等长 | 变长序列 |
| 信息来源 | attention 聚合 | 所有位置均值 |

### 冻结 vs Fine-tuning

| | 冻结 | Fine-tuning |
|---|---|---|
| 可训练参数 | 仅分类头 (8,386) | 全部 (~478K) |
| 学习率 | 1e-3 | encoder 1e-4 + classifier 1e-3 |
| 训练速度 | 快 | 慢 |
| 数据需求 | 少 | 多 |
| 风险 | 低 | 灾难性遗忘 |
| 适用 | 标注少、任务简单 | 标注充足、任务复杂 |

### 降维方法

| | PCA | t-SNE |
|---|---|---|
| 变换类型 | 线性 | 非线性 |
| 速度 | 快 | 慢 |
| 保留结构 | 全局 | 局部 |
| 可复现 | 完全确定 | 依赖 random_state |
| 适用 | 快速检查 | 聚类可视化 |

---

## 代码架构总览

```
week10/
├── day1/
│   └── mlm_data.py          ← 词表、MASK 策略、数据生成
├── day2/
│   └── protein_bert.py      ← PositionalEncoding + ProteinBERT + encode
├── day3/
│   └── train_mlm.py         ← train_step + train + plot_loss_curve
├── day4/
│   └── extract_embedding.py ← cls_pooling + mean_pooling
├── day5/
│   └── classify.py          ← make_classification_batch + ClassificationHead + train
├── day6/
│   └── finetune.py          ← has_hydrophobic_motif + train_model + run_experiments
└── day7/
    └── visualize.py         ← make_class_data + PCA + t-SNE + compute_distance_ratio
```

**依赖链：** day1 → day2 → day3 → day4 → day5/6/7

---

## 核心概念清单

| 概念 | 含义 | 首次出现 |
|---|---|---|
| MLM | Masked Language Model，遮盖部分 token 让模型预测 | Day 1 |
| 80-10-10 | MASK 80% / 随机替换 10% / 保持 10% | Day 1 |
| ignore_index | cross_entropy 跳过指定 label 位置 | Day 1 |
| batch_first | Transformer 输入格式 (batch, seq, d) vs (seq, batch, d) | Day 2 |
| log(vocab_size) | 均匀分布下的理论初始 loss | Day 3 |
| encode vs forward | 是否过 MLM Head | Day 4 |
| CLS / Mean 池化 | 序列压缩为固定维度向量的两种策略 | Day 4 |
| 冻结 Encoder | requires_grad=False，只训练分类头 | Day 5 |
| torch.no_grad() | 不构建计算图，节省显存 | Day 5 |
| 分组学习率 | 不同参数组用不同 lr | Day 6 |
| 灾难性遗忘 | Fine-tuning 破坏预训练知识 | Day 6 |
| PCA | 线性降维，找方差最大方向 | Day 7 |
| t-SNE | 非线性降维，保留邻居关系 | Day 7 |
| 类内/类间距离比 | 聚类质量定量指标 | Day 7 |

---

## 踩坑汇总

| 问题 | 出现 | 原因 | 修正 |
|---|---|---|---|
| `nn.ModuleList` vs list | Day 2 | 普通 list 不注册子模块 | 用 `nn.ModuleList` |
| `batch_first=True` | Day 2 | 维度不匹配 | 设置 batch_first |
| loss 卡在 ~2.88 | Day 3 | 合成数据无模式 | 换真实数据 |
| `loss_history` 为 None | Day 3 | 忘记 return | 加 `return loss_history` |
| `model.eval()` 遗忘 | Day 4 | Dropout 未关闭 | 提取嵌入前切换 |
| Mean 池化除数错误 | Day 4 | 用 seq_len 代替 mask.sum | 用有效长度做除数 |
| `n_iter` 报错 | Day 7 | sklearn API 变更 | `n_iter` → `max_iter` |
| 中文字体警告 | Day 7 | matplotlib 无中文字体 | 改英文标题 |

---

## 下一步方向

Week 10 完成了 BERT-style MLM 的完整 pipeline。后续可以：

1. **换真实蛋白质数据**（UniRef50 片段）重新预训练，看到 loss 真正下降
2. **增大模型**（d_model=256, 6 层）提升表示质量
3. **更难的下游任务**（二级结构预测、蛋白质功能分类）体现预训练价值
4. **对比实验**：不同预训练策略（MLM vs CLM）对下游任务的影响

---

_Week 10 完成于 2026-06-22，共 7 天。_
