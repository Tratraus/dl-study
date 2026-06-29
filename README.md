# 🧬 dl-study — 深度学习 × 计算生物学学习笔记

> 从零开始，系统性学习深度学习在计算生物学中的应用。
> 每一天的代码、复盘、踩坑记录，全部留档。

## 📍 当前进度

| 阶段 | Weeks | 状态 |
|------|-------|------|
| 深度学习基础 | W1 ~ W9 | ✅ 已完成 |
| 迁移学习 + 预训练 | W10 ~ W11 | ✅ 已完成 |
| 注意力机制深入理解 | W12 | ✅ 已完成 |
| 迁移学习 + 工程实践 | W13 | 🔜 进行中 |
| 多标签分类 → GNN → 结构建模 → 单细胞 → 生成模型 → 多组学 → 综合项目 | W14 ~ W20 | ⬜ 计划中 |

## 🗺️ 全景路线图

```
W1  PyTorch 基础          W8  PEFT (Adapter/LoRA/多任务)
W2  训练流程              W9  Seq2Seq (Attention/Beam Search)
W3  数据工程              W10 BERT-style MLM 预训练
W4  RNN / LSTM            W11 ESM-2 + 真实蛋白质数据
W5  Transformer 从零      W12 注意力机制深入理解 ✅
W6  蛋白质二级结构预测    W13 迁移学习 + 工程实践 ← 当前
W7  ESM-2 迁移学习        W14 多标签分类 → W15 GNN
                          W16 结构建模 → W17 单细胞
                          W18 生成模型 → W19 多组学
                          W20 综合项目
```

## 📁 仓库结构

```
dl-study/
├── week1/          # PyTorch 基础：张量、自动求导、MLP
├── week2/          # 训练流程：train/val split、评估、checkpoint
├── week3/          # 数据工程：Dataset、归一化、Dropout、调度器
├── week4/          # 序列模型：RNN → LSTM → BiLSTM，蛋白质序列处理
├── week5/          # Transformer：Self-Attention、多头注意力、位置编码
├── week6/          # 蛋白质二级结构预测：完整工程 pipeline
├── week7/          # ESM-2 迁移学习：冻结/解冻策略、分层学习率
├── week8/          # PEFT：Adapter、LoRA、多任务学习
├── week9/          # Seq2Seq：Cross-Attention、Greedy/Beam Search
├── week10/         # BERT-style MLM：预训练、嵌入提取、微调对比
├── week11/         # ESM-2 + 真实数据：亚细胞定位、Fine-tuning
├── week12/         # 注意力机制深入理解：从零实现 Transformer + 注意力可视化 + 复杂度分析
├── checkpoints/    # 模型权重
├── data/           # 数据集
├── notes/          # 学习笔记
├── dl_study周计划.md  # 最新学习计划（19 周）
└── env_check.py    # 环境检查脚本
```

## 🔬 关键实验结果

### Week 12 — 注意力机制深入理解（从零实现 Transformer）

| 模型 | 参数量 | 预训练 | Test Acc | Macro F1 |
|------|--------|--------|----------|----------|
| 自实现 Encoder (3层, d=128) | 600K | ❌ | 61.0% | 0.485 |
| ESM-2 Frozen + MLP | 8M (42K 可训练) | ✅ | 63.4% | 0.551 |
| ESM-2 Fine-tuned (last 2 layers) | 8M (2.5M 可训练) | ✅ | 67.7% | 0.581 |

- 核心结论：600K 从头训练 < 42K 冻结预训练，预训练不可替代
- 注意力可视化：从头训练 → 各头趋同（对角线局部模式）；预训练 → 各头分化（列型/行型/块型分工）
- 复杂度：O(n²d + nd²)，n 增加 4 倍 → QKᵀ 计算量增加 16 倍

### Week 7 — ESM-2 迁移学习（合成数据）

| 策略 | 可训练参数 | Val Acc |
|------|-----------|---------|
| 全冻结 | 963 | 86.4% |
| 解冻最后 2 层 | 2,466,883 | 97.58% |

### Week 8 — PEFT + 多任务学习

| 方法 | 可训练参数 | F1-H |
|------|-----------|------|
| 全冻结 | 963 | 0.281 |
| Adapter (b=64) | 83,651 | 0.494 |
| LoRA (r=8) | 21,443 | 0.315 |
| 多任务 Adapter | 84,293 | **0.499** |

### Week 10 — BERT-style MLM 预训练

- 预训练：3 层 Encoder，469,657 参数，loss 3.41 → 2.86
- 下游分类：冻结 Encoder + 分类头，K/R 正电荷分类 **99.92%** Test Acc
- 嵌入可视化：预训练 Encoder 聚类质量优于随机初始化

### Week 11 — ESM-2 + 真实蛋白质数据（亚细胞定位，10 类）

| 模型 | Test Acc | Macro F1 |
|------|----------|----------|
| Frozen ESM-2 + MLP | 66.4% | 0.551 |
| Fine-tuned ESM-2 (last 2 layers) | 67.6% | 0.581 |

- 最大提升类别：ER (+0.082)、Plastid (+0.064)
- 注意力可视化：ESM-2 学到了信号肽区域（N端信号肽、NLS、MTS）

## 🛠️ 技术栈

- **框架**: PyTorch
- **预训练模型**: ESM-2 (HuggingFace Transformers)
- **数据处理**: NumPy, Pandas, scikit-learn
- **可视化**: Matplotlib, Seaborn
- **蛋白质工具**: Biopython, HuggingFace ESM
- **环境**: Python 3.10+, CUDA

## 📝 学习方法

每个 Day 包含：
- **Task**: 当日任务说明（`Week X Day Y task.md`）
- **Code**: 实现代码（`*.py`）
- **Review**: 复盘笔记（`Week X Day Y review.md`），包含：
  - 代码结构分析
  - 数据流 / Tensor 形状变化追踪
  - 关键知识点
  - 踩坑与易错点

每个 Week 结束后写周复盘（`Week X review.md`），汇总技术对比、关键收获、技能树。

## 📖 学习计划

详细计划见 [`dl_study周计划.md`](./dl_study周计划.md)（20 周完整规划，含算法导论嵌入内容）。

---