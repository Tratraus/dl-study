# Week 11 Review：ESM-2 + 真实蛋白质数据

## 本周全景

Week 11 是从**合成数据**到**真实蛋白质**的跨越。前 10 周用随机序列验证概念，本周用真实的亚细胞定位数据（8388 条蛋白质，10 个类别）跑通了完整的迁移学习 pipeline。

```
Day 1：ESM-2 8M 加载 + 嵌入提取 → 10 条蛋白质，余弦相似度分析
Day 2：真实数据集构造（ProtST-SubcellularLocalization）→ 闭包 collate_fn + 掩码 Mean Pooling
Day 3：冻结 ESM-2 + MLP 分类头 → Val Acc 63.4%（42,378 可训练参数）
Day 4：测试集评估 + 混淆矩阵 → Test Acc 66.4%，Macro F1 0.551
Day 5：Fine-tuning 最后 2 层 → Val Acc 677%（+4.3%），差分学习率
Day 6：Fine-tuned 测试集评估 → Test Acc 676%（+1.2%），Macro F1 0.581（+3.0%）
Day 7：注意力可视化 + t-SNE 收官 → 信号肽注意力 + 类别分离验证
```

---

## 技术栈与数据流全景

```
输入：蛋白质序列（字符串）
  │
  ▼ ESM-2 Tokenizer（HuggingFace）
input_ids: (B, L+2), attention_mask: (B, L+2)
  │
  ▼ ESM-2 8M（facebook/esm2_t6_8M_UR50D）
last_hidden_state: (B, L+2, 320)
  │
  ▼ 掩码 Mean Pooling（去 <cls>/<eos>，排除 PAD）
embeddings: (B, 320)
  │
  ▼ ProteinClassifier（两层 MLP：320→128→10）
logits: (B, 10)
  │
  ▼ CrossEntropyLoss（带类别权重）
loss → backward → optimizer.step()
```

### 关键技术点

| 技术 | Day | 要点 |
|------|-----|------|
| 掩码 Mean Pooling | 2 | 修复 Day1 的 PAD 污染 bug（最大差异 0.605） |
| 闭包 collate_fn | 2 | 把 tokenizer 打包进函数，避免全局变量 |
| 冻结 + eval + no_grad 三件套 | 3 | 三者缺一不可：冻结防更新，eval 关 dropout，no_grad 省显存 |
| 类别权重 | 3 | weight[c] = total / (K × count[c])，Peroxisome 权重是 Nucleus 的 26 倍 |
| 差分学习率 | 5 | ESM-2 解冻层 1e-5，分类头 1e-3，100 倍差距防灾难性遗忘 |
| 热启动 | 5 | 从 Day3 的分类头权重初始化，加速收敛 |
| 选择性解冻 | 5 | 只解冻最后 2 层（32.8% 参数），保持预训练表示 |
| eager 注意力后端 | 7 | sdpa 不支持 output_attentions，需用 attn_implementation="eager" |

---

## 核心数据对比

### 训练策略对比

| 策略 | 可训练参数 | Val Acc | Test Acc | Macro F1 |
|------|-----------|---------|----------|----------|
| 冻结 ESM-2 + MLP | 42,378（0.6%） | 63.4% | 66.4% | 0.551 |
| Fine-tuning 最后 2 层 | 2,508,298（33.4%） | 677% | 676% | 0.581 |
| **Δ** | **+59×** | **+4.3%** | **+1.2%** | **+3.0%** |

### Val vs Test 提升差距

| 数据集 | 提升幅度 | 说明 |
|--------|---------|------|
| Val Acc | +4.3% | Fine-tuning 在验证集上的提升 |
| Test Acc | +1.2% | 实际泛化提升 |
| 差距 | 3.1% | 过拟合验证集的证据（泛化提升仅为 Val 的 28%） |

### 逐类别 F1 变化

| 类别 | Frozen | FT | Δ | 生物学解释 |
|------|:------:|:--:|:--:|-----------|
| Extracellular | 0.894 | 0.924 | +0.030 | 信号肽特征明确 |
| Plastid | 0.707 | 0.771 | +0.064 | 转运肽特征被学到 |
| Mitochondria | 0.629 | 0.688 | +0.059 | MTS 被识别 |
| Endoplasmic reticulum | 0.481 | 0.563 | **+0.082** | ER 信号肽最受益 |
| Golgi apparatus | 0.338 | 0.386 | +0.048 | 样本少但有进步 |
| Peroxisome | 0.214 | 0.250 | +0.036 | Recall 0.231→0.385 |
| Cell membrane | 0.683 | 0.694 | +0.011 | 已饱和 |
| Nucleus | 0.753 | 0.744 | -0.009 | 微降，被其他类"抢走"样本 |
| Cytoplasm | 0.520 | 0.516 | -0.004 | "默认类别"，信号弱 |
| Lysosome/Vacuole | 0.291 | 0.270 | **-0.021** | 与 Cytoplasm 混淆加剧 |

**规律**：有明确序列信号（信号肽、转运肽、MTS）的类别受益最大；"默认类别"（Cytoplasm）和信号弱的类别（Lysosome）反而受损。

---

## 类别不平衡问题

### 问题严重程度

| 类别 | 训练样本 | 权重 | F1 (FT) |
|------|---------|------|---------|
| Nucleus | ~1697 | 0.35 | 0.744 |
| Peroxisome | ~65 | 9.02 | 0.250 |
| **比值** | **26×** | **26×** | — |

加权 CrossEntropyLoss 通过 26:1 的权重比试图平衡，但效果有限——Peroxisome 的 F1 仍然只有 0.250。模型基本"放弃"了这个类别（Precision 和 Recall 都低）。

### 可能的改进方向

| 方法 | 原理 | 预期效果 |
|------|------|---------|
| Focal Loss | 对难样本加权，降低易样本贡献 | 改善少数类 Recall |
| 过采样（SMOTE/重复） | 增加少数类训练样本 | 直接增加少数类学习机会 |
| 数据增强 | 序列扰动、突变模拟 | 增加多样性 |
| Early Stopping | 在 val loss 上升前停止 | 减轻过拟合 |
| 更强分类头 | 加深 MLP / 加 BatchNorm | 提升表达能力 |

---

## 注意力可视化发现

ESM-2 在 Fine-tuning 后学到了**生物学上有意义的注意力模式**：

| 类别 | 注意力集中区域 | 对应生物学特征 |
|------|--------------|---------------|
| Extracellular | N 端（位置 0-20） | 信号肽（signal peptide） |
| Nucleus | 位置 10-20, 40-60 | 核定位信号（NLS） |
| Mitochondria | N 端（位置 0-30） | 线粒体靶向序列（MTS） |

这说明 Fine-tuning 不仅提升了分类准确率，还让 ESM-2 的注意力"聚焦"到了有生物学功能的序列区域。

---

## t-SNE 嵌入空间对比

| 模型 | 类别分离 | 关键观察 |
|------|---------|---------|
| Frozen ESM-2 | ❌ 几乎无分离 | 所有类别混成一团，预训练表示是"通用的" |
| Fine-tuned ESM-2 | ✅ 有清晰分离 | Extracellular 和 Plastid 聚类最好，Nucleus/Cytoplasm 重叠 |

**核心结论**：预训练模型的表示是"通用的"，不是"任务特定的"。Fine-tuning 的本质是把通用表示空间"旋转"成对下游任务友好的方向。

---

## 踩坑汇总

| 坑 | Day | 影响 | 修复 |
|----|-----|------|------|
| PAD 污染 Mean Pooling | 1→2 | 嵌入差异最大 0.605 | 掩码 Mean Pooling |
| `n_iter` vs `max_iter` | 7 | sklearn 版本兼容 | 改为 `max_iter` |
| `sdpa` 不支持 `output_attentions` | 7 | 注意力权重无法提取 | 用 `attn_implementation="eager"` |
| `torch.load` 缺 `weights_only` | 3-7 | PyTorch 2.6+ warning | 建议统一加 `weights_only=True` |
| 无 Early Stopping | 5 | 过拟合未及时停止 | Epoch 8 后 val acc 停滞 |

---

## 与 Week 10 的对比

| 维度 | Week 10 (ProteinBERT) | Week 11 (ESM-2) |
|------|----------------------|-----------------|
| 数据 | 合成随机序列 | 真实蛋白质（8388 条） |
| 模型 | ProteinBERT ~470K 参数 | ESM-2 8M（7.5M 参数） |
| 嵌入维度 | 128 | 320 |
| 任务 | 二分类（正电荷） | 10 类亚细胞定位 |
| 预训练 vs 随机 | 差别不大（合成数据无模式） | 差别显著（真实数据有生物信号） |
| Fine-tuning | 未做 | 解冻 2 层，+1.2% Test Acc |
| 注意力可视化 | 未做 | 信号肽区域注意力集中 |

**关键洞察**：Week 10 用合成数据看不到预训练的优势，Week 11 用真实数据才验证了迁移学习的价值。**数据质量决定了一切。**

---

## 技能树更新

```
Week 11 新增技能：
├── ESM-2 迁移学习全流程
│   ├── HuggingFace 加载预训练蛋白质模型
│   ├── 真实数据集构造（HuggingFace Datasets）
│   ├── 闭包 collate_fn + 掩码 Mean Pooling
│   ├── 冻结/解冻策略 + 差分学习率
│   ├── 类别不平衡处理（加权 CrossEntropy）
│   └── 测试集系统性评估（P/R/F1 + 混淆矩阵）
├── 注意力可视化
│   ├── output_attentions=True + eager 后端
│   ├── 多头平均注意力矩阵提取
│   └── 信号肽区域注意力分析
└── 嵌入空间可视化
    ├── t-SNE 降维（sklearn TSNE）
    └── Frozen vs Fine-tuned 对比
```

---

## 总结

Week 11 完成了从**数据准备 → 冻结分类 → 测试评估 → Fine-tuning → 可视化**的完整闭环。

**核心收获**：
1. 真实数据上，预训练 ESM-2 的嵌入需要 Fine-tuning 才能有效区分亚细胞定位
2. 选择性解冻 + 差分学习率是安全的 Fine-tuning 策略
3. 类别不平衡是真实数据的核心挑战，加权 Loss 有帮助但不够
4. 注意力可视化能揭示模型学到了哪些生物学特征
5. 过拟合是小数据集 Fine-tuning 的主要风险

**下一步方向**：
- Focal Loss / 过采样处理少数类
- Early Stopping 防止过拟合
- 尝试更大的 ESM-2（150M / 3B）
- GNN（Week 12）引入蛋白质结构信息

---

_Week 11 Review by Talos | 2026-06-25_
