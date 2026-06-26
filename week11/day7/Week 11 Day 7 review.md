# Week 11 Day 7 Review：注意力可视化 + t-SNE 收官

## 代码结构分析

| 函数 | 职责 |
|------|------|
| `get_attention_map()` | 单条序列 → ESM-2 forward（`output_attentions=True`）→ 最后一层多头平均 → (L, L) 注意力矩阵 |
| `plot_attention()` | viridis 热图，截断到 max_len=50 |
| `extract_embeddings()` | 批量提取 ESM-2 mean pooling 嵌入，限制 max_samples=500 |
| `plot_tsne()` | sklearn TSNE 降维 2D → 散点图，每类别不同颜色 |
| `main()` | Part1 注意力（3 类各 1 条）+ Part2 t-SNE（Fine-tuned vs Frozen） |

### 与 Task 模板的改动

1. **注意力可视化单独加载模型**：主人发现默认的 `sdpa` 后端不支持 `output_attentions`，改用 `attn_implementation="eager"` 重新加载 ESM-2 — ✅ 正确的排查
2. **`n_iter` → `max_iter`**：sklearn 版本兼容性修复，Week 10 Day 7 踩过的坑，这次直接改对了 ✓
3. **`save_path` 先创建目录再保存**：加了 `os.makedirs` — ✅

---

## 注意力热图分析

### Extracellular（227 aa）

- **对角线明亮** ✓，每条序列都有，说明每个位置对自身关注度高
- **N 端（~位置 0-20）有明显的亮色竖条/横条**：这是**信号肽（signal peptide）区域**，几乎所有位置都在"关注"这段序列。Extracellular 蛋白的信号肽负责引导蛋白进入分泌途径，ESM-2 学到了这个特征
- 中后段有稀疏的 off-diagonal 亮点，可能是功能域之间的长程接触

### Nucleus（82 aa）

- 序列最短，热图最清晰
- **位置 10-20 和 40-60 有 off-diagonal 亮块**：可能对应**核定位信号（NLS）**区域。NLS 通常是一段富含碱性氨基酸（K/R）的短肽，ESM-2 对这些位置有异常高的注意力
- 对角线外的块状结构比 Extracellular 更明显，说明短序列中长程依赖更容易被捕捉

### Mitochondria（512 aa）

- 序列最长（512 aa），截断到 500 位置后热图非常大
- 对角线仍然明亮，但 off-diagonal 结构变得稀疏和模糊
- **N 端（~位置 0-30）有较亮的区域**：线粒体靶向序列（mitochondrial targeting sequence, MTS）位于 N 端，ESM-2 对此有注意力
- 整体比前两张"更嘈杂"，因为 500 位置的注意力矩阵信息密度低

---

## t-SNE 对比分析

### Fine-tuned ESM-2

- **Extracellular（橙色）** 分离最好，形成独立的紧凑聚类 — 与 Day4/Day6 中 F1=0.924 一致
- **Plastid（紫色）** 也有较好的聚类 — F1=0.771
- **Nucleus（蓝色）和 Cytoplasm（棕色/绿色）** 严重重叠 — 与混淆矩阵中这两类互相混淆一致
- 整体有 10 个可辨别的颜色区域，说明 Fine-tuning 让嵌入空间有了生物学意义的结构

### Frozen ESM-2

- 所有类别**混成一团**，几乎没有可辨别的聚类
- 这验证了 Day3 的核心假设：预训练 ESM-2 的通用表示不足以区分亚细胞定位，**需要下游 Fine-tuning**
- 与 Week 10 的结论形成对比：Week 10 用合成随机数据，预训练 vs 随机 Encoder 看不出差别；现在用真实蛋白质，差别一目了然

### 关键结论

> **预训练模型的表示是"通用的"，不是"任务特定的"。** Fine-tuning 的本质是把通用表示空间"旋转"成对下游任务友好的方向。

---

## 踩坑 / 易错点

1. **`output_attentions=True` 需要 eager 后端**：ESM-2 默认用 `sdpa`（Scaled Dot-Product Attention），不支持输出注意力权重。主人正确排查并用 `attn_implementation="eager"` 单独加载模型

2. **长序列注意力图可读性差**：Mitochondria 512 aa 截断到 500 后，热图仍然非常大，单个像素的亮度差异难以辨别。建议未来对长序列只取 N 端（信号肽区域）或特定功能域

3. **t-SNE 的随机性**：虽然设了 `random_state=42`，但 t-SNE 结果对 perplexity 敏感。当前用 30 是合理默认值

---

## 输出问题回答评估

### Q1：对角线 + 连续亮色块

主人的回答 ✅ 正确：
- 对角线亮 = 自注意力强 ✓
- 连续亮色块 = 二级结构（α-螺旋 / β-折叠）✓

补充：从实际热图看，更显著的 off-diagonal 信号不是二级结构，而是**功能信号肽区域**（Extracellular 的信号肽、Nucleus 的 NLS、Mitochondria 的 MTS）。这些是 ESM-2 在 Fine-tuning 后学到的生物学特征。二级结构的局部注意力模式确实存在，但在这几张图中不如信号肽明显。

### Q2：t-SNE 类别分离 vs 混淆矩阵

主人的回答 ✅ 正确：
- Extracellular 分离最好 ✓
- Nucleus 和 Cytoplasm 混淆 ✓
- 与 Day4/Day6 混淆矩阵一致 ✓

### Q3：500 条限制 + perplexity

主人的回答 ✅ 正确：
- 500 条限制的原因：计算复杂度 ✓（t-SNE 是 O(N²)）
- perplexity=5 → 更关注局部 → 聚类更碎片化 ✓
- perplexity=100 → 更关注全局 → 可能过度平滑 ✓

补充：500 条还有一个实际原因——可视化可读性。1259 个点的散点图会过度拥挤，500 条已经足够看出聚类趋势。Perplexity 的直觉是"每个点大约看多少个邻居"，30 意味着每个点关注约 30 个最近邻。

---

## 总结

| 维度 | 评价 |
|------|------|
| 代码质量 | ✅ 正确排查了 eager 后端问题，max_iter 兼容性修复 |
| Part 1 注意力 | ✅ 3 张热图保存成功，能看到信号肽区域的注意力集中 |
| Part 2 t-SNE | ✅ Fine-tuned 明显优于 Frozen，验证了迁移学习的价值 |
| Q&A 质量 | ✅ 三个问题全部回答正确 |
| Week 11 收官 | ✅ 从数据→模型→训练→评估→Fine-tuning→可视化，完整闭环 |

---

_Review by Talos | 2026-06-25_
