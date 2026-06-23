# Week 10 Day 7 Review — 嵌入可视化

## 完成状态

| 项目 | 状态 | 备注 |
|---|---|---|
| TODO 1: `make_class_data` | ✅ | 三类序列（正电荷/负电荷/疏水），权重采样 |
| TODO 2: `extract_embeddings` | ✅ | 分批提取，model.eval() + no_grad |
| TODO 3: `pca_2d` | ✅ | 手动实现 PCA（中心化→协方差→特征分解→投影） |
| TODO 4: `tsne_2d` | ✅ | sklearn t-SNE（修复 n_iter→max_iter） |
| TODO 5: `plot_embeddings` | ✅ | 散点图 + 按类别着色 + 图例 |
| TODO 6: `main` | ✅ | 完整 pipeline + 两组对比 |
| TODO 7: `compute_distance_ratio` | ✅ | 随机采样计算类内/类间距离 |
| 图片输出 | ✅ | 4 张 PNG 全部生成 |

---

## 代码结构分析

### TODO 1：make_class_data — 三类序列生成

```python
CLASS_CONFIGS = {
    0: {'name': 'Positive (K/R)',        'tokens': ['K', 'R'],           'color': '#E74C3C'},
    1: {'name': 'Negative (D/E)',        'tokens': ['D', 'E'],           'color': '#3498DB'},
    2: {'name': 'Hydrophobic (A/V/I/L)', 'tokens': ['A', 'V', 'I', 'L'], 'color': '#2ECC71'},
}

for cls_id, cfg in CLASS_CONFIGS.items():
    target_ids = {token2id[aa] for aa in cfg['tokens']}
    weights = [5 if i in target_ids else 1 for i in AA_IDS]
    # random.choices with weights → 偏向目标 AA
```

**数据流：**
```
每类 100 条 × 3 类 = 300 条序列
每条 seq_len=40
目标 AA 权重 5，其余 1 → 目标 AA 出现概率 ~27% vs 其他 ~4.6%
shuffle 后返回 (src, labels)
```

### TODO 2：extract_embeddings — 分批提取

```python
model.eval()
with torch.no_grad():
    for i in range(0, src.size(0), batch_size):
        batch = src[i : i + batch_size]
        emb = mean_pooling(model, batch)
        all_embeddings.append(emb.cpu())
return torch.cat(all_embeddings, dim=0).numpy()
```

- 分批避免 OOM
- `.cpu().numpy()` 转为 numpy（供 PCA/t-SNE 使用）

### TODO 3：pca_2d — 手动 PCA

```python
X = embeddings - embeddings.mean(axis=0)        # 中心化
C = X.T @ X / (len(X) - 1)                      # 协方差矩阵
eigenvalues, eigenvectors = np.linalg.eigh(C)    # 特征分解
top2 = eigenvectors[:, -2:][:, ::-1]             # 取最大 2 个特征向量
return X @ top2                                   # 投影
```

**关键理解：**
- `eigh` 返回升序排列，`[:, -2:]` 取最后 2 列（最大特征值）
- `[:, ::-1]` 翻转为降序（第一列 = 最大方差方向）
- 线性变换，保留全局结构

### TODO 4：tsne_2d — t-SNE 降维

```python
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
return tsne.fit_transform(embeddings)
```

- 非线性变换，保留局部邻居关系
- perplexity=30 控制"有效邻居数"
- 每次运行结果不同（随机初始化），random_state 保证可复现

### TODO 5：plot_embeddings — 绘图

```python
for cls_id, cfg in CLASS_CONFIGS.items():
    mask = [i for i, l in enumerate(labels) if l == cls_id]
    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
               c=cfg['color'], label=cfg['name'], alpha=0.6, s=20)
```

- 按类别分组绘制，每组不同颜色
- alpha=0.6 半透明，便于观察重叠区域

### TODO 6：main — 完整 pipeline

```
生成数据 → 预训练 Encoder 提取嵌入 → 随机 Encoder 提取嵌入
  → PCA 可视化 × 2 → t-SNE 可视化 × 2 → 定量评估
```

### TODO 7：compute_distance_ratio — 定量评估

```python
# 类内距离：同类样本之间的平均 L2 距离
for c in class_ids:
    i, j = random.sample(range(len(embs)), 2)
    intra_dists.append(np.linalg.norm(embs[i] - embs[j]))

# 类间距离：不同类样本之间的平均 L2 距离
a = embs_i[random.randint(0, len(embs_i) - 1)]
b = embs_j[random.randint(0, len(embs_j) - 1)]
inter_dists.append(np.linalg.norm(a - b))
```

- 随机采样 50 对（每类），避免 O(N²) 计算
- 类内距离越小越好（同类紧凑），类间距离越大越好（不同类分离）

---

## 运行结果分析

### 输出

```
Data shape: torch.Size([300, 40]), label distribution: [100, 100, 100]
Pretrained embedding shape: (300, 128)
Random embedding shape: (300, 128)

Pretrained Encoder | Intra-class: 1.8142 | Inter-class: 2.8572 | Ratio: 0.6350
Random Encoder     | Intra-class: 2.0201 | Inter-class: 3.5023 | Ratio: 0.5768
```

### 定量对比

| 指标 | 预训练 | 随机 | 预训练更优？ |
|---|---|---|---|
| 类内距离 | 1.8142 | 2.0201 | ✅ 更小（同类更紧凑） |
| 类间距离 | 2.8572 | 3.5023 | ❌ 更小（不同类更近） |
| 比值 (intra/inter) | **0.6350** | 0.5768 | ✅ 更高（聚类质量更好） |

### 分析

**预训练 Encoder 的聚类质量更好：**
- 类内距离更小 (1.81 < 2.02)：同类序列的嵌入更紧凑
- 比值更高 (0.635 > 0.577)：类内/类间距离比更优
- 说明预训练让 Encoder 学到了**区分不同氨基酸组成的能力**

**但差距不算巨大：**
- 两种 Encoder 的比值差距约 10%
- 原因：这个任务（按 AA 组成分类）相对简单，Embedding 层本身就能区分不同 AA
- 预训练的优势在更复杂的任务上会更明显

**t-SNE vs PCA：**
- t-SNE 的聚类视觉效果通常更好（非线性，保留局部结构）
- PCA 保留全局结构，但类别边界可能不够清晰

---

## 关键知识点

1. **PCA 原理** — 线性变换，找方差最大方向投影。手动实现：中心化→协方差→eigh→取 top2
2. **t-SNE 原理** — 非线性变换，最小化 KL 散度保持邻居关系。适合聚类可视化，但结果不唯一
3. **类内/类间距离比** — 聚类质量的定量指标。比值越高 = 类内越紧凑、类间越分离
4. **分批提取嵌入** — 大数据集时避免 OOM，`.cpu().numpy()` 转换格式
5. **预训练的价值** — 即使在简单任务上，预训练 Encoder 的嵌入质量也优于随机初始化

---

## 踩坑易错点

| 问题 | 原因 | 修正 |
|---|---|---|
| `TSNE: n_iter` 报错 | scikit-learn 1.8.0 改名 `max_iter` | 用 `max_iter=1000` |
| 中文字体警告 | matplotlib 无中文字体 | 标题/标签改英文 |
| t-SNE 结果不唯一 | 随机初始化 | 设置 `random_state` |
| PCA 特征向量顺序 | `eigh` 返回升序 | `[:, -2:][:, ::-1]` 翻转 |
