# Week 10 Day 7：嵌入可视化

## 最小必要理论（10 分钟）

### 1. 为什么要可视化嵌入？

128 维的向量人眼无法直接理解。可视化的目标是回答一个问题：

> **预训练 Encoder 有没有把"相似的序列"映射到"相近的向量"？**

如果预训练有效，同类序列的嵌入应该在空间中聚集成团；如果无效，所有序列的嵌入会随机分布。

---

### 2. 两种降维方法

#### PCA（主成分分析）

```text
原理：找到方差最大的方向，投影到低维空间
特点：线性变换，快速，保留全局结构
适合：快速检查嵌入的整体分布
```

$$z = X W$$

其中 $$W$$ 是前 2 个主成分方向。

#### t-SNE

```text
原理：在低维空间中保持高维空间中的"邻居关系"
特点：非线性，慢，保留局部结构，聚类效果更好
适合：展示聚类结构，发表论文的图
```

t-SNE 的核心思想：

$$p_{ij} \propto \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right) \quad \text{（高维中的相似度）}$$

$$q_{ij} \propto \left(1 + \|z_i - z_j\|^2\right)^{-1} \quad \text{（低维中的相似度）}$$

最小化 $$KL(P \| Q)$$，让低维分布尽量接近高维分布。

今天两种都实现，对比效果。

---

### 3. 今天的实验设计

生成**三类**序列，每类有明确的特征：

```text
类别 0（正电荷）：K/R 比例高（权重 = 5）
类别 1（负电荷）：D/E 比例高（权重 = 5）
类别 2（疏水）  ：A/V/I/L 比例高（权重 = 5）
```

如果预训练 Encoder 学到了氨基酸的化学性质，这三类序列的嵌入应该在 2D 空间中形成三个可分离的簇。

---

### 4. 预期结果 vs 实际结果

由于我们的预训练数据是**随机均匀序列**，Encoder 没有学到真实的氨基酸化学性质，所以：

- **预期**：三类簇的分离程度有限，但应该比随机初始化的 Encoder 稍好
- **对照**：随机初始化的 Encoder 的嵌入应该完全混在一起

这个对比本身就是今天的核心结论。

---

## 代码任务

新建 `week10/day7/visualize.py`：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os, random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))

from mlm_data import AA_IDS, token2id
from protein_bert import ProteinBERT
from extract_embedding import mean_pooling


# ── 常量 ──────────────────────────────────────────────────────
CLASS_CONFIGS = {
    0: {'name': '正电荷 (K/R)',  'tokens': ['K', 'R'],          'color': '#E74C3C'},
    1: {'name': '负电荷 (D/E)',  'tokens': ['D', 'E'],          'color': '#3498DB'},
    2: {'name': '疏水 (A/V/I/L)','tokens': ['A', 'V', 'I', 'L'],'color': '#2ECC71'},
}


# ── TODO 1：生成三类序列数据 ──────────────────────────────────
def make_class_data(
    n_per_class: int = 100,
    seq_len:     int = 40,
    device:      torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, list[int]]:
    """
    生成三类序列及其标签。

    对每个类别：
      - 目标 token 的采样权重设为 5，其余为 1
      - 生成 n_per_class 条序列

    返回：
      src:    (n_per_class * 3, seq_len)  dtype=torch.long
      labels: list[int]，长度 n_per_class * 3，值为 0/1/2
    """
    all_seqs   = []
    all_labels = []

    for cls_id, cfg in CLASS_CONFIGS.items():
        target_ids = {token2id[aa] for aa in cfg['tokens']}
        weights    = [5 if i in target_ids else 1 for i in AA_IDS]

        for _ in range(n_per_class):
            seq = random.choices(AA_IDS, weights=weights, k=seq_len)
            all_seqs.append(seq)
            all_labels.append(cls_id)

    # 打乱顺序
    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)

    src = torch.tensor(list(all_seqs), dtype=torch.long, device=device)
    return src, list(all_labels)


# ── TODO 2：提取嵌入 ──────────────────────────────────────────
def extract_embeddings(
    model:  ProteinBERT,
    src:    torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    """
    分批提取嵌入，返回 numpy 数组。

    步骤：
      1. 将 src 按 batch_size 分批
      2. 每批调用 mean_pooling，收集结果
      3. 拼接后转为 numpy，shape: (N, d_model)

    注意：
      - model.eval() + torch.no_grad()
      - 分批处理避免大 batch 导致内存问题
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, src.size(0), batch_size):
            batch = src[i : i + batch_size]
            emb   = mean_pooling(model, batch)   # (batch, d_model)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


# ── TODO 3：PCA 降维 ──────────────────────────────────────────
def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """
    用 PCA 将嵌入降到 2 维。

    不使用 sklearn，手动实现：
      1. 中心化：X = embeddings - mean
      2. 计算协方差矩阵：C = X.T @ X / (N-1)
      3. 特征值分解：np.linalg.eigh(C)
      4. 取最大的 2 个特征向量
      5. 投影：X @ top2_eigenvectors

    返回：(N, 2)
    """
    X = embeddings - embeddings.mean(axis=0)
    C = X.T @ X / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # eigh 返回升序排列，取最后 2 列（最大特征值）
    top2 = eigenvectors[:, -2:][:, ::-1]   # (d_model, 2)，降序
    return X @ top2                          # (N, 2)


# ── TODO 4：t-SNE 降维 ────────────────────────────────────────
def tsne_2d(embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """
    用 sklearn 的 t-SNE 将嵌入降到 2 维。

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000)
    return tsne.fit_transform(embeddings)
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000)
    return tsne.fit_transform(embeddings)


# ── TODO 5：绘图函数 ──────────────────────────────────────────
def plot_embeddings(
    coords_2d:   np.ndarray,
    labels:      list[int],
    title:       str,
    save_path:   str,
):
    """
    绘制 2D 散点图，按类别着色。

    要求：
      - 每个类别用 CLASS_CONFIGS 中定义的颜色
      - 图例显示类别名称
      - 标题使用 title 参数
      - 保存到 save_path
      - 点的透明度 alpha=0.6，大小 s=20
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls_id, cfg in CLASS_CONFIGS.items():
        mask = [i for i, l in enumerate(labels) if l == cls_id]
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=cfg['color'],
            label=cfg['name'],
            alpha=0.6,
            s=20,
        )

    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"✅ 图片已保存：{save_path}")


# ── TODO 6：主程序 ────────────────────────────────────────────
def main():
    device    = torch.device('cpu')
    ckpt_path = 'week10/day3/protein_bert_mlm.pt'
    out_dir   = 'week10/day7'

    # 生成数据
    print("生成三类序列数据...")
    src, labels = make_class_data(n_per_class=100, seq_len=40, device=device)
    print(f"数据 shape: {src.shape}，标签分布: {[labels.count(i) for i in range(3)]}\n")

    # ── 预训练 Encoder ──
    print("加载预训练 Encoder...")
    model_pt = ProteinBERT().to(device)
    ckpt     = torch.load(ckpt_path, map_location=device)
    model_pt.load_state_dict(ckpt['model_state_dict'])
    emb_pt   = extract_embeddings(model_pt, src)
    print(f"预训练嵌入 shape: {emb_pt.shape}\n")

    # ── 随机 Encoder（对照组）──
    print("初始化随机 Encoder（对照组）...")
    model_rand = ProteinBERT().to(device)
    emb_rand   = extract_embeddings(model_rand, src)
    print(f"随机嵌入 shape: {emb_rand.shape}\n")

    # ── PCA 可视化 ──
    print("PCA 降维中...")
    pca_pt   = pca_2d(emb_pt)
    pca_rand = pca_2d(emb_rand)

    plot_embeddings(pca_pt,   labels,
                   title='PCA - 预训练 Encoder',
                   save_path=f'{out_dir}/pca_pretrained.png')
    plot_embeddings(pca_rand, labels,
                   title='PCA - 随机 Encoder',
                   save_path=f'{out_dir}/pca_random.png')

    # ── t-SNE 可视化 ──
    print("\nt-SNE 降维中（需要约 10~30 秒）...")
    tsne_pt   = tsne_2d(emb_pt)
    tsne_rand = tsne_2d(emb_rand)

    plot_embeddings(tsne_pt,   labels,
                   title='t-SNE - 预训练 Encoder',
                   save_path=f'{out_dir}/tsne_pretrained.png')
    plot_embeddings(tsne_rand, labels,
                   title='t-SNE - 随机 Encoder',
                   save_path=f'{out_dir}/tsne_random.png')

    # ── 定量评估：类内距离 vs 类间距离 ──
    print("\n── 嵌入质量定量评估 ──")
    for name, emb in [('预训练', emb_pt), ('随机', emb_rand)]:
        intra, inter = compute_distance_ratio(emb, labels)
        print(f"{name} Encoder | 类内距离: {intra:.4f} | 类间距离: {inter:.4f} | 比值: {intra/inter:.4f}")


# ── TODO 7：定量评估 ──────────────────────────────────────────
def compute_distance_ratio(
    embeddings: np.ndarray,
    labels:     list[int],
) -> tuple[float, float]:
    """
    计算类内平均距离和类间平均距离。

    类内距离：同类样本嵌入之间的平均 L2 距离
    类间距离：不同类样本嵌入之间的平均 L2 距离

    返回：(intra_dist, inter_dist)

    提示：
      - 可以用随机采样（每类采 50 对）来加速计算，避免 O(N²)
      - np.linalg.norm(a - b)
    """
    n_samples = 50   # 每类随机采样的对数

    class_ids  = list(CLASS_CONFIGS.keys())
    class_embs = {
        c: embeddings[[i for i, l in enumerate(labels) if l == c]]
        for c in class_ids
    }

    # 类内距离
    intra_dists = []
    for c in class_ids:
        embs = class_embs[c]
        for _ in range(n_samples):
            i, j = random.sample(range(len(embs)), 2)
            intra_dists.append(np.linalg.norm(embs[i] - embs[j]))

    # 类间距离
    inter_dists = []
    for i in range(len(class_ids)):
        for j in range(i + 1, len(class_ids)):
            embs_i = class_embs[class_ids[i]]
            embs_j = class_embs[class_ids[j]]
            for _ in range(n_samples):
                a = embs_i[random.randint(0, len(embs_i) - 1)]
                b = embs_j[random.randint(0, len(embs_j) - 1)]
                inter_dists.append(np.linalg.norm(a - b))

    return np.mean(intra_dists), np.mean(inter_dists)


if __name__ == "__main__":
    main()
```

---

## 完成标准

1. 四张图片生成成功（PCA × 2，t-SNE × 2）
2. 定量评估输出类内/类间距离比值
3. 观察并描述：预训练 vs 随机 Encoder 的聚类效果差异

---

## 输出问题

**Q1**：PCA 手动实现里，为什么要先做中心化（`X - mean`）？不中心化会有什么问题？

**Q2**：t-SNE 的 `perplexity` 参数控制什么？如果设得太小（比如 5）或太大（比如 100）会有什么效果？

**Q3**：你的定量评估结果中，预训练 Encoder 的类内/类间距离比值和随机 Encoder 相比如何？这说明了什么？

---

准备好后提交代码、四张图片和三个问题的回答。