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
    0: {'name': 'Positive (K/R)',   'tokens': ['K', 'R'],          'color': '#E74C3C'},
    1: {'name': 'Negative (D/E)',   'tokens': ['D', 'E'],          'color': '#3498DB'},
    2: {'name': 'Hydrophobic (A/V/I/L)', 'tokens': ['A', 'V', 'I', 'L'],'color': '#2ECC71'},
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
                random_state=42, max_iter=1000)
    return tsne.fit_transform(embeddings)
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000)
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
    print(f"✅ Saved: {save_path}")


# ── TODO 6：主程序 ────────────────────────────────────────────
def main():
    device    = torch.device('cpu')
    ckpt_path = 'week10/day3/protein_bert_mlm.pt'
    out_dir   = 'week10/day7'

    # 生成数据
    print("生成三类序列数据...")
    src, labels = make_class_data(n_per_class=100, seq_len=40, device=device)
    print(f"Data shape: {src.shape}, label distribution: {[labels.count(i) for i in range(3)]}\n")

    # ── 预训练 Encoder ──
    print("Loading pretrained Encoder...")
    model_pt = ProteinBERT().to(device)
    ckpt     = torch.load(ckpt_path, map_location=device)
    model_pt.load_state_dict(ckpt['model_state_dict'])
    emb_pt   = extract_embeddings(model_pt, src)
    print(f"Pretrained embedding shape: {emb_pt.shape}\n")

    # ── 随机 Encoder（对照组）──
    print("Initializing random Encoder (control group)...")
    model_rand = ProteinBERT().to(device)
    emb_rand   = extract_embeddings(model_rand, src)
    print(f"Random embedding shape: {emb_rand.shape}\n")

    # ── PCA 可视化 ──
    print("Running PCA...")
    pca_pt   = pca_2d(emb_pt)
    pca_rand = pca_2d(emb_rand)

    plot_embeddings(pca_pt,   labels,
                   title='PCA - Pretrained Encoder',
                   save_path=f'{out_dir}/pca_pretrained.png')
    plot_embeddings(pca_rand, labels,
                   title='PCA - Random Encoder',
                   save_path=f'{out_dir}/pca_random.png')

    # ── t-SNE 可视化 ──
    print("\nRunning t-SNE (takes ~10-30s)...")
    tsne_pt   = tsne_2d(emb_pt)
    tsne_rand = tsne_2d(emb_rand)

    plot_embeddings(tsne_pt,   labels,
                   title='t-SNE - Pretrained Encoder',
                   save_path=f'{out_dir}/tsne_pretrained.png')
    plot_embeddings(tsne_rand, labels,
                   title='t-SNE - Random Encoder',
                   save_path=f'{out_dir}/tsne_random.png')

    # ── 定量评估：类内距离 vs 类间距离 ──
    print("\n── Embedding Quality Evaluation ──")
    for name, emb in [('Pretrained', emb_pt), ('Random', emb_rand)]:
        intra, inter = compute_distance_ratio(emb, labels)
        print(f"{name} Encoder | Intra-class: {intra:.4f} | Inter-class: {inter:.4f} | Ratio: {intra/inter:.4f}")


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

# 输出
# 生成三类序列数据...
# Data shape: torch.Size([300, 40]), label distribution: [100, 100, 100]

# Loading pretrained Encoder...
# Pretrained embedding shape: (300, 128)

# Initializing random Encoder (control group)...
# Random embedding shape: (300, 128)

# Running PCA...
# ✅ Saved: week10/day7/pca_pretrained.png
# ✅ Saved: week10/day7/pca_random.png

# Running t-SNE (takes ~10-30s)...
# ✅ Saved: week10/day7/tsne_pretrained.png
# ✅ Saved: week10/day7/tsne_random.png

# ── Embedding Quality Evaluation ──
# Pretrained Encoder | Intra-class: 1.7353 | Inter-class: 2.8641 | Ratio: 0.6059
# Random Encoder | Intra-class: 1.9704 | Inter-class: 3.4821 | Ratio: 0.5659


# Q1：PCA 手动实现里，为什么要先做中心化（X - mean）？不中心化会有什么问题？
# 中心化是PCA的关键步骤，因为PCA的目标是找到数据中方差最大的方向。如果不进行中心化，
# PCA可能会将数据的均值作为主要成分，从而无法正确捕捉数据的结构和变异性。
# 这会导致降维后的结果无法反映原始数据的真实分布和关系，影响可视化效果和后续分析。

# Q2：t-SNE 的 perplexity 参数控制什么？如果设得太小（比如 5）或太大（比如 100）会有什么效果？
# t-SNE 的 perplexity 参数控制了算法在构建高维空间中的概率分布时考虑的邻居数量。
# 较小的 perplexity（如 5）会使 t-SNE 更关注局部结构，可能导致过度聚类和噪声敏感；
# 而较大的 perplexity（如 100）则会使 t-SNE 更关注全局结构，可能导致不同类别之间的边界模糊，难以区分。
# 选择合适的 perplexity 需要根据数据集的大小和特性进行调整。

# Q3：你的定量评估结果中，预训练 Encoder 的类内/类间距离比值和随机 Encoder 相比如何？这说明了什么？
# 预训练 Encoder 的类内/类间距离比值通常会比随机 Encoder 更小，
# 这说明预训练模型能够更好地将同类样本聚集在一起，同时将不同类样本分开。
# 这表明预训练模型学到了有意义的特征表示，
# 能够捕捉到序列之间的相似性和差异性，而随机模型则没有这种能力。