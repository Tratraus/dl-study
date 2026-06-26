# Week 11 · Day 7：注意力可视化 + t-SNE 收官

## 今日目标

两件事，形成 Week 11 的完整闭环：

1. **注意力权重可视化**：ESM-2 在预测亚细胞定位时，"看"了序列的哪些位置？
2. **Fine-tuned 嵌入 t-SNE**：真实蛋白质的嵌入空间长什么样？与 Week 10 随机 Encoder 对比

---

## Part 1 理论：注意力权重是什么

ESM-2 是 Transformer Encoder，每一层有多个注意力头，每个头会对序列中每个位置计算一个**注意力分布**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$ 就是注意力权重矩阵，shape 为 `(L, L)`，表示：

> 位置 `i` 在生成输出时，对位置 `j` 的关注程度

HuggingFace 的 ESM-2 支持直接输出注意力权重，只需在调用时加一个参数：

```python
outputs = esm_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True   # ← 加这一行
)
# outputs.attentions: tuple，长度 = 层数
# 每个元素 shape: (B, num_heads, L, L)
```

---

## Part 2 理论：为什么要做 t-SNE

Week 10 的结论是：

> 合成随机序列上，预训练 Encoder 和随机 Encoder 的嵌入空间**视觉上几乎一样好**

今天用**真实蛋白质**做同样的实验，预期结论会反转：

| 实验 | 预期 |
|------|------|
| Week 10 随机序列 t-SNE | 预训练 ≈ 随机，因为序列没有真实规律 |
| Week 11 真实蛋白质 t-SNE | Fine-tuned > Frozen > 随机，因为真实序列有生物学信号 |

---

## 代码任务

新建 `week11/day7/visualize.py`：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day3'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day5'))
from esm2_embed import load_esm2
from protein_dataset import (
    load_localization_data, split_dataset,
    make_collate_fn, mean_pooling_with_mask,
    LOCALIZATION_CLASSES
)
from train_classifier import ProteinClassifier
from finetune import unfreeze_last_n_layers


# ══════════════════════════════════════════════════════════
# Part 1：注意力权重可视化
# ══════════════════════════════════════════════════════════

def get_attention_map(
    esm_model,
    tokenizer,
    sequence: str,
    device:   torch.device,
    layer:    int = -1,   # 默认取最后一层
) -> tuple[np.ndarray, list[str]]:
    """
    对单条蛋白质序列提取注意力权重。

    返回：
      attn_map : np.ndarray, shape (L, L)
                 对所有注意力头取平均后的注意力矩阵
      tokens   : list[str]，对应每个位置的 token 字符串

    步骤：
      1. tokenize 单条序列（加空格分隔，ESM-2 的输入格式）
      2. 前向传播，output_attentions=True
      3. 取 outputs.attentions[layer]，shape (1, num_heads, L, L)
      4. squeeze batch 维度，对 num_heads 取平均 → (L, L)
      5. 去掉首尾的 <cls> 和 <eos> token（只保留氨基酸位置）
    """
    esm_model.eval()
    # ESM-2 tokenizer 要求氨基酸之间加空格
    spaced = " ".join(list(sequence))
    inputs = tokenizer(spaced, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = esm_model(
            **inputs,
            output_attentions=True
        )

    # outputs.attentions: tuple of (1, num_heads, L, L)
    attn = outputs.attentions[layer]   # (1, num_heads, L, L)
    attn = attn.squeeze(0)             # (num_heads, L, L)
    attn = attn.mean(dim=0)            # (L, L)，对所有头取平均
    attn = attn.cpu().numpy()

    # 获取 token 字符串
    token_ids = inputs["input_ids"].squeeze(0).tolist()
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)

    # 去掉首尾特殊 token（<cls> 和 <eos>）
    attn   = attn[1:-1, 1:-1]
    tokens = tokens[1:-1]

    return attn, tokens


def plot_attention(
    attn_map: np.ndarray,
    tokens:   list[str],
    title:    str = "Attention Map",
    save_path: str = "week11/day7/attention.png",
    max_len:  int = 50,   # 序列太长时截断显示
) -> None:
    """
    画注意力热图。
    如果序列长度 > max_len，只显示前 max_len 个位置。
    """
    if len(tokens) > max_len:
        attn_map = attn_map[:max_len, :max_len]
        tokens   = tokens[:max_len]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticklabels(tokens, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"注意力图已保存：{save_path}")


# ══════════════════════════════════════════════════════════
# Part 2：t-SNE 嵌入可视化
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(
    esm_model,
    classifier,   # 传入但不使用，只用 ESM-2 的嵌入
    loader,
    device,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    提取测试集前 max_samples 条的 ESM-2 嵌入向量。
    返回 (embeddings, labels)，shape 分别为 (N, d_model) 和 (N,)
    """
    esm_model.eval()
    all_embs, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        if sum(len(e) for e in all_embs) >= max_samples:
            break
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs    = esm_model(input_ids=input_ids,
                               attention_mask=attention_mask)
        embeddings = mean_pooling_with_mask(
            outputs.last_hidden_state, attention_mask)

        all_embs.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    embs   = np.concatenate(all_embs,   axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    return embs, labels


def plot_tsne(
    embeddings:  np.ndarray,
    labels:      np.ndarray,
    class_names: dict,
    title:       str = "t-SNE",
    save_path:   str = "week11/day7/tsne.png",
) -> None:
    """
    用 sklearn 的 TSNE 对嵌入降维到 2D，画散点图。
    每个类别用不同颜色。
    """
    from sklearn.manifold import TSNE

    print(f"  运行 t-SNE（{len(embeddings)} 条样本）...")
    tsne   = TSNE(n_components=2, random_state=42,
                  perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for c, name in class_names.items():
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[c]], label=name,
                   alpha=0.6, s=20)

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=8, markerscale=2)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  t-SNE 图已保存：{save_path}")


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 加载数据 ──────────────────────────────────────────
    sequences, labels = load_localization_data()
    _, _, test_ds = split_dataset(sequences, labels, seed=42)

    # ── 加载 Fine-tuned 模型 ──────────────────────────────
    tokenizer, esm_model = load_esm2(device=device)
    unfreeze_last_n_layers(esm_model, n=2)
    classifier = ProteinClassifier().to(device)

    ckpt_path = os.path.join(
        os.path.dirname(__file__), '..', 'day5', 'best_finetune.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    esm_model.load_state_dict(ckpt['esm_state'])
    classifier.load_state_dict(ckpt['clf_state'])
    print("已加载 Fine-tuned 模型 ✓")

    # ══════════════════════════════════════════════════════
    # Part 1：注意力可视化
    # 选取 3 条有代表性的序列（不同类别）
    # ══════════════════════════════════════════════════════
    print("\n── Part 1：注意力权重可视化 ──")

    # 从测试集中各取一条：Extracellular(9)、Nucleus(6)、Mitochondria(5)
    target_classes = {9: "Extracellular", 6: "Nucleus", 5: "Mitochondria"}
    selected = {}
    for seq, lbl in zip(test_ds.sequences, test_ds.labels):
        if lbl in target_classes and lbl not in selected:
            selected[lbl] = seq
        if len(selected) == len(target_classes):
            break

    for lbl, seq in selected.items():
        class_name = target_classes[lbl]
        print(f"  处理 {class_name}，序列长度 {len(seq)}")
        attn_map, tokens = get_attention_map(
            esm_model, tokenizer, seq, device, layer=-1)
        plot_attention(
            attn_map, tokens,
            title=f"ESM-2 Attention – {class_name} (last layer, avg heads)",
            save_path=f"week11/day7/attention_{class_name.lower()}.png"
        )

    # ══════════════════════════════════════════════════════
    # Part 2：t-SNE 对比
    # Frozen ESM-2 vs Fine-tuned ESM-2
    # ══════════════════════════════════════════════════════
    print("\n── Part 2：t-SNE 嵌入可视化 ──")

    collate_fn  = make_collate_fn(tokenizer)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, collate_fn=collate_fn)

    # 2-a：Fine-tuned ESM-2 的嵌入
    print("提取 Fine-tuned 嵌入...")
    embs_ft, lbls = extract_embeddings(
        esm_model, classifier, test_loader, device, max_samples=500)
    plot_tsne(
        embs_ft, lbls, LOCALIZATION_CLASSES,
        title="t-SNE · Fine-tuned ESM-2 (test set, 500 samples)",
        save_path="week11/day7/tsne_finetuned.png"
    )

    # 2-b：Frozen ESM-2（重新加载原始权重）
    print("重新加载 Frozen ESM-2 权重...")
    _, esm_frozen = load_esm2(device=device)
    for p in esm_frozen.parameters():
        p.requires_grad = False

    print("提取 Frozen 嵌入...")
    embs_frozen, _ = extract_embeddings(
        esm_frozen, None, test_loader, device, max_samples=500)
    plot_tsne(
        embs_frozen, lbls, LOCALIZATION_CLASSES,
        title="t-SNE · Frozen ESM-2 (test set, 500 samples)",
        save_path="week11/day7/tsne_frozen.png"
    )

    print("\n── Week 11 Day 7 完成 ──")
    print("生成文件：")
    print("  week11/day7/attention_extracellular.png")
    print("  week11/day7/attention_nucleus.png")
    print("  week11/day7/attention_mitochondria.png")
    print("  week11/day7/tsne_finetuned.png")
    print("  week11/day7/tsne_frozen.png")


if __name__ == "__main__":
    main()
```

---

## 完成标准

| 检查项 | 预期 |
|--------|------|
| 3 张注意力热图 | 成功保存，能看到颜色分布 |
| Fine-tuned t-SNE | 10 个类别有一定分离趋势 |
| Frozen t-SNE | 分离程度**弱于** Fine-tuned |

---

## 输出问题

**Q1**：注意力热图中，对角线（位置 `i` 关注自身）通常比较亮，这说明什么？如果某个区域有一片连续的亮色块，可能代表什么生物学结构？

**Q2**：Fine-tuned 和 Frozen 的 t-SNE 图，哪些类别分离得最好？哪些类别最容易混在一起？与 Day4/Day6 的混淆矩阵结论是否一致？

**Q3**：t-SNE 只取了 500 条样本，而不是全部 1259 条，原因是什么？如果把 `perplexity` 从 30 改成 5 或 100，图会有什么变化？

---

提交：5 张图片 + 三个问题的回答，Week 11 正式收官。