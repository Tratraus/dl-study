# Week 12 · Day 6：注意力权重的生物学解读

## 今天要回答的问题

> 模型在做分类决策时，**"看"了序列的哪些位置**？
> 自实现模型 vs ESM-2，两者的注意力模式有何本质差异？

---

## 理论准备：注意力头的已知模式

文献（Vig & Belinkov, 2019；Rives et al., 2021）发现 Transformer 的注意力头会自发涌现出几类模式：

| 模式类型 | 外观 | 生物学含义 |
|---------|------|-----------|
| **对角线型** | 热图沿主对角线高亮 | 关注相邻氨基酸（局部序列上下文） |
| **列型** | 某几列特别亮 | 特定位置被全局关注（可能是活性位点） |
| **行型** | 某几行特别亮 | 某位置强烈关注全局（信息汇聚点） |
| **块型** | 热图出现矩形块 | 关注特定功能域内部的残基交互 |
| **均匀型** | 热图接近均匀 | 该头尚未学到有意义的模式（常见于从头训练的浅层模型） |

---

## 代码任务

新建 `week12/day6/visualize_attention.py`：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))

from transformer_encoder import TransformerEncoder
from protein_dataset import load_localization_data, LOCALIZATION_CLASSES
from esm2_embed import load_esm2
from day5.train_classifier import ProteinClassifier   # 复用分类模型


# ── 工具函数 ──────────────────────────────────────────────

def get_custom_attn(model, input_ids, mask=None):
    """
    提取自实现模型所有层的注意力权重。
    返回: list of (num_heads, L, L)，取 batch[0]
    """
    model.eval()
    with torch.no_grad():
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.unsqueeze(1).unsqueeze(1).bool()
        _, all_attn = model.encoder(input_ids, attn_mask)
    # all_attn: list of (B, num_heads, L, L)
    return [a[0].cpu().numpy() for a in all_attn]   # list of (num_heads, L, L)


def get_esm2_attn(esm_model, tokenizer, sequence, device):
    """
    提取 ESM-2 所有层的注意力权重。
    返回: list of (num_heads, L, L)
    """
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = esm_model(
            **inputs,
            output_attentions=True
        )
    # outputs.attentions: tuple of (1, num_heads, L, L)
    return [a[0].cpu().numpy() for a in outputs.attentions]


def plot_attention_grid(attn_list, title, seq_short, max_len=60):
    """
    绘制所有层 × 所有头的注意力热图网格。
    attn_list: list[num_layers] of (num_heads, L, L)
    """
    num_layers = len(attn_list)
    num_heads  = attn_list[0].shape[0]
    L          = min(attn_list[0].shape[1], max_len)

    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(num_heads * 2.5, num_layers * 2.5)
    )
    if num_layers == 1:
        axes = axes[np.newaxis, :]
    if num_heads == 1:
        axes = axes[:, np.newaxis]

    for i in range(num_layers):
        for j in range(num_heads):
            ax  = axes[i][j]
            mat = attn_list[i][j, :L, :L]
            im  = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"Head {j+1}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Layer {i+1}", fontsize=9)

    fig.suptitle(f"{title}\n序列前 {L} 个氨基酸", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_mean_attention(attn_list, title, max_len=60):
    """
    对所有头取平均，绘制每层的平均注意力热图（更清晰地看层间演化）。
    """
    num_layers = len(attn_list)
    L = min(attn_list[0].shape[1], max_len)

    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 3, 3))
    if num_layers == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mat = attn_list[i][:, :L, :L].mean(axis=0)   # (L, L)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_title(f"Layer {i+1}", fontsize=10)
        ax.set_xlabel("Key position")
        if i == 0:
            ax.set_ylabel("Query position")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{title} — 各层平均注意力", fontsize=11)
    plt.tight_layout()
    return fig


# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- 选取样本 ----------
    # 各取一条：Nucleus（类别6，最多）和 Peroxisome（类别7，最少）
    sequences, labels = load_localization_data()
    samples = {}
    for target_class in [6, 7]:
        for seq, lbl in zip(sequences, labels):
            if lbl == target_class and len(seq) > 30:
                samples[target_class] = seq[:60]   # 截取前 60 个氨基酸可视化
                break
    print(f"Nucleus    样本: {samples[6]}")
    print(f"Peroxisome 样本: {samples[7]}")

    # ---------- 加载自实现模型 ----------
    tokenizer, esm_model = load_esm2(device)
    VOCAB_SIZE = len(tokenizer)

    custom_model = ProteinClassifier(
        num_classes=10,
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        max_len=512,
        dropout=0.0   # 可视化时关闭 dropout
    ).to(device)
    custom_model.load_state_dict(
        torch.load("week12/day5/best_model.pt", map_location=device)
    )
    custom_model.eval()

    # ---------- 对每个样本可视化 ----------
    for class_id, seq in samples.items():
        class_name = LOCALIZATION_CLASSES[class_id]
        print(f"\n{'='*50}")
        print(f"类别: {class_name} | 序列长度: {len(seq)}")

        # tokenize（复用 ESM-2 tokenizer）
        enc = tokenizer(
            seq, return_tensors="pt",
            padding=False, truncation=True, max_length=512
        ).to(device)
        input_ids = enc["input_ids"]          # (1, L)
        mask      = enc["attention_mask"]     # (1, L)

        # ── 自实现模型注意力 ──
        custom_attn = get_custom_attn(custom_model, input_ids, mask)
        fig1 = plot_attention_grid(
            custom_attn,
            f"自实现 Encoder — {class_name}",
            seq
        )
        fig1.savefig(
            f"week12/day6/custom_attn_{class_name.replace('/', '_')}.png",
            dpi=120, bbox_inches="tight"
        )

        # ── ESM-2 注意力 ──
        esm2_attn = get_esm2_attn(esm_model, tokenizer, seq, device)
        # ESM-2 有 6 层，只取前 3 层对比
        fig2 = plot_attention_grid(
            esm2_attn[:3],
            f"ESM-2（前3层）— {class_name}",
            seq
        )
        fig2.savefig(
            f"week12/day6/esm2_attn_{class_name.replace('/', '_')}.png",
            dpi=120, bbox_inches="tight"
        )

        # ── 平均注意力对比 ──
        fig3, axes = plt.subplots(1, 2, figsize=(10, 3))
        L = min(len(seq), 60)

        # 自实现：3层平均
        for i, ax in enumerate(axes):
            src   = custom_attn if i == 0 else esm2_attn[:3]
            label = "自实现 Encoder" if i == 0 else "ESM-2（前3层）"
            # 所有层所有头的总平均
            mat = np.stack([a[:, :L, :L].mean(axis=0) for a in src]).mean(axis=0)
            ax.imshow(mat, cmap="Blues")
            ax.set_title(f"{label}\n全局平均注意力", fontsize=10)
            ax.set_xlabel("Key position")
            if i == 0:
                ax.set_ylabel("Query position")
            ax.set_xticks([])
            ax.set_yticks([])

        fig3.suptitle(f"{class_name} — 注意力模式对比", fontsize=11)
        plt.tight_layout()
        fig3.savefig(
            f"week12/day6/compare_{class_name.replace('/', '_')}.png",
            dpi=120, bbox_inches="tight"
        )
        plt.show()

    print("\n所有热图已保存至 week12/day6/")
```

---

## 观察框架

运行完成后，对照下表分析你的热图：

```
观察 1：主对角线是否高亮？
  → 是：模型在关注局部邻居（短程依赖）
  → 否：模型在做全局聚合

观察 2：是否有某几列特别亮？
  → 是：找出那几个位置的氨基酸是什么
       （Proline/Cysteine 常是结构关键点）

观察 3：自实现 vs ESM-2 的模式差异？
  → 自实现通常更"均匀"或"对角线"
  → ESM-2 通常有更稀疏、更特异的列模式

观察 4：Nucleus vs Peroxisome 两个类别的模式是否不同？
  → 不同类别的定位信号序列特征不同，
    ESM-2 应该能区分，自实现可能区分不明显
```

---

## 输出问题

**Q1**：观察你的热图——自实现模型的注意力模式和 ESM-2 最显著的视觉差异是什么？用一句话描述。

**Q2**：ESM-2 的注意力热图中，如果某一列特别亮（即某个位置被全局关注），从生物学角度这意味着什么？

**Q3**：注意力权重能直接用来解释"模型为什么做出这个预测"吗？有什么局限性？（提示：思考 attention 和 gradient 的区别）