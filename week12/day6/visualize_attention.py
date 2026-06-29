import torch
import torch.nn as nn
import numpy as np
import sys, os
import matplotlib
import matplotlib.font_manager as fm

# ✅ 直接用文件路径注册，绕过字体名称查找
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day5'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
from transformer_encoder import TransformerEncoder
from protein_dataset import load_localization_data, LOCALIZATION_CLASSES
from esm2_embed import load_esm2
from train_classifier import ProteinClassifier   # 复用分类模型


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
    tokenizer, esm_model = load_esm2(device=device)

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

# 输出
# Device: cuda
# 加载 ProtST-SubcellularLocalization 数据集...
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# 过滤后数据集大小：8388
#   类别 0（Cell membrane            ）： 800 条
#   类别 1（Cytoplasm                ）：1635 条
#   类别 2（Endoplasmic reticulum    ）： 516 条
#   类别 3（Golgi apparatus          ）： 214 条
#   类别 4（Lysosome/Vacuole         ）： 192 条
#   类别 5（Mitochondria             ）： 906 条
#   类别 6（Nucleus                  ）：2424 条
#   类别 7（Peroxisome               ）：  93 条
#   类别 8（Plastid                  ）： 453 条
#   类别 9（Extracellular            ）：1155 条
# Nucleus    样本: MSGEGNLGKDHEEENEAPLPGFRFHPTDEELLGYYLRRKVENKTIKLELIKQIDIYKYDP
# Peroxisome 样本: MAIPEEFDVIVCGGGSTGCVIAGRLANVDENLKVLLIENGENNLNNPWVYLPGIYPRNMR
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1896.12it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     |
# --------------------------+------------+-
# lm_head.dense.bias        | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.bias   | UNEXPECTED |
# pooler.dense.bias         | MISSING    |
# pooler.dense.weight       | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

# ==================================================
# 类别: Nucleus | 序列长度: 60

# ==================================================
# 类别: Peroxisome | 序列长度: 60

# 所有热图已保存至 week12/day6/

# Q1：观察你的热图——自实现模型的注意力模式和 ESM-2 最显著的视觉差异是什么？用一句话描述。
# ESM-2的注意力模式更集中，尤其在某些位置上表现出明显的全局关注，而自实现模型的注意力模式更分散。

# Q2：ESM-2 的注意力热图中，如果某一列特别亮（即某个位置被全局关注），从生物学角度这意味着什么？
# 这意味着该位置的氨基酸可能在蛋白质的结构或功能中具有重要作用，
# 可能是关键的活性位点、结合位点或结构稳定性相关的残基，因此模型在预测时给予了更多关注。

# Q3：注意力权重能直接用来解释"模型为什么做出这个预测"吗？有什么局限性？（提示：思考 attention 和 gradient 的区别）
# 注意力权重可以提供模型在处理输入时关注的区域，但它们并不直接等同于模型的决策依据。局限性包括：
# 1. 注意力权重只是模型内部的一种机制，不能完全反映模型的决策过程。
# 2. 高注意力权重不一定意味着该位置对最终预测结果有决定性影响，可能只是模型在特定层次上的关注。
# 3. 注意力权重可能受到模型结构和训练方式的影响，可能不完全反映生物学上的重要性。