import torch
import torch.nn as nn
import numpy as np
from collections import Counter

def compute_pos_weight(label_counts: dict, N: int, cap: float = None) -> torch.Tensor:
    """
    根据Top-K标签的正样本频次，计算BCEWithLogitsLoss所需的pos_weight向量。

    Args:
        label_counts: {label_name: 正样本出现次数}，按固定顺序排列
        N: 总样本数（过滤后的样本数，你的场景中是1213）
        cap: 可选，对pos_weight设置上限，防止极端稀有标签权重过大导致训练不稳定
    """
    weights = []
    for label, n_pos in label_counts.items():
        n_neg = N - n_pos
        w = n_neg / n_pos
        if cap is not None:
            w = min(w, cap)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(pos_weight: torch.Tensor) -> nn.Module:
    """
    正确的多标签损失函数构造方式：
    - 不在模型最后一层加sigmoid
    - 直接用BCEWithLogitsLoss处理原始logits
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


if __name__ == "__main__":
    # 用你Day1的真实数据做sanity check
    N = 1213
    top50_counts = {
    "GO:0042802": 238,   "GO:0008270": 167,   "GO:0003723": 137,
    "GO:0000981": 130,   "GO:0046872": 121,   "GO:0000978": 115,
    "GO:0042803": 104,   "GO:1990837": 96,    "GO:0003677": 92,
    "GO:0001228": 75,    "GO:0005524": 75,    "GO:0005509": 72,
    "GO:0005525": 61,    "GO:0019899": 57,    "GO:0003924": 54,
    "GO:0003700": 53,    "GO:0005102": 50,    "GO:0005125": 49,
    "GO:0046982": 49,    "GO:0003682": 46,    "GO:0008083": 41,
    "GO:0019901": 41,    "GO:0044877": 40,    "GO:0000977": 40,
    "GO:0031625": 38,    "GO:0061630": 35,    "GO:0019904": 34,
    "GO:0004930": 33,    "GO:0043565": 33,    "GO:0003713": 32,
    "GO:0004888": 31,    "GO:0061629": 31,    "GO:0048018": 30,
    "GO:0045296": 29,    "GO:0001227": 28,    "GO:0003779": 28,
    "GO:0030674": 27,    "GO:0000976": 27,    "GO:0008201": 26,
    "GO:0000287": 25,    "GO:0005179": 25,    "GO:0003735": 25,
    "GO:0003925": 25,    "GO:0003729": 24,    "GO:0046983": 24,
    "GO:0005198": 23,    "GO:0030527": 23,    "GO:0038023": 22,
    "GO:0004252": 22,    "GO:0051087": 21,
    }
    pos_weight = compute_pos_weight(top50_counts, N, cap=50.0)
    print("pos_weight:", pos_weight)

    loss_fn = build_loss_fn(pos_weight)

    # 模拟一个batch：batch_size=4, num_labels=2
    batch_size = 4
    num_labels = len(top50_counts)

    logits = torch.randn(batch_size, num_labels, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_labels), dtype=torch.float32)

    loss = loss_fn(logits, targets)
    loss.backward()

    print("loss:", loss.item())
    print("grad存在且非NaN:", not torch.isnan(logits.grad).any().item())
