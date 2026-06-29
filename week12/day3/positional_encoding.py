import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构建位置编码矩阵 PE: (max_len, d_model)
        PE = torch.zeros(max_len, d_model)

        # pos: (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term: (d_model/2,)
        # 公式：10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        i = torch.arange(0, d_model, 2).float()           # 偶数索引：0,2,4,...
        div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数维用 sin，奇数维用 cos
        PE[:, 0::2] = torch.sin(pos * div_term)   # sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)   # cos(pos * div_term)

        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        # 增加 batch 维：(1, max_len, d_model)
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
          x: (B, L, d_model)  ← 输入嵌入
        返回：
          x + PE[:, :L, :]    ← 加上位置编码后的嵌入，shape 不变
        """
        L = x.shape[1]
        x = x + self.PE[:, :L, :]           # 取 PE 的前 L 个位置
        return self.dropout(x)


# ── 验证 ──────────────────────────────────────────────────
if __name__ == "__main__":
    d_model = 64
    max_len = 100

    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # 1. shape 验证
    x = torch.zeros(2, 50, d_model)
    out = pe(x)
    print(f"输入  shape: {x.shape}")
    print(f"输出  shape: {out.shape}")   # 期望: (2, 50, 64)

    # 2. 验证：不同位置的 PE 向量不同
    pe_matrix = pe.PE.squeeze(0)         # (max_len, d_model)
    print(f"\n位置 0 和位置 1 的 PE 是否相同: {torch.allclose(pe_matrix[0], pe_matrix[1])}")
    print(f"位置 0 和位置 0 的 PE 是否相同: {torch.allclose(pe_matrix[0], pe_matrix[0])}")

    # 3. 验证：打乱顺序后输出不同
    torch.manual_seed(0)
    x_ordered   = torch.randn(1, 5, d_model)
    x_shuffled  = x_ordered[:, [2, 0, 4, 1, 3], :]   # 打乱顺序

    out_ordered  = pe(x_ordered)
    out_shuffled = pe(x_shuffled)
    print(f"\n加位置编码后，打乱顺序的输出 ≠ 原顺序: {not torch.allclose(out_ordered, out_shuffled)}")

    # 4. 热图可视化
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        pe_matrix.numpy(),
        aspect='auto',
        cmap='RdBu',
        vmin=-1, vmax=1
    )
    ax.set_xlabel("Dimension index (d_model)")
    ax.set_ylabel("Position (pos)")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("week12/day3/pe_heatmap.png", dpi=150)
    plt.show()
    print("\n热图已保存为 week12/day3/pe_heatmap.png")

# 输出
# 输入  shape: torch.Size([2, 50, 64])
# 输出  shape: torch.Size([2, 50, 64])

# 位置 0 和位置 1 的 PE 是否相同: False
# 位置 0 和位置 0 的 PE 是否相同: True

# 加位置编码后，打乱顺序的输出 ≠ 原顺序: True

# Q1：register_buffer 和直接把 PE 存成 self.PE 有什么区别？为什么位置编码要用 register_buffer？
# register_buffer 会把 PE 注册为模型的 buffer，这样在调用 model.eval() 时，PE 不会被更新，也不会参与梯度计算。
# 同时，注册的 buffer 会随模型保存和加载，而直接存成 self.PE 则不会有这些特性。
# 位置编码是固定的，不需要训练，所以使用 register_buffer 更合适。

# Q2：热图中，左侧（低维度）的条纹比右侧（高维度）宽，这对应公式里的哪个部分？直觉上代表什么？
# 热图中左侧（低维度）的条纹宽，右侧（高维度）的条纹窄，这对应公式中的 div_term 部分。
# div_term 控制了不同维度的频率，低维度的频率较低，变化较慢，因此条纹宽；高维度的频率较高，变化较快，因此条纹窄。
# 直觉上，这意味着低维度的编码更平滑，而高维度的编码更敏感于位置变化。

# Q3：ESM-2 的可学习位置编码最大长度是 1026（1024 + 2个特殊 token）。如果你要处理一条长度 2000 的蛋白质序列，用 ESM-2 会遇到什么问题？用正弦编码会遇到同样的问题吗？
# ESM-2 的可学习位置编码最大长度是 1026，如果处理长度为 2000 的蛋白质序列，模型会报错或截断输入，因为位置编码无法覆盖超过 1026 的位置。
# 使用正弦编码不会遇到同样的问题，因为正弦编码是基于公式计算的，可以动态生成任意长度的编码，只要 d_model 和 max_len 足够大即可。因此，正弦编码在处理长序列时更灵活。

