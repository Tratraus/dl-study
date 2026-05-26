import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        d_model : 向量维度
        max_len : 支持的最大序列长度
        dropout : 防止过拟合
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算所有位置的编码，形状：(max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position：(max_len, 1)，每个位置的索引
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term：(d_model/2,)，每个时钟的频率
        # 公式：1 / 10000^(2i/d_model) = exp(-2i/d_model * log(10000))
        # 提示：torch.arange(0, d_model, 2) 生成 [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        # 提示：pe[:, 0::2] 取所有行的偶数列
        #       pe[:, 1::2] 取所有行的奇数列
        pe[:, 0::2] = torch.sin(position * div_term)   # sin
        pe[:, 1::2] = torch.cos(position * div_term)   # cos

        # 注册为 buffer（不是参数，不参与梯度更新，但会随模型保存）
        # 增加 batch 维度：(1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
        """
        # 把位置编码加到输入上
        # 注意：只取前 seq_len 个位置
        # 提示：self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def smoke_test():
    d_model, max_len = 64, 100
    pe_layer = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # 测试形状
    x = torch.zeros(2, 20, d_model)   # batch=2, seq_len=20
    out = pe_layer(x)
    print(f"input  shape : {x.shape}")    # (2, 20, 64)
    print(f"output shape : {out.shape}")  # (2, 20, 64)
    assert out.shape == (2, 20, d_model)

    # 验证：不同位置的编码不同
    pe_vals = pe_layer.pe[0]   # (max_len, d_model)
    assert not torch.allclose(pe_vals[0], pe_vals[1]), "位置0和位置1的编码相同！"
    assert not torch.allclose(pe_vals[0], pe_vals[10]), "位置0和位置10的编码相同！"
    print("✅ 通过")

    # 可视化：画出位置编码矩阵
    plt.figure(figsize=(12, 4))
    plt.imshow(pe_vals[:50, :].numpy(), aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Positional Encoding (pos=0~49, d_model=64)")
    plt.tight_layout()
    plt.savefig("./week5/day4/pe_visualization.png", dpi=120)
    print("图片已保存：./week5/day4/pe_visualization.png")

smoke_test()

