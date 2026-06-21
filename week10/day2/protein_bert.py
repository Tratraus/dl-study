import torch
import torch.nn as nn
import sys
import os

# 复用 Day 1 的词表定义
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
from mlm_data import (
    VOCAB_SIZE, PAD, MASK, IGNORE_INDEX,
    make_mlm_batch, token2id, id2token
)



# ── TODO 1：实现 PositionalEncoding ──────────────────────────
class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learnable PE）。
    和 Week 9 一致，用 nn.Embedding 实现。

    输入：x，shape (batch, seq_len, d_model)
    输出：x + pos_emb，shape (batch, seq_len, d_model)

    提示：
      positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
      pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb


# ── TODO 2：实现 ProteinBERT ──────────────────────────────────
class ProteinBERT(nn.Module):
    """
    BERT 式蛋白质语言模型。

    结构：
      Embedding(vocab_size, d_model, padding_idx=PAD)
      + PositionalEncoding
      + Dropout
      → TransformerEncoderLayer × num_layers
      → LayerNorm
      → Linear(d_model, vocab_size)   ← MLM Head

    参数：
      vocab_size:  词表大小（直接用 VOCAB_SIZE）
      d_model:     嵌入维度，默认 128
      num_heads:   注意力头数，默认 4
      num_layers:  Encoder 层数，默认 3
      d_ff:        FFN 中间维度，默认 256
      max_len:     最大序列长度，默认 512
      dropout:     dropout 比例，默认 0.1

    forward 输入：
      src:                  (batch, seq_len)，含 MASK token 的序列
      src_key_padding_mask: (batch, seq_len)，True 表示该位置是 PAD，默认 None

    forward 输出：
      logits: (batch, seq_len, vocab_size)

    注意：
      - 不需要 Causal Mask
      - src_key_padding_mask 传给每一个 TransformerEncoderLayer
      - 最终输出是 logits，不是 softmax 后的概率（loss 函数会处理）
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model:    int = 128,
        num_heads:  int = 4,
        num_layers: int = 3,
        d_ff:       int = 256,
        max_len:    int = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()
        # TODO：定义以下模块
        # self.embedding = ...
        # self.pe = ...
        # self.dropout = ...
        # self.layers = ...   ← nn.ModuleList，每个元素是 TransformerEncoderLayer
        # self.norm = ...     ← LayerNorm
        # self.mlm_head = ... ← Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = PAD)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # TODO：
        # 1. Embedding + PE + Dropout
        # 2. 逐层过 TransformerEncoderLayer（传入 src_key_padding_mask）
        # 3. LayerNorm
        # 4. MLM Head → logits
        x = self.embedding(src)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        logits = self.mlm_head(x)
        return logits


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_model():
    """
    验证 ProteinBERT 的 forward 输出 shape 是否正确。

    步骤：
      1. 用 make_mlm_batch 生成一批数据（batch=4, seq_len=20）
      2. 构造 src_key_padding_mask（全 False，因为当前序列等长无 PAD）
      3. 实例化 ProteinBERT（用默认参数）
      4. 前向传播，打印 logits 的 shape
      5. 验证 shape == (4, 20, 25)

    额外验证：
      - 统计模型总参数量（可训练参数数）
      - 打印每个子模块的参数量
    """
    device = torch.device('cpu')
    masked_src, labels = make_mlm_batch(batch_size=4, seq_len=20, device=device)

    # src_key_padding_mask：当前无 PAD，全为 False
    src_key_padding_mask = (masked_src == PAD)   # (4, 20)，全 False

    model = ProteinBERT().to(device)
    logits = model(masked_src, src_key_padding_mask=src_key_padding_mask)

    print(f"logits shape: {logits.shape}")
    assert logits.shape == (4, 20, VOCAB_SIZE), \
        f"shape 错误！期望 (4, 20, {VOCAB_SIZE})，实际 {logits.shape}"
    print("✅ shape 验证通过")
    print()

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量：    {total:,}")
    print(f"可训练参数量：{trainable:,}")
    print()

    # 打印各子模块参数量
    print("各子模块参数量：")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {params:,}")


if __name__ == "__main__":
    verify_model()

# 输出
# logits shape: torch.Size([4, 20, 25])
# ✅ shape 验证通过

# 总参数量：    469,657
# 可训练参数量：469,657

# 各子模块参数量：
#   embedding      : 3,200
#   pe             : 65,536
#   dropout        : 0
#   layers         : 397,440
#   norm           : 256
#   mlm_head       : 3,225


# Q1：ProteinBERT 和 Week 9 的 Encoder 在代码上有哪两处关键差异？
# 去掉了Decoder部分，增加了MLM Head

# Q2：src_key_padding_mask = (masked_src == PAD) 这一行，当前输出全是 False。那么在 Day 3 引入变长序列后，这个 mask 的作用是什么？
# 在变长序列中，PAD token 用于填充序列，使得每个 batch 内的序列长度一致。src_key_padding_mask 会标记这些 PAD 位置，
# 确保 Transformer 在计算注意力时不会关注这些填充位置，从而避免对模型训练产生干扰。

# Q3：MLM Head 是 Linear(d_model, vocab_size)，输出是 logits 而不是概率。为什么不在模型里加 softmax，而是把它留给 loss 函数处理？
# 在训练过程中，使用 CrossEntropyLoss 时，它会自动将 logits 转换为概率分布并计算损失。