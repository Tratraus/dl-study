import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    把输入序列编码成上下文表示（memory）。

    输入：src tokens，shape = (batch, src_len)
    输出：memory，shape = (batch, src_len, d_model)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len = 512, dropout=0.1):
        super().__init__()
        # TODO 1：定义以下组件（只写 __init__，不写 forward）
        # - token embedding：把 token id 映射成向量
        # - positional encoding：给每个位置加上位置信息
        # - N 个 TransformerEncoderLayer
        # - 最后一个 LayerNorm
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        # TODO 2：写出 forward 的数据流，用注释标出每步的 shape
        # step 1: embedding，shape = (batch, src_len, d_model)
        # step 2: 加 positional encoding，shape = (batch, src_len, d_model)
        x = self.embedding(src) + self.pos_embedding(torch.arange(src.size(1), device=src.device))  # (batch, src_len, d_model)
        x = self.dropout(x)
        # step 3: 过 N 层 TransformerEncoderLayer，shape = (batch, src_len, d_model)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)  # (batch, src_len, d_model)
        # step 4: LayerNorm，shape = (batch, src_len, d_model)
        memory = self.norm(x)  # (batch, src_len, d_model)
        # 返回 memory (batch, src_len, d_model)
        return memory

class Decoder(nn.Module):
    """
    根据 memory 和已生成序列，预测下一个 token。

    输入：
      tgt tokens（已生成部分），shape = (batch, tgt_len)
      memory（来自 Encoder），shape = (batch, src_len, d_model)
    输出：
      logits，shape = (batch, tgt_len, vocab_size)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len = 512, dropout=0.1):
        super().__init__()
        # TODO 3：定义以下组件
        # - token embedding
        # - positional encoding
        # - N 个 TransformerDecoderLayer
        # - 最后一个 LayerNorm
        # - 输出投影：Linear(d_model, vocab_size)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # TODO 4：写出 forward 的数据流，用注释标出每步的 shape
        # step 1: embedding，shape = (batch, tgt_len, d_model)
        x = self.embedding(tgt)  # (batch, tgt_len, d_model)
        # step 2: 加 positional encoding，shape = (batch, tgt_len, d_model)
        x = x + self.pos_embedding(torch.arange(x.size(1), device=tgt.device))  # (batch, tgt_len, d_model)
        x = self.dropout(x)
        # step 3: 过 N 层 TransformerDecoderLayer，shape = (batch, tgt_len, d_model)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # (batch, tgt_len, d_model)
        # step 4: LayerNorm，shape = (batch, tgt_len, d_model)
        x = self.norm(x)  # (batch, tgt_len, d_model)
        # step 5: 输出投影，shape = (batch, tgt_len, vocab_size)
        logits = self.proj(x)  # (batch, tgt_len, vocab_size)
        # 返回 logits
        return logits



# Q1：Encoder 的输出 memory 的形状是什么？它在 Decoder 的哪个子层被使用？（提示：Decoder 每层有三个子层）
# (batch, src_len, d_model)，在Decoder的每一层的第二个子层（MultiheadAttention）中被使用，作为 key 和 value。

# Q2：Teacher Forcing 中，tgt_input 和 tgt_output 分别是什么？如果原始目标序列是 [<BOS>, 3, 1, 4, 1, 5, <EOS>]（长度 7），那么 tgt_input 和 tgt_output 各是什么？
# input是去掉最后一个token，输出是去掉第一个token。
# tgt_input 是 [<BOS>, 3, 1, 4, 1, 5]，长度为 6；tgt_output 是 [3, 1, 4, 1, 5, <EOS>]，长度为 6。

# Q3：nn.TransformerDecoderLayer 接收哪几个参数？其中哪个参数对应 memory？（查一下 PyTorch 文档或回忆一下函数签名）
# nn.TransformerDecoderLayer 接收以下参数：tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask。其中 memory 参数对应 Encoder 的输出 memory。