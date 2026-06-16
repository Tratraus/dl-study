import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── 复用 Day 1 的 Encoder / Decoder ──────────────────────────
# 直接把 Day 1 的代码粘贴过来，或者 import

class Encoder(nn.Module):
    """
    把输入序列编码成上下文表示（memory）。

    输入：src tokens，shape = (batch, src_len)
    输出：memory，shape = (batch, src_len, d_model)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len = 512, dropout=0.1):
        super().__init__()
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

def make_causal_mask(size, device):
    """
    生成一个因果 mask，shape = (size, size)，上三角（不含对角线）为 -inf，其余为 0。
    这个 mask 会被加到 attention score 上，使得模型只能 attend 到当前位置及之前的位置。
    """
    mask = torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)
    return mask

# ── Seq2Seq 组装 ─────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """
    组装 Encoder 和 Decoder。

    forward 输入：
      src:        (batch, src_len)
      tgt_input:  (batch, tgt_len)   ← 已经是 tgt[:, :-1]
    forward 输出：
      logits:     (batch, tgt_len, vocab_size)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        # TODO 1：实例化 Encoder 和 Decoder
        self.encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.decoder = Decoder(vocab_size, d_model, num_heads, num_layers, d_ff)


    def forward(self, src, tgt_input):
        # TODO 2：
        # step 1: 生成因果 mask，shape = (tgt_len, tgt_len)
        tgt_len = tgt_input.size(1)
        tgt_mask = make_causal_mask(tgt_len, tgt_input.device)
        # step 2: Encoder 前向，得到 memory
        memory = self.encoder(src)
        # step 3: Decoder 前向，传入 tgt_input、memory、tgt_mask
        logits = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
        # 返回 logits
        return logits



# ── 合成数据生成 ──────────────────────────────────────────────
BOS, EOS, PAD = 1, 2, 0
VOCAB_SIZE = 20   # token id 范围 3~19

def make_batch(batch_size, seq_len, device):
    """
    生成一批序列翻转任务的数据。
    src:  随机序列，token id 在 [3, VOCAB_SIZE) 之间
    tgt:  <BOS> + src 翻转 + <EOS>
    """
    src = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    eos = torch.full((batch_size, 1), EOS, device=device)
    tgt = torch.cat([bos, src.flip(dims=[1]), eos], dim=1)
    return src, tgt


# ── 训练循环 ─────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    D_MODEL    = 64
    NUM_HEADS  = 4
    NUM_LAYERS = 2
    D_FF       = 128
    BATCH_SIZE = 64
    SEQ_LEN    = 10
    STEPS      = 500

    model = Seq2Seq(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO 3：写训练循环
    # 每个 step：
    #   1. make_batch 生成数据
    #   2. 切分 tgt_input 和 tgt_output
    #   3. 前向传播得到 logits
    #   4. 计算 CrossEntropyLoss（注意 ignore_index=PAD）
    #   5. 反向传播 + optimizer.step()
    #   6. 每 100 步打印一次 loss

    for step in range(1, STEPS + 1):
        src, tgt = make_batch(BATCH_SIZE, SEQ_LEN, device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt_output.reshape(-1), ignore_index=PAD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("训练完成")


if __name__ == "__main__":
    train()

# Q1：CrossEntropyLoss 为什么需要 ignore_index=PAD？在这个合成任务里，PAD 出现了吗？如果没有，为什么还要写？
# CrossEntropyLoss 需要 ignore_index=PAD 是因为在实际应用中，输入序列可能会被 padding，
# 而我们不希望模型把注意力放在 PAD 上，也不希望 PAD 对 loss 计算产生影响。
# 因为在这个合成任务里，PAD 没有出现（我们生成的数据没有用到 PAD），但是在实际应用中，输入序列可能会被 padding，
# 所以习惯上我们在定义 loss 时就加上 ignore_index=PAD，以防万一。

# Q2：logits.view(-1, vocab_size) 和 tgt_output.view(-1) 是为了什么？不 view 直接传给 loss 会发生什么？
# logits 的 shape 是 (batch_size, seq_len, vocab_size)，tgt_output 的 shape 是 (batch_size, seq_len)
# CrossEntropyLoss 期望输入的 shape 是 (N, C)，目标的 shape 是 (N,)
# 所以我们需要把 logits 和 tgt_output 展平。
# 如果不 view 直接传给 loss，会报 shape 不匹配的错误。

# Q3：训练 500 步后，loss 降到了多少？如果 loss 没有下降，你会先检查哪里？
# Step 100, Loss: 0.0597
# Step 200, Loss: 0.0136
# Step 300, Loss: 0.0074
# Step 400, Loss: 0.0040
# Step 500, Loss: 0.0027
# 首先检查数据生成和反馈传播