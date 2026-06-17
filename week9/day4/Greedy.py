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

        self.encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.decoder = Decoder(vocab_size, d_model, num_heads, num_layers, d_ff)


    def forward(self, src, tgt_input):
        # step 1: 生成因果 mask，shape = (tgt_len, tgt_len)
        tgt_len = tgt_input.size(1)
        tgt_mask = make_causal_mask(tgt_len, tgt_input.device)
        # step 2: Encoder 前向，得到 memory
        src_key_padding_mask = (src == PAD)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
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

    model.train()
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

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt_output.reshape(-1), ignore_index=PAD)


        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("训练完成")
    evaluate(model, device)


# ── 自回归推理 ────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """
    对单条或一批 src 序列进行 Greedy Decoding。

    输入：
      src:     (batch, src_len)
      max_len: 最大生成长度（防止无限循环）
    输出：
      generated: (batch, generated_len)  不含 <BOS>，含 <EOS>
    """
    model.eval()

    # TODO 1：用 Encoder 编码 src，得到 memory
    # memory shape: (batch, src_len, d_model)
    memory = model.encoder(src)

    batch_size = src.size(0)

    # TODO 2：初始化 generated，shape = (batch, 1)，全部填 BOS
    generated = torch.full((batch_size, 1), BOS, device=device)

    # TODO 3：自回归循环
    for _ in range(max_len):
        # step 1: 生成因果 mask
        tgt_mask = make_causal_mask(generated.size(1), device)

        # step 2: Decoder 前向
        # logits shape: (batch, current_len, vocab_size)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)

        # step 3: 取最后一个位置的 logits，argmax 得到下一个 token
        # next_token shape: (batch, 1)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # step 4: 拼接到 generated
        generated = torch.cat([generated, next_token], dim=1)

        # step 5: 如果所有样本都生成了 <EOS>，提前退出
        # 提示：检查 generated 中是否每行都含有 EOS
        if (generated == EOS).any(dim=1).all():
            break

    # 去掉开头的 <BOS>，返回
    return generated[:, 1:]


# ── 验证函数 ─────────────────────────────────────────────────
def evaluate(model, device, n_samples=5):
    """
    随机生成 n_samples 条样本，打印 src、期望输出、模型输出。
    """
    model.eval()
    src, tgt = make_batch(n_samples, 10, device)

    generated = greedy_decode(model, src, max_len=20, device=device)

    print("\n── 推理结果 ──")
    for i in range(n_samples):
        src_list = src[i].tolist()
        # tgt 是 [<BOS>, ...翻转..., <EOS>]，去掉首尾
        expected = tgt[i][1:-1].tolist()
        # generated 可能含 <EOS>，截断到第一个 <EOS>
        pred = generated[i].tolist()
        if EOS in pred:
            pred = pred[:pred.index(EOS)]

        match = "✅" if pred == expected else "❌"
        print(f"{match} src:      {src_list}")
        print(f"   expected: {expected}")
        print(f"   pred:     {pred}")
        print()



if __name__ == "__main__":
    train()

# 输出
# Step 100, Loss: 0.0624
# Step 200, Loss: 0.0130
# Step 300, Loss: 0.0058
# Step 400, Loss: 0.0037
# Step 500, Loss: 0.0029
# 训练完成

# ── 推理结果 ──
# ✅ src:      [16, 17, 9, 11, 13, 15, 13, 11, 17, 10]
#    expected: [10, 17, 11, 13, 15, 13, 11, 9, 17, 16]
#    pred:     [10, 17, 11, 13, 15, 13, 11, 9, 17, 16]

# ✅ src:      [8, 6, 8, 9, 9, 5, 8, 13, 19, 15]
#    expected: [15, 19, 13, 8, 5, 9, 9, 8, 6, 8]
#    pred:     [15, 19, 13, 8, 5, 9, 9, 8, 6, 8]

# ✅ src:      [3, 11, 15, 15, 6, 10, 3, 13, 10, 19]
#    expected: [19, 10, 13, 3, 10, 6, 15, 15, 11, 3]
#    pred:     [19, 10, 13, 3, 10, 6, 15, 15, 11, 3]

# ✅ src:      [3, 7, 10, 8, 11, 18, 15, 9, 9, 7]
#    expected: [7, 9, 9, 15, 18, 11, 8, 10, 7, 3]
#    pred:     [7, 9, 9, 15, 18, 11, 8, 10, 7, 3]

# ✅ src:      [17, 8, 4, 9, 11, 15, 4, 5, 14, 15]
#    expected: [15, 14, 5, 4, 15, 11, 9, 4, 8, 17]
#    pred:     [15, 14, 5, 4, 15, 11, 9, 4, 8, 17]

# Q1：推理时为什么还需要因果 mask？训练时加 mask 是为了防止"看到未来"，推理时 Decoder 的输入本来就只有"已生成的部分"，为什么还要加？
# 因为在推理时，Decoder 的输入是逐步生成的，虽然当前输入只包含已生成的部分，但在计算 self-attention 时，如果不加因果 mask，
# 模型仍然可以 attend 到当前输入中的未来位置（即还未生成的部分），这会导致信息泄露，使得模型在预测下一个 token 时能够"看到未来"的信息，从而影响生成质量。
# 因此，在推理时也需要加上因果 mask 来确保模型只能 attend 到当前位置及之前的位置。
# Answer:
# 推理时因果 mask 理论上不是必须的（因为输入本来就没有未来 token），但加上它有两个好处：
  # 和训练行为保持一致：训练时有 mask，推理时没有，模型看到的注意力模式不同，可能导致分布偏移
  # 工程上的防御性编程：统一加 mask，避免未来修改代码时出错
# Q2：generated[:, 1:] 去掉了 <BOS>。为什么最终输出不需要 <BOS>？
# 因为 <BOS> 是 Decoder 的输入标记，表示生成序列的开始，但它不是生成序列的一部分。在评估模型输出时，我们关心的是模型生成的内容，而不是输入标记，所以我们去掉 <BOS>。

# Q3：Greedy Decoding 有一个著名的缺陷：它不能保证找到全局最优序列。举一个简单的例子说明为什么。
# 贪心算法只能保证局部最优解
# 例如，假设模型在第一步预测时，给出了两个 token A 和 B，A 的概率稍微高于 B，所以 Greedy Decoding 选择了 A。
# 但是如果选择了 B，后续的预测可能会更好，最终生成的序列质量更高。Greedy Decoding 只关注当前步骤的最优选择，而没有考虑整体序列的最优性，因此可能错过更好的生成结果。