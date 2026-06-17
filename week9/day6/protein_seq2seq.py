import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ── 氨基酸词表 ────────────────────────────────────────────────
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

AA_CHARS = list('ACDEFGHIKLMNPQRSTVWY')  # 20 种标准氨基酸

VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + AA_CHARS
# PAD=0, BOS=1, EOS=2, UNK=3, A=4, C=5, ...

token2id = {tok: i for i, tok in enumerate(VOCAB)}
id2token = {i: tok for i, tok in enumerate(VOCAB)}

PAD = token2id[PAD_TOKEN]
BOS = token2id[BOS_TOKEN]
EOS = token2id[EOS_TOKEN]
UNK = token2id[UNK_TOKEN]
VOCAB_SIZE = len(VOCAB)  # 24

def encode(seq: str) -> list[int]:
    """氨基酸字符串 → token id 列表"""
    return [token2id.get(aa, UNK) for aa in seq.upper()]

def decode(ids: list[int]) -> str:
    """token id 列表 → 氨基酸字符串（跳过特殊 token）"""
    return ''.join(
        id2token[i] for i in ids
        if i not in (PAD, BOS, EOS, UNK)
    )


# ── 合成蛋白质数据生成 ────────────────────────────────────────
def make_protein_batch(batch_size, device, min_len=20, max_len=60):
    """
    模拟蛋白质 N端→C端预测任务。

    生成策略：
      1. 随机生成长度在 [min_len, max_len] 之间的氨基酸序列
      2. 前半段作为 src（N端），后半段作为 tgt（C端）
      3. 用 PAD 对齐到 batch 内最长长度

    返回：
      src:          (batch, max_src_len)，含 PAD
      tgt:          (batch, max_tgt_len+2)，含 BOS/EOS/PAD
      src_padding_mask: (batch, max_src_len)，True 表示 PAD 位置
    """
    src_seqs, tgt_seqs = [], []

    for _ in range(batch_size):
        length = random.randint(min_len, max_len)
        # 随机生成氨基酸序列
        seq = [random.choice(AA_CHARS) for _ in range(length)]
        mid = length // 2
        n_term = seq[:mid]   # N 端
        c_term = seq[mid:]   # C 端

        src_seqs.append(encode(''.join(n_term)))
        tgt_seqs.append(
            [BOS] + encode(''.join(c_term)) + [EOS]
        )

    # TODO 1：对 src_seqs 做 padding，对齐到 batch 内最长长度
    # 提示：用 torch.nn.utils.rnn.pad_sequence，或手动 padding
    # src shape: (batch, max_src_len)
    # src_padding_mask shape: (batch, max_src_len)，PAD 位置为 True
    src_lengths = [len(seq) for seq in src_seqs]
    max_src_len = max(src_lengths)
    src = torch.full((batch_size, max_src_len), PAD, dtype=torch.long, device=device)
    src_padding_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(src_seqs):
        src[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        src_padding_mask[i, :len(seq)] = False  # 非 PAD 位置为 False

    # TODO 2：对 tgt_seqs 做 padding
    # tgt shape: (batch, max_tgt_len)
    tgt_lengths = [len(seq) for seq in tgt_seqs]
    max_tgt_len = max(tgt_lengths)
    tgt = torch.full((batch_size, max_tgt_len), PAD, dtype=torch.long, device=device)
    for i, seq in enumerate(tgt_seqs):
        tgt[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    return src, tgt, src_padding_mask


# ── 复用 Day 3 的模型 ─────────────────────────────────────────
# 把 Encoder、Decoder、Seq2Seq、make_causal_mask 直接复制过来
# （不需要修改任何模型代码）
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
        positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)  # (1, src_len)
        x = self.embedding(src) + self.pos_embedding(positions)  # 语义更清晰
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

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # step 1: embedding，shape = (batch, tgt_len, d_model)
        x = self.embedding(tgt)  # (batch, tgt_len, d_model)
        # step 2: 加 positional encoding，shape = (batch, tgt_len, d_model)
        positions = torch.arange(x.size(1), device=tgt.device).unsqueeze(0)  # (1, tgt_len)
        x = x + self.pos_embedding(positions)  # (batch, tgt_len, d_model)
        x = self.dropout(x)
        # step 3: 过 N 层 TransformerDecoderLayer，shape = (batch, tgt_len, d_model)
        for layer in self.layers:
            x = layer(x, memory,
                  tgt_mask=tgt_mask,
                  tgt_key_padding_mask=tgt_key_padding_mask,
                  memory_key_padding_mask=memory_key_padding_mask)  # ← 传入  # (batch, tgt_len, d_model)
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
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
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


    def forward(self, src, tgt_input, src_key_padding_mask=None):
        # step 1: 生成因果 mask，shape = (tgt_len, tgt_len)
        tgt_len = tgt_input.size(1)
        tgt_mask = make_causal_mask(tgt_len, tgt_input.device)
        # step 2: Encoder 前向，得到 memory
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        # step 3: Decoder 前向，传入 tgt_input、memory、tgt_mask
        # 在 Seq2Seq.forward 中构建 tgt_padding_mask
        tgt_padding_mask = (tgt_input == PAD)  # (batch, tgt_len)

        logits = self.decoder(
            tgt_input, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # 返回 logits
        return logits



# ── 训练循环 ─────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL    = 128
    NUM_HEADS  = 4
    NUM_LAYERS = 3
    D_FF       = 256
    BATCH_SIZE = 32
    STEPS      = 1000

    model = Seq2Seq(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, STEPS + 1):
        # TODO 3：调用 make_protein_batch，完成训练循环
        # 注意：这次 src 有 padding，需要把 src_padding_mask 传给 Seq2Seq
        # 需要修改 Seq2Seq.forward 接受外部传入的 src_key_padding_mask
        src, tgt, src_padding_mask = make_protein_batch(BATCH_SIZE, device)

        tgt_input = tgt[:, :-1]  # (batch, tgt_len)
        tgt_output = tgt[:, 1:]   # (batch, tgt_len)

        optimizer.zero_grad()
        logits = model(src, tgt_input, src_key_padding_mask=src_padding_mask)

        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            tgt_output.reshape(-1),
            ignore_index=PAD
        )

        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("训练完成")
    evaluate(model, device)


# ── 评估函数 ─────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, src, src_padding_mask, max_len, device):

    # 1. Encoder 编码（传 mask）
    memory = model.encoder(src, src_key_padding_mask=src_padding_mask)

    # 2. 初始化：BOS
    batch_size = src.size(0)
    generated = torch.full((batch_size, 1), BOS, device=device)

    # 3. 自回归循环
    for _ in range(max_len):
        tgt_mask = make_causal_mask(generated.size(1), device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=src_padding_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # 所有样本都生成了 EOS 就停
        if (generated == EOS).any(dim=1).all():
            break

    # 4. 去掉开头 BOS
    return generated[:, 1:]

def evaluate(model, device, n_samples=3):
    """
    打印几条样本的 N端输入、真实C端、模型预测C端。
    注意：这个任务没有唯一正确答案（随机生成的序列），
    所以评估指标改为打印序列，肉眼观察模型是否生成了合法的氨基酸序列。
    """
    model.eval()
    src, tgt, src_padding_mask = make_protein_batch(n_samples, device, min_len=20, max_len=30)

    # TODO 4：用 greedy_decode 生成预测序列
    # 注意：greedy_decode 也需要传 src_padding_mask
    generated = greedy_decode(model, src, src_padding_mask, max_len=50, device=device)

    print("\n── 蛋白质序列预测 ──")
    for i in range(n_samples):
        n_term = decode(src[i].tolist())
        c_term_true = decode(tgt[i].tolist())
        c_term_pred = decode(generated[i].tolist())

        print(f"N端输入:   {n_term}")
        print(f"真实C端:   {c_term_true}")
        print(f"预测C端:   {c_term_pred}")
        print(f"长度匹配:  真实={len(c_term_true)}, 预测={len(c_term_pred)}")
        print()


if __name__ == "__main__":
    train()

# 输出
# Step 200, Loss: 2.9513
# Step 400, Loss: 2.9105
# Step 600, Loss: 2.9036
# Step 800, Loss: 2.9102
# Step 1000, Loss: 2.9039
# 训练完成

# ── 蛋白质序列预测 ──
# N端输入:   PLNVWGNWFYVKME
# 真实C端:   WGWIMYALCRDMNT
# 预测C端:   VVVVVVVVVVVVVV
# 长度匹配:  真实=14, 预测=14

# N端输入:   DNHKRLWHNS
# 真实C端:   FFQSWTVEGW
# 预测C端:   VVVVVVVVVV
# 长度匹配:  真实=10, 预测=10

# N端输入:   VQLRKMRIGRFTQK
# 真实C端:   LWNYQCIRICSKWT
# 预测C端:   VVVVVVVVVVVVVV
# 长度匹配:  真实=14, 预测=14

# Q1：src_padding_mask 的作用是什么？如果不传，会发生什么？
# 遮盖掉 src 中的 PAD 位置，使得 Encoder 在 self-attention 时不会关注 PAD。
# 如果不传，Encoder 会把 PAD 也当作有效 token，可能会影响到 memory 的表示，从而影响 Decoder 的生成结果。

# Q2：这个任务的初始 loss 理论值是多少？（提示：词表大小 24，但实际只会预测 20 种氨基酸 + EOS）
# log(21) ≈ 3.04，因为模型在一开始是随机初始化的，预测每个 token 的概率是均匀分布的，所以 CrossEntropyLoss 的初始值接近 log(21)。

# Q3：训练 1000 步后，loss 大概降到了多少？和序列翻转任务相比，为什么这个任务更难收敛？
# 训练 1000 步后，loss 大概降到了 2.9 左右。
# 这个任务更难收敛的原因是：
# 1. 序列翻转任务是确定性的，模型只需要学习一个简单的映射关系，而蛋白质预测任务是随机生成的，没有唯一正确答案，
#    模型需要学习到氨基酸的分布规律，难度更大。
# 2. 蛋白质序列的长度不固定，且可能较长，模型需要处理变长序列和更多的上下文信息。
# 3. 蛋白质序列的预测涉及到更复杂的依赖关系，而序列翻转任务只是简单的顺序关系，模型容易学习到。