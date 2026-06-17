import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用之前的模型定义和训练好的权重
# （把 Day 4 的完整代码复制过来，在 train() 末尾保存模型）
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

    # memory shape: (batch, src_len, d_model)
    memory = model.encoder(src)

    batch_size = src.size(0)

    generated = torch.full((batch_size, 1), BOS, device=device)

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
    torch.save(model.state_dict(), "seq2seq.pt")

# ── 在 train() 末尾添加 ───────────────────────────────────────
# torch.save(model.state_dict(), "seq2seq.pt")

# ── Beam Search ──────────────────────────────────────────────
@torch.no_grad()
def beam_search(model, src, beam_width, max_len, device):
    """
    单条序列的 Beam Search。

    输入：
      src:        (1, src_len)   ← 注意是单条
      beam_width: int
      max_len:    int
    输出：
      best_tokens: List[int]，最优序列（不含 BOS，含 EOS）
    """
    model.eval()

    # TODO 1：Encoder 编码
    memory = model.encoder(src)   # (1, src_len, d_model)

    # 初始化 beams
    # 每个 beam 是一个 dict：
    #   "tokens": List[int]，当前已生成的 token（含 BOS）
    #   "score":  float，累计 log 概率
    #   "done":   bool，是否已生成 EOS
    beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

    for _ in range(max_len):

        # TODO 2：如果所有 beam 都 done，退出
        if all(beam["done"] for beam in beams):
            break

        all_candidates = []

        for beam in beams:
            # TODO 3：跳过已完成的 beam（直接加入 candidates，不展开）
            if beam["done"]:
                all_candidates.append(beam)
                continue

            # TODO 4：把当前 beam 的 tokens 转成 tensor
            # shape: (1, current_len)
            tgt = torch.tensor([beam["tokens"]], device=device)

            # TODO 5：生成因果 mask，Decoder 前向
            # logits shape: (1, current_len, vocab_size)
            tgt_mask = make_causal_mask(tgt.size(1), device)
            logits = model.decoder(tgt, memory, tgt_mask=tgt_mask)

            # TODO 6：取最后一个位置，计算 log softmax
            # log_probs shape: (vocab_size,)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

            # TODO 7：展开 beam_width 个最优候选
            # 提示：用 torch.topk(log_probs, beam_width)
            # 对每个候选，新 score = beam["score"] + log_prob
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for log_prob, idx in zip(topk_log_probs, topk_indices):
                candidate = {
                    "tokens": beam["tokens"] + [idx.item()],
                    "score": beam["score"] + log_prob.item(),
                    "done": idx.item() == EOS
                }
                all_candidates.append(candidate)

        # TODO 8：从 all_candidates 中选出得分最高的 beam_width 个
        # 提示：按 score 排序，取前 beam_width 个
        all_candidates.sort(key=lambda c: c["score"], reverse=True)

        beams = all_candidates[:beam_width]

    # TODO 9：从所有 beam 中选出得分最高的，返回其 tokens（去掉 BOS）
    best = max(beams, key=lambda b: b["score"])
    return best["tokens"][1:]   # 去掉 BOS


# ── 对比验证 ─────────────────────────────────────────────────
def compare(model, device, n_samples=5):
    """
    对比 Greedy 和 Beam Search 的结果。
    """
    model.eval()
    src_batch, tgt_batch = make_batch(n_samples, 10, device)

    print("\n── Greedy vs Beam Search ──")
    for i in range(n_samples):
        src = src_batch[i].unsqueeze(0)   # (1, src_len)
        expected = tgt_batch[i][1:-1].tolist()

        # Greedy
        greedy_out = greedy_decode(model, src, max_len=20, device=device)
        greedy_pred = greedy_out[0].tolist()
        if EOS in greedy_pred:
            greedy_pred = greedy_pred[:greedy_pred.index(EOS)]

        # Beam Search
        beam_pred = beam_search(model, src, beam_width=3, max_len=20, device=device)
        if EOS in beam_pred:
            beam_pred = beam_pred[:beam_pred.index(EOS)]

        g_mark = "✅" if greedy_pred == expected else "❌"
        b_mark = "✅" if beam_pred   == expected else "❌"

        print(f"src:      {src_batch[i].tolist()}")
        print(f"expected: {expected}")
        print(f"greedy  {g_mark}: {greedy_pred}")
        print(f"beam    {b_mark}: {beam_pred}")
        print()

if __name__ == "__main__":
    train()

# 输出
# Step 100, Loss: 0.0672
# Step 200, Loss: 0.0124
# Step 300, Loss: 0.0063
# Step 400, Loss: 0.0042
# Step 500, Loss: 0.0050
# 训练完成

# ── 推理结果 ──
# ✅ src:      [10, 11, 15, 7, 14, 19, 11, 5, 4, 16]
#    expected: [16, 4, 5, 11, 19, 14, 7, 15, 11, 10]
#    pred:     [16, 4, 5, 11, 19, 14, 7, 15, 11, 10]

# ✅ src:      [8, 11, 17, 4, 14, 3, 17, 10, 4, 6]
#    expected: [6, 4, 10, 17, 3, 14, 4, 17, 11, 8]
#    pred:     [6, 4, 10, 17, 3, 14, 4, 17, 11, 8]

# ✅ src:      [12, 14, 8, 9, 8, 16, 4, 5, 12, 15]
#    expected: [15, 12, 5, 4, 16, 8, 9, 8, 14, 12]
#    pred:     [15, 12, 5, 4, 16, 8, 9, 8, 14, 12]

# ✅ src:      [15, 9, 7, 18, 14, 5, 6, 6, 4, 4]
#    expected: [4, 4, 6, 6, 5, 14, 18, 7, 9, 15]
#    pred:     [4, 4, 6, 6, 5, 14, 18, 7, 9, 15]

# ✅ src:      [9, 19, 5, 7, 10, 12, 7, 12, 11, 14]
#    expected: [14, 11, 12, 7, 12, 10, 7, 5, 19, 9]
#    pred:     [14, 11, 12, 7, 12, 10, 7, 5, 19, 9]

# Q1：为什么用 log 概率累加 而不是概率直接相乘？
# log之后，值线性化，不容易出现极大值和极小值的情况，数值更稳定。

# Q2：Beam Search 的时间复杂度是 Greedy 的多少倍？（从每步的计算量角度分析）
# Greedy 每步只计算 1 个候选，Beam Search 每步计算 beam_width 个候选，所以时间复杂度是 Greedy 的 beam_width 倍。

# Q3：在这个序列翻转任务上，Beam Search 和 Greedy 的结果几乎一样。什么样的任务上 Beam Search 会有明显优势？（提示：想想什么情况下"局部最优 ≠ 全局最优"）
# 当模型在某些步骤上可能会犯错，导致 Greedy 选择了一个局部最优但全局次优的 token 时，Beam Search 可以保留多个候选，增加找到全局最优序列的机会。
# 例如，在机器翻译中，某个词的翻译可能有多种可能，Greedy 可能选了一个不太合适的翻译，而 Beam Search 可以同时考虑多个翻译选项，最终生成更流畅、更准确的句子。