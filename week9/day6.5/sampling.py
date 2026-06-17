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
    src_key_padding_mask = (src == PAD)
    memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)

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

    src_key_padding_mask = (src == PAD)
    memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)   # (1, src_len, d_model)

    # 初始化 beams
    # 每个 beam 是一个 dict：
    #   "tokens": List[int]，当前已生成的 token（含 BOS）
    #   "score":  float，累计 log 概率
    #   "done":   bool，是否已生成 EOS
    beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

    for _ in range(max_len):

        if all(beam["done"] for beam in beams):
            break

        all_candidates = []

        for beam in beams:
            if beam["done"]:
                all_candidates.append(beam)
                continue

            # shape: (1, current_len)
            tgt = torch.tensor([beam["tokens"]], device=device)

            # logits shape: (1, current_len, vocab_size)
            tgt_mask = make_causal_mask(tgt.size(1), device)
            logits = model.decoder(tgt, memory, tgt_mask=tgt_mask)

            # log_probs shape: (vocab_size,)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for log_prob, idx in zip(topk_log_probs, topk_indices):
                candidate = {
                    "tokens": beam["tokens"] + [idx.item()],
                    "score": beam["score"] + log_prob.item(),
                    "done": idx.item() == EOS
                }
                all_candidates.append(candidate)

        all_candidates.sort(key=lambda c: c["score"], reverse=True)

        beams = all_candidates[:beam_width]

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

# ── 1. 带长度归一化的 Beam Search ────────────────────────────
@torch.no_grad()
def beam_search_with_length_penalty(model, src, beam_width, max_len, device, alpha=0.6):
    """
    在 Day 5 的 beam_search 基础上，排序时使用长度归一化得分。

    唯一的修改点：
      选出最优 beam 时，用 score / (len ** alpha) 排序
      但累计 score 本身仍然是原始 log 概率之和（不归一化）
    """
    model.eval()
    src_key_padding_mask = (src == PAD)
    memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
    beams = [{"tokens": [BOS], "score": 0.0, "done": False}]

    for _ in range(max_len):
        if all(b["done"] for b in beams):
            break

        all_candidates = []
        for beam in beams:
            if beam["done"]:
                all_candidates.append(beam)
                continue

            tgt = torch.tensor([beam["tokens"]], device=device)
            tgt_mask = make_causal_mask(tgt.size(1), device)
            logits = model.decoder(tgt, memory, tgt_mask=tgt_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_tokens = beam["tokens"] + [idx.item()]
                all_candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + log_prob.item(),
                    "done": idx.item() == EOS
                })

        # TODO 1：排序时使用长度归一化
        # 提示：排序 key = score / (len(tokens) ** alpha)
        # 注意：len(tokens) 包含 BOS，所以实际生成长度 = len(tokens) - 1
        # 注意：alpha=0 时应退化为原始 beam search
        beams = sorted(
            all_candidates,
            key=lambda b: b["score"] / (max(len(b["tokens"]) - 1, 1) ** alpha),
            reverse=True
        )[:beam_width]

    best = max(beams, key=lambda b: b["score"] / (max(len(b["tokens"]) - 1, 1) ** alpha))
    return best["tokens"][1:]


# ── 2. Temperature Sampling ───────────────────────────────────
@torch.no_grad()
def temperature_sample(model, src, max_len, device, temperature=1.0):
    """
    用 Temperature Sampling 生成序列。
    结构和 greedy_decode 几乎一样，只改一行。

    输入：
      src:         (1, src_len)   ← 单条序列
      temperature: float，控制随机性
    输出：
      generated:   List[int]，不含 BOS，含 EOS
    """
    model.eval()
    src_key_padding_mask = (src == PAD)
    memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
    generated = torch.full((1, 1), BOS, device=device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(generated.size(1), device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)

        # TODO 2：用 temperature 缩放 logits，然后采样（不是 argmax）
        # 提示：torch.multinomial(probs, num_samples=1)
        scaled_logits = logits[:, -1, :] / temperature        # 缩放
        probs = F.softmax(scaled_logits, dim=-1)              # 转概率
        next_token = torch.multinomial(probs, num_samples=1)  # 采样

        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == EOS:
            break

    return generated[0, 1:].tolist()


# ── 3. Top-k Sampling ────────────────────────────────────────
@torch.no_grad()
def topk_sample(model, src, max_len, device, k=10, temperature=1.0):
    """
    Top-k Sampling：只从概率最高的 k 个 token 中采样。

    输入：
      src:         (1, src_len)
      k:           int，保留的 token 数量
      temperature: float
    输出：
      generated:   List[int]
    """
    model.eval()
    src_key_padding_mask = (src == PAD)
    memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
    generated = torch.full((1, 1), BOS, device=device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(generated.size(1), device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)

        last_logits = logits[0, -1, :].unsqueeze(0)   # (1, vocab_size)

        # TODO 3：实现 Top-k 截断
        # step 1: 找到第 k 大的 logit 值（阈值）
        # step 2: 把低于阈值的位置设为 -inf
        # step 3: 用 temperature 缩放，softmax，采样
        # 提示：torch.topk(last_logits, k) 返回 (values, indices)
        #       阈值 = values[-1]（第 k 大的值）
        # 替换 ... 部分：
        # 1. 找第 k 大的 logit 作为阈值
        topk_values, _ = torch.topk(last_logits, k)
        threshold = topk_values[0, -1]

        # 2. 低于阈值的设为 -inf
        last_logits = last_logits.masked_fill(last_logits < threshold, float('-inf'))

        # 3. temperature 缩放 + softmax + 采样
        scaled_logits = last_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == EOS:
            break

    return generated[0, 1:].tolist()


# ── 对比实验 ─────────────────────────────────────────────────
def compare_all(model, device, n_samples=3, T = 0.8):
    model.eval()
    src_batch, tgt_batch = make_batch(n_samples, 10, device)

    print(f"\n── 四种解码策略对比(T ={T}) ──")
    for i in range(n_samples):
        src = src_batch[i].unsqueeze(0)
        expected = tgt_batch[i][1:-1].tolist()

        # Greedy
        g = greedy_decode(model, src, 20, device)[0].tolist()
        if EOS in g: g = g[:g.index(EOS)]

        # Beam Search（无长度惩罚）
        b = beam_search(model, src, 3, 20, device)
        if EOS in b: b = b[:b.index(EOS)]

        # Beam Search（有长度惩罚）
        bl = beam_search_with_length_penalty(model, src, 3, 20, device, alpha=0.6)
        if EOS in bl: bl = bl[:bl.index(EOS)]

        # Temperature Sampling（T=0.8）
        t = temperature_sample(model, src, 20, device, temperature=T)
        if EOS in t: t = t[:t.index(EOS)]

        # Top-k Sampling（k=5, T=0.8）
        tk = topk_sample(model, src, 20, device, k=5, temperature=T)
        if EOS in tk: tk = tk[:tk.index(EOS)]

        g_mark  = "✅" if g  == expected else "❌"
        b_mark  = "✅" if b  == expected else "❌"
        bl_mark = "✅" if bl == expected else "❌"

        print(f"expected:      {expected}")
        print(f"greedy      {g_mark}: {g}")
        print(f"beam        {b_mark}: {b}")
        print(f"beam+len   {bl_mark}: {bl}")
        print(f"temp({T})     : {t}  ← 随机，不评对错")
        print(f"topk(5,{T})   : {tk} ← 随机，不评对错")
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
    return model, device


if __name__ == "__main__":
    model, device = train()
    compare_all(model, device, T = 0.8)
    compare_all(model, device, T = 1.2)

# 输出
# Step 100, Loss: 0.0519
# Step 200, Loss: 0.0135
# Step 300, Loss: 0.0072
# Step 400, Loss: 0.0043
# Step 500, Loss: 0.0030
# 训练完成

# ── 推理结果 ──
# ✅ src:      [15, 3, 3, 12, 5, 10, 14, 11, 11, 8]
#    expected: [8, 11, 11, 14, 10, 5, 12, 3, 3, 15]
#    pred:     [8, 11, 11, 14, 10, 5, 12, 3, 3, 15]

# ✅ src:      [8, 6, 19, 12, 14, 6, 9, 6, 17, 3]
#    expected: [3, 17, 6, 9, 6, 14, 12, 19, 6, 8]
#    pred:     [3, 17, 6, 9, 6, 14, 12, 19, 6, 8]

# ✅ src:      [3, 6, 4, 7, 12, 5, 18, 5, 18, 19]
#    expected: [19, 18, 5, 18, 5, 12, 7, 4, 6, 3]
#    pred:     [19, 18, 5, 18, 5, 12, 7, 4, 6, 3]

# ✅ src:      [7, 18, 19, 12, 7, 16, 3, 18, 10, 3]
#    expected: [3, 10, 18, 3, 16, 7, 12, 19, 18, 7]
#    pred:     [3, 10, 18, 3, 16, 7, 12, 19, 18, 7]

# ✅ src:      [7, 8, 7, 19, 12, 4, 16, 13, 12, 7]
#    expected: [7, 12, 13, 16, 4, 12, 19, 7, 8, 7]
#    pred:     [7, 12, 13, 16, 4, 12, 19, 7, 8, 7]


# ── 四种解码策略对比(T =0.8) ──
# expected:      [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
# greedy      ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
# beam        ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
# beam+len   ✅: [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]
# temp(0.8)     : [5, 10, 13, 5, 8, 19, 9, 4, 12, 15]  ← 随机，不评对错
# topk(5,0.8)   : [5, 10, 13, 5, 8, 19, 9, 4, 12, 15] ← 随机，不评对错

# expected:      [18, 6, 7, 3, 15, 13, 15, 8, 9, 9]
# greedy      ✅: [18, 6, 7, 3, 15, 13, 15, 8, 9, 9]
# beam        ✅: [18, 6, 7, 3, 15, 13, 15, 8, 9, 9]
# beam+len   ✅: [18, 6, 7, 3, 15, 13, 15, 8, 9, 9]
# temp(0.8)     : [18, 6, 7, 3, 15, 13, 15, 8, 9, 9]  ← 随机，不评对错
# topk(5,0.8)   : [18, 6, 7, 3, 15, 13, 15, 8, 9, 9] ← 随机，不评对错

# expected:      [18, 9, 17, 17, 17, 19, 8, 10, 3, 3]
# greedy      ✅: [18, 9, 17, 17, 17, 19, 8, 10, 3, 3]
# beam        ✅: [18, 9, 17, 17, 17, 19, 8, 10, 3, 3]
# beam+len   ✅: [18, 9, 17, 17, 17, 19, 8, 10, 3, 3]
# temp(0.8)     : [18, 9, 17, 17, 17, 19, 8, 10, 3, 3]  ← 随机，不评对错
# topk(5,0.8)   : [18, 9, 17, 17, 17, 19, 8, 10, 3, 3] ← 随机，不评对错


# ── 四种解码策略对比(T =1.2) ──
# expected:      [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]
# greedy      ✅: [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]
# beam        ✅: [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]
# beam+len   ✅: [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]
# temp(1.2)     : [19, 7, 3, 18, 18, 16, 14, 7, 18, 5]  ← 随机，不评对错
# topk(5,1.2)   : [19, 7, 3, 18, 18, 16, 14, 7, 16, 5] ← 随机，不评对错

# expected:      [8, 9, 9, 15, 14, 13, 10, 19, 11, 4]
# greedy      ✅: [8, 9, 9, 15, 14, 13, 10, 19, 11, 4]
# beam        ✅: [8, 9, 9, 15, 14, 13, 10, 19, 11, 4]
# beam+len   ✅: [8, 9, 9, 15, 14, 13, 10, 19, 11, 4]
# temp(1.2)     : [8, 9, 9, 15, 14, 13, 10, 19, 11, 4]  ← 随机，不评对错
# topk(5,1.2)   : [8, 9, 9, 15, 14, 13, 10, 19, 11, 4] ← 随机，不评对错

# expected:      [13, 5, 5, 15, 9, 7, 16, 11, 11, 3]
# greedy      ✅: [13, 5, 5, 15, 9, 7, 16, 11, 11, 3]
# beam        ✅: [13, 5, 5, 15, 9, 7, 16, 11, 11, 3]
# beam+len   ✅: [13, 5, 5, 15, 9, 7, 16, 11, 11, 3]
# temp(1.2)     : [13, 5, 5, 15, 9, 7, 16, 11, 11, 3]  ← 随机，不评对错
# topk(5,1.2)   : [13, 5, 5, 15, 9, 7, 16, 11, 11, 3] ← 随机，不评对错


# Q1：Temperature = 0.8 和 Temperature = 1.2 生成的序列有什么区别？运行几次，观察随机性的变化。
# 理论上，Temperature 越小，生成的序列越确定（更接近贪心解），而 Temperature 越大，生成的序列越随机（多样性更高）。

# Q2：Top-k 中，k 的选择有什么 trade-off？k 太小和 k 太大分别会导致什么问题？
# 太小会退化为greedy，太大会变成随机采样，失去控制。
# 一般来说，k 的选择需要根据具体任务和模型的表现来调整，常见的值是 5、10 或 20。

# Q3：在蛋白质序列生成场景中，你会选择哪种解码策略？为什么？（没有标准答案，说出你的思考即可）
# 取决于具体需求，Top-k和Temperature可以提供更多样性的回答，而greedy和beam search更倾向于生成高概率的序列。