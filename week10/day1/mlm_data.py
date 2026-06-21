import torch
import random

# ── 词表定义（在 Week 9 基础上新增 MASK token）────────────────
PAD_TOKEN  = '<PAD>'
BOS_TOKEN  = '<BOS>'
EOS_TOKEN  = '<EOS>'
UNK_TOKEN  = '<UNK>'
MASK_TOKEN = '<MASK>'   # 🆕 新增

AA_CHARS = list('ACDEFGHIKLMNPQRSTVWY')

VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN] + AA_CHARS
# PAD=0, BOS=1, EOS=2, UNK=3, MASK=4, A=5, C=6, ...

token2id = {tok: i for i, tok in enumerate(VOCAB)}
id2token = {i: tok for i, tok in enumerate(VOCAB)}

PAD  = token2id[PAD_TOKEN]
BOS  = token2id[BOS_TOKEN]
EOS  = token2id[EOS_TOKEN]
UNK  = token2id[UNK_TOKEN]
MASK = token2id[MASK_TOKEN]
VOCAB_SIZE = len(VOCAB)   # 25

AA_IDS = [token2id[aa] for aa in AA_CHARS]   # 氨基酸的 id 列表，用于随机替换

IGNORE_INDEX = -100   # cross_entropy 跳过的标记


def encode(seq: str) -> list[int]:
    return [token2id.get(aa, UNK) for aa in seq.upper()]

def decode(ids: list[int]) -> str:
    return ''.join(
        id2token[i] for i in ids
        if i not in (PAD, BOS, EOS, UNK, MASK)
    )


# ── TODO 1：实现 mask_sequence ────────────────────────────────
def mask_sequence(token_ids: list[int], mask_prob: float = 0.15) -> tuple[list[int], list[int]]:
    """
    对一条序列应用 BERT 的 80-10-10 mask 策略。

    输入：
      token_ids:  List[int]，原始 token id 列表（只含氨基酸，不含特殊 token）
      mask_prob:  float，被选中进行 mask 的概率，默认 0.15

    输出：
      masked_ids: List[int]，经过 mask 处理后的序列
      labels:     List[int]，被 mask 位置填原始 token id，其余位置填 IGNORE_INDEX

    策略：
      1. 遍历每个位置，以 mask_prob 的概率选中该位置
      2. 对选中的位置：
         - 80% 概率：替换为 MASK token
         - 10% 概率：替换为随机氨基酸 id（从 AA_IDS 中随机选）
         - 10% 概率：保持原 token 不变
      3. 未选中的位置：labels 填 IGNORE_INDEX，masked_ids 保持不变

    提示：
      r = random.random()
      if r < 0.8:    → MASK
      elif r < 0.9:  → 随机 token
      else:          → 保持不变
    """
    masked_ids = []
    labels = []
    for token_id in token_ids:
        r1 = random.random()
        if r1 < mask_prob:
            r2 = random.random()
            if r2 < 0.8:
                # 替换为 MASK token
                masked_id = MASK
            elif r2 < 0.9:
                masked_id = random.choice(AA_IDS)  # 替换为随机氨基酸 id
            else:
                masked_id = token_id  # 保持不变
            masked_ids.append(masked_id)
            labels.append(token_id)  # 只有被选中位置才有有效 label
        else:
            masked_ids.append(token_id)
            labels.append(IGNORE_INDEX)

    return masked_ids, labels
# ── TODO 2：实现 make_mlm_batch ───────────────────────────────
def make_mlm_batch(batch_size: int, seq_len: int, device, return_original=False) -> tuple:
    """
    生成一批 MLM 训练数据。

    步骤：
      1. 随机生成 batch_size 条氨基酸序列，每条长度为 seq_len
      2. 对每条序列调用 mask_sequence
      3. 把 masked_ids 和 labels 分别 stack 成 tensor

    返回：
      masked_src: (batch, seq_len)，含 MASK token 的输入序列
      labels:     (batch, seq_len)，只有 mask 位置有有效 id，其余为 IGNORE_INDEX

    注意：这里暂时不加 PAD（序列等长），PAD 处理留到 Day 2 的模型里。
    """
    all_original = []
    all_masked = []
    all_labels = []

    for _ in range(batch_size):
        seq = random.choices(AA_IDS, k=seq_len)
        masked_ids, labels = mask_sequence(seq)

        all_original.append(seq)
        all_masked.append(masked_ids)
        all_labels.append(labels)

    original = torch.tensor(all_original, dtype=torch.long, device=device)
    masked_src = torch.tensor(all_masked, dtype=torch.long, device=device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)

    if return_original:
        return original, masked_src, labels
    return masked_src, labels


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_batch():
    """
    生成一个小 batch，打印第一条样本，肉眼验证：
      1. masked_src 中约 15% 位置被替换（MASK=4 或随机氨基酸）
      2. labels 中只有被选中位置有有效 id，其余为 -100
      3. masked_src 和 labels 中，被选中位置的原始 token 和 label 对应正确
    """
    device = torch.device('cpu')
    original, masked_src, labels = make_mlm_batch(
    batch_size=4,
    seq_len=20,
    device=device,
    return_original=True
    )
    print(f"original shape:   {original.shape}")
    print(f"masked_src shape: {masked_src.shape}")
    print(f"labels shape:     {labels.shape}")
    print()

    # 打印第一条样本
    src_0   = masked_src[0].tolist()
    label_0 = labels[0].tolist()
    original_0 = original[0].tolist()

    print("位置  original  masked_src  label")
    print("----  --------  ----------  -----")
    for i, (o, s, l) in enumerate(zip(original_0, src_0, label_0)):
        marker = "← MASKED" if l != IGNORE_INDEX else ""
        print(f"  {i:2d}  {id2token.get(o, '?'):>8}  {id2token.get(s, '?'):>10}  {l:5d}  {marker}")

    # 统计 mask 比例
    total = len(label_0)
    masked = sum(1 for l in label_0 if l != IGNORE_INDEX)
    print(f"\nmask 比例: {masked}/{total} = {masked/total:.1%}（理论约 15%）")


if __name__ == "__main__":
    verify_batch()

# 输出
# original shape:   torch.Size([4, 20])
# masked_src shape: torch.Size([4, 20])
# labels shape:     torch.Size([4, 20])

# 位置  original  masked_src  label
# ----  --------  ----------  -----
#    0         A           A   -100
#    1         H      <MASK>     11  ← MASKED
#    2         K           K   -100
#    3         W           W   -100
#    4         Q           Q   -100
#    5         R           R   -100
#    6         F           F   -100
#    7         I           I   -100
#    8         C           C   -100
#    9         S           S   -100
#   10         W           W   -100
#   11         N           N   -100
#   12         M           M   -100
#   13         N           N   -100
#   14         P           P   -100
#   15         M      <MASK>     15  ← MASKED
#   16         E           E   -100
#   17         K           K   -100
#   18         D           D   -100
#   19         I           Y     12  ← MASKED

# mask 比例: 3/20 = 15.0%（理论约 15%）

# mask 比例: 2/20 = 10.0%（理论约 15%）

# Q1：为什么 MLM 只对 mask 位置计算 loss，而不是对整个序列？
# 因为 MLM 的目标是预测被 mask 的 token，其他位置不需要参与 loss 计算。
# 如果对整个序列计算 loss，模型可能会过度关注非 mask 位置，导致训练效果不佳。

# Q2：BERT 的 80-10-10 策略中，为什么不把 15% 全部替换为 [MASK]？
# 因为在实际应用中，输入序列中不会有 [MASK] token，模型需要学会在没有明确标记的情况下预测被遮盖的 token。
# 通过随机替换和保持不变，模型可以学会更鲁棒地处理各种情况，而不仅仅是依赖于 [MASK] 标记。

# Q3：Encoder-only 和 Seq2Seq 的注意力方向有什么本质区别？对蛋白质序列建模来说，双向注意力为什么更合适？
# Encoder-only 模型（如 BERT）允许每个位置同时关注序列中的所有其他位置（双向注意力），而 Seq2Seq 模型（如 GPT）通常是单向的，当前 token 只能关注之前的 token。
# 对蛋白质序列建模来说，双向注意力更合适，因为蛋白质的功能和结构通常依赖于整个序列的信息，单向模型可能无法捕捉到全局的上下文关系。