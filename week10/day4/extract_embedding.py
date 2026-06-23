import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))

from mlm_data import (
    VOCAB_SIZE, PAD, BOS, IGNORE_INDEX,
    make_mlm_batch, AA_IDS, token2id, id2token
)
from protein_bert import ProteinBERT


# ── TODO 1：CLS 池化 ──────────────────────────────────────────
def cls_pooling(
    model: ProteinBERT,
    src: torch.Tensor,
) -> torch.Tensor:
    """
    使用 [BOS] token 作为序列全局表示。

    步骤：
      1. 在 src 每条序列的开头插入 BOS token
         原始 src:  (batch, seq_len)
         插入后:    (batch, seq_len + 1)
      2. 构造 src_key_padding_mask（插入 BOS 后，BOS 位置不是 PAD）
      3. 调用 model.encode() 得到 hidden (batch, seq_len+1, d_model)
      4. 取第 0 个位置：hidden[:, 0, :]  →  (batch, d_model)

    提示：
      bos_col = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
      src_with_bos = torch.cat([bos_col, src], dim=1)
    """
    bos_col = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    src_with_bos = torch.cat([bos_col, src], dim=1)

    src_key_padding_mask = (src_with_bos == PAD)

    hidden = model.encode(src_with_bos, src_key_padding_mask=src_key_padding_mask)

    cls_embedding = hidden[:, 0, :]
    return cls_embedding


# ── TODO 2：平均池化 ──────────────────────────────────────────
def mean_pooling(
    model: ProteinBERT,
    src: torch.Tensor,
) -> torch.Tensor:
    """
    对所有非 PAD 位置的输出向量取平均。

    步骤：
      1. 构造 src_key_padding_mask: (batch, seq_len)，PAD 位置为 True
      2. 调用 model.encode() 得到 hidden (batch, seq_len, d_model)
      3. 构造 attention mask（非 PAD 位置为 1.0，PAD 位置为 0.0）
         mask: (batch, seq_len, 1)，用于广播
      4. 对非 PAD 位置求加权平均：
         sum(hidden * mask) / sum(mask)

    提示：
      mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
      embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    """
    src_key_padding_mask = (src == PAD)

    hidden = model.encode(src, src_key_padding_mask=src_key_padding_mask)

    mask = (~src_key_padding_mask).float().unsqueeze(-1)

    embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding


# ── TODO 3：验证函数 ──────────────────────────────────────────
def verify_embeddings():
    """
    验证两种池化方法的输出 shape 和基本性质。

    步骤：
      1. 加载 Day 3 保存的 checkpoint（protein_bert_mlm.pt）
      2. 生成一批等长序列（batch=4, seq_len=20）
      3. 分别用 cls_pooling 和 mean_pooling 提取嵌入
      4. 打印 shape、L2 范数

    额外验证：
      - 同一条序列，两种方法得到的嵌入是否相同？（应该不同）
      - 同一条序列输入两次，同一种方法得到的嵌入是否相同？
        （model.eval() 下应该相同，model.train() 下可能不同，为什么？）
    """
    device = torch.device('cpu')

    # 加载 checkpoint
    checkpoint_path = 'week10/day3/protein_bert_mlm.pt'
    model = ProteinBERT().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()   # ← 重要

    print(f"✅ 模型加载成功，final_loss = {checkpoint['final_loss']:.4f}")
    print()

    # 生成测试数据（不需要 MLM mask，直接用原始序列）
    src = torch.tensor(
        [__import__('random').choices(AA_IDS, k=20) for _ in range(4)],
        dtype=torch.long,
        device=device
    )

    # CLS 池化
    with torch.no_grad():
        cls_emb  = cls_pooling(model, src)
        mean_emb = mean_pooling(model, src)

    print(f"CLS  pooling shape: {cls_emb.shape}")
    print(f"Mean pooling shape: {mean_emb.shape}")
    print()

    # 打印每条序列的 L2 范数
    print("各序列嵌入的 L2 范数：")
    print(f"{'序列':>4}  {'CLS 范数':>10}  {'Mean 范数':>10}")
    print("-" * 30)
    for i in range(4):
        cls_norm  = cls_emb[i].norm().item()
        mean_norm = mean_emb[i].norm().item()
        print(f"  {i:2d}  {cls_norm:10.4f}  {mean_norm:10.4f}")

    print()

    # 验证：两种方法是否相同？
    diff = (cls_emb - mean_emb).norm(dim=-1)
    print(f"CLS vs Mean 差异（L2）：{diff.tolist()}")
    print("（应该不同，因为提取位置不同）")

    print()

    # 验证：同一输入，两次推理结果是否一致？
    with torch.no_grad():
        emb1 = mean_pooling(model, src)
        emb2 = mean_pooling(model, src)
    diff2 = (emb1 - emb2).norm(dim=-1).max().item()
    print(f"同一输入两次推理的最大差异：{diff2:.6f}")
    print("（model.eval() 下应该接近 0）")


if __name__ == "__main__":
    verify_embeddings()



# 输出
# ✅ 模型加载成功，final_loss = 2.8625

# CLS  pooling shape: torch.Size([4, 128])
# Mean pooling shape: torch.Size([4, 128])

# 各序列嵌入的 L2 范数：
#   序列      CLS 范数     Mean 范数
# ------------------------------
#    0     10.6471      6.5460
#    1     10.6500      6.7355
#    2     10.6713      6.6096
#    3     10.6675      6.7698

# CLS vs Mean 差异（L2）：[8.396041870117188, 8.53609561920166, 7.683117389678955, 8.282112121582031]
# （应该不同，因为提取位置不同）

# 同一输入两次推理的最大差异：0.000000
# （model.eval() 下应该接近 0）

# Q1：encode 方法和 forward 方法的区别是什么？为什么不直接用 forward 的输出来做池化？
# encode 方法只经过 Transformer 编码层，输出每个位置的隐藏状态；
# forward 方法在 encode 基础上还经过 mlm_head，输出每个位置的 token 预测分布。
# 池化应该基于 encode 的输出，因为我们需要的是隐藏状态，而不是预测分布。

# Q2：model.eval() 对嵌入提取有什么影响？如果忘记写 model.eval()，同一输入两次推理的结果会一样吗？
# model.eval() 会关闭 dropout 和其他训练时特有的行为，确保同一输入得到相同的输出。

# Q3：平均池化时，为什么要用 mask.sum(dim=1) 作为除数，而不是直接用 seq_len？
# 因为不同序列的有效长度可能不同，直接用 seq_len 会导致 PAD 位置的向量也被平均进去，影响结果。
# 使用 mask.sum(dim=1) 可以正确处理变长序列，只对有效位置的向量求平均。