import torch
import esm

# ════════════════════════════════════════════════════════════
# Part 1：加载模型
# ════════════════════════════════════════════════════════════

def load_esm(device):
    # 第一次运行会自动下载权重（约 150MB），之后缓存在本地
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    #                                 ↑
    #   t6  = 6 层 Transformer
    #   8M  = 800 万参数（最小版本，用于验证环境）
    #   UR50D = 训练数据集

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    print(f"模型加载完成")
    print(f"  层数：    {model.num_layers}")
    print(f"  d_model： {model.embed_dim}")
    print(f"  参数量：  {sum(p.numel() for p in model.parameters()):,}")

    return model, alphabet, batch_converter


# ════════════════════════════════════════════════════════════
# Part 2：提取表示向量
# ════════════════════════════════════════════════════════════

def extract_repr(sequences, model, batch_converter, device):
    """
    sequences : List of (name, seq_str)
    返回      : dict，{name: tensor(seq_len, d_model)}
    """
    # ESM 的标准输入格式
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[model.num_layers],   # 取最后一层
            return_contacts=False
        )

    # shape: (batch, seq_len+2, d_model)，+2 是 <cls> 和 <eos>
    token_repr = results["representations"][model.num_layers]

    # 去掉首尾特殊 token，只保留氨基酸对应的表示
    repr_dict = {}
    for i, (name, seq) in enumerate(sequences):
        repr_dict[name] = token_repr[i, 1:len(seq)+1]   # (seq_len, d_model)

    return repr_dict


# ════════════════════════════════════════════════════════════
# Part 3：分析表示向量
# ════════════════════════════════════════════════════════════

def analyze(repr_dict):
    print("\n── 表示向量基本信息 ──")
    for name, rep in repr_dict.items():
        print(f"  {name:15s}  shape={str(rep.shape):20s}  "
              f"mean={rep.mean().item():+.4f}  "
              f"std={rep.std().item():.4f}")

    # Mean Pooling：对每条序列取位置平均，得到"序列级别"向量
    # shape: (d_model,)
    pooled = {name: rep.mean(dim=0) for name, rep in repr_dict.items()}

    print("\n── 序列间余弦相似度 ──")
    print("  （越接近 1.0 表示 ESM-2 认为两条序列越相似）")
    names = list(pooled.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = pooled[names[i]]
            b = pooled[names[j]]
            sim = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            print(f"  {names[i]:15s} vs {names[j]:15s} : {sim:.4f}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    model, alphabet, batch_converter = load_esm(device)

    sequences = [
        ("helix_pure",   "AVILMAVILMAVILMAVILM"),   # 全疏水
        ("strand_pure",  "FYWSTFYWSTFYWST"),          # 全芳香/极性
        ("mixed",        "MKTAYIAKQRQISFVKSHFS"),    # 混合
        ("helix_pure2",  "LLLLLVVVVVAAAAALLLL"),     # 另一条全疏水
    ]

    repr_dict = extract_repr(sequences, model, batch_converter, device)
    analyze(repr_dict)

main()


# 1、helix_pure 和 helix_pure2 的相似度，比 helix_pure 和 strand_pure 的相似度高吗？
# helix_pure和2的相似度略低于helix_pure和strand_pure的相似度

# 2、repr_dict["helix_pure"] 的 shape 是什么？每个数字代表什么含义？
# torch.Size([20, 320])  mean=-0.0120  std=0.4457

# 20：序列长度
# 320：d_model
# mean：ESM学计算出的位置平均值
# std：标准差