import torch
import esm
import torch.nn.functional as F

def load_esm(device):
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter  = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    return model, alphabet, batch_converter


def extract_all_layers(sequences, model, batch_converter, device):
    """
    提取所有层的表示向量

    返回：dict，{name: tensor(num_layers, seq_len, d_model)}
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)

    num_layers = model.num_layers

    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=list(range(1, num_layers + 1)),  # 取全部层
            return_contacts=False
        )

    # TODO 1：对每条序列，把所有层的表示 stack 成一个 tensor
    # 提示：
    #   results["representations"] 是一个 dict，key 是层号（1,2,...,6）
    #   每个 value 的 shape 是 (batch, seq_len+2, d_model)
    #   目标：对每条序列，得到 shape=(num_layers, seq_len, d_model) 的 tensor
    #
    #   layer_reprs = []
    #   for layer_idx in range(1, num_layers + 1):
    #       rep = results["representations"][layer_idx][i, 1:len(seq)+1]
    #       layer_reprs.append(rep)
    #   all_layers[name] = torch.stack(layer_reprs, dim=0)

    all_layers = {}
    for i, (name, seq) in enumerate(sequences):
        layer_reprs = []
        for layer_idx in range(1, num_layers + 1):
            rep = results["representations"][layer_idx][i, 1:len(seq)+1]
            layer_reprs.append(rep)
        all_layers[name] = torch.stack(layer_reprs, dim=0)

    return all_layers


def analyze_layers(all_layers):
    """
    分析不同层之间的差异
    """
    print("── 各层 Mean Pooling 后的余弦相似度 ──")
    print("  （比较 helix_pure 和 strand_pure 在每一层的区分度）")
    print()

    helix_layers  = all_layers["helix_pure"]   # (num_layers, seq_len, d_model)
    strand_layers = all_layers["strand_pure"]  # (num_layers, seq_len, d_model)

    num_layers = helix_layers.shape[0]

    print(f"  {'层':>4}   {'余弦相似度':>10}   {'区分度（1-sim）':>14}")
    print(f"  {'-'*40}")

    for layer in range(num_layers):
        # TODO 2：计算第 layer 层的 helix_pure 和 strand_pure 的余弦相似度
        # 提示：先 mean(dim=0) 做 mean pooling，再 cosine_similarity
        helix_mean  = helix_layers[layer].mean(dim=0)
        strand_mean = strand_layers[layer].mean(dim=0)
        sim = F.cosine_similarity(helix_mean, strand_mean, dim=0).item()

        print(f"  第 {layer+1:1d} 层   {sim:>10.4f}   {1-sim:>14.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    model, alphabet, batch_converter = load_esm(device)

    sequences = [
        ("helix_pure",  "AVILMAVILMAVILMAVILM"),
        ("strand_pure", "FYWSTFYWSTFYWST"),
        ("mixed",       "MKTAYIAKQRQISFVKSHFS"),
    ]

    all_layers = extract_all_layers(sequences, model, batch_converter, device)

    # 验证 shape
    print("── 各序列的表示 shape ──")
    for name, rep in all_layers.items():
        print(f"  {name:15s}  shape={rep.shape}")
        # 期望：(6, seq_len, 320)

    print()
    analyze_layers(all_layers)

main()

#   层        余弦相似度       区分度（1-sim）
# ----------------------------------------
# 第 1 层       0.7589           0.2411
# 第 2 层       0.7506           0.2494
# 第 3 层       0.9107           0.0893
# 第 4 层       0.9264           0.0736
# 第 5 层       0.8398           0.1602
# 第 6 层       0.6674           0.3326
# 随着层数加深，helix_pure 和 strand_pure 的区分度是变大还是变小？为什么？
# 根据输出结果，二者的区分度在中间在第3、4层变小，而在之后又变大呈现U型。
# 可能在最开始模型更关注局部特征（如氨基酸类型），中间层可能更关注一些混合特征，导致区分度下降；
# 而在后面层数加深时，模型可能逐渐提取出更全局的结构信息，从而区分度又提升了。