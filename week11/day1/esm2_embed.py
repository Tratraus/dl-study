# 首先安装依赖（在终端执行，不要放在代码里）：
# pip install transformers

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


# ── 常量 ──────────────────────────────────────────────────────
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 10 条来自 UniProt 的真实蛋白质序列（人工精选，覆盖不同功能）
TEST_SEQUENCES = [
    # 血红蛋白 alpha 链（氧气运输）
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    # 胰岛素（激素）
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    # 绿色荧光蛋白 GFP（荧光）
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    # 溶菌酶（抗菌）
    "MRSLLILVLCFLPLAALGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
    # 肌红蛋白（氧气储存）
    "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG",
    # 泛素（蛋白质降解标签）
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    # 组蛋白 H4（染色质结构）
    "MSGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG",
    # 热休克蛋白 HSP70（分子伴侣）
    "MAKAAAIGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVALNPQNTVFDAKRLIGRKFGDPVVQSDMKHWPFQVINDGDKPKVHVTKDLRSLIGRRFEDAEEADKKFLDKCNQVSFTVEGESLKDYLLDSGKDVNHLLHAEIAQLSEGKDQKIHDLVNKTKIPAVKDLKPKHIEEISENVEGLLGRFEIFNKTKPYIQVDIGGGQTKTFAPEEISAMVLTKMKETAEAYLGKTVTNAVVTVPAYFNDSQRQATKDAGAIAGLNVLRIINEPTAAAIAYGLDRTGKGERNVLIFDLGGGTFDVSILTIDDGIFEVKSTAGDTHLGGEDFDNRLVSHFVEEFKRKHKKDISENKRAVRRLRTACERAKRTLSSSTQASIEIDSLFEGIDFYTSITRARFEELCSDLFRSTLEPVEKALRDAKLDKAQIHDLVLVGGSTRIPKVQKLLQDFFNGRDLNKSINPDEAVAYGAAVQAAILMGDKSENVQDLLLLDVAPLSLGLETAGGVMTALIKRNSTIPTKQTQTFTTYSDNQPGVLIQVYEGERAMTKDNNLLGRFELSGIPPAPRGVPQIEVTFDIDANGILNVSAVDKSTGKENKITITNDKGRLSKEEIERMVQEAEKYKAEDEVQRERVAAKNALESYAFNMKSAVEDEGLKGKISEADKKKVLDKCQEVISWLDSNTLAEKEEFEHQQKELEKVCNPIISGLYQGAGGPGGFGAQAPKGGSGSGPTIEEVD",
    # 细胞色素 C（电子传递）
    "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",
    # 肌动蛋白（细胞骨架）
    "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF",
]


# ── TODO 1：加载模型和 tokenizer ──────────────────────────────
def load_esm2(model_name: str = MODEL_NAME, device: torch.device = torch.device('cpu')):
    """
    加载 ESM-2 的 tokenizer 和模型。

    步骤：
      1. AutoTokenizer.from_pretrained(model_name)
      2. AutoModel.from_pretrained(model_name)
      3. 将模型移动到 device
      4. 设置为 eval 模式

    返回：(tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


# ── TODO 2：提取单批次嵌入 ────────────────────────────────────
def embed_batch(
    tokenizer,
    model,
    sequences: list[str],
    device:    torch.device,
) -> torch.Tensor:
    """
    对一批序列提取 ESM-2 嵌入。

    步骤：
      1. tokenizer(sequences, return_tensors="pt",
                   padding=True, truncation=True, max_length=512)
      2. 将 inputs 移动到 device
      3. model(**inputs) 得到 outputs
      4. 取 outputs.last_hidden_state
         shape: (B, L+2, 320)  ← +2 是 <cls> 和 <eos>
      5. 去掉首尾特殊 token：hidden[:, 1:-1, :]
      6. 对 dim=1 取均值（Mean Pooling）

    注意：
      - 用 torch.no_grad()
      - 返回 shape: (B, 320)，dtype=torch.float32

    提示：
      - padding=True 会自动对齐 batch 内的序列长度
      - truncation=True 会截断超过 max_length 的序列
    """
    tokenizer_output = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokenizer_output['input_ids'].to(device)
    attention_mask = tokenizer_output['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L+2, 320)
        # 去掉首尾特殊 token
        hidden_states = hidden_states[:, 1:-1, :]  # (B, L, 320)
        # Mean Pooling
        embeddings = hidden_states.mean(dim=1)     # (B, 320)
    return embeddings


# ── TODO 3：分批处理大量序列 ──────────────────────────────────
def extract_embeddings(
    tokenizer,
    model,
    sequences:  list[str],
    batch_size: int = 4,
    device:     torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    对大量序列分批提取嵌入，返回 numpy 数组。

    步骤：
      1. 将 sequences 按 batch_size 分批
      2. 每批调用 embed_batch
      3. 收集所有结果，拼接后转为 numpy

    返回：(N, 320)
    """
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        embeddings = embed_batch(tokenizer, model, batch, device)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0).cpu().numpy()


# ── TODO 4：基本信息打印 ──────────────────────────────────────
def print_model_info(model):
    """
    打印模型的基本信息：
      - 总参数量
      - 层数（num_hidden_layers）
      - d_model（hidden_size）
      - 注意力头数（num_attention_heads）

    提示：
      - 总参数量：sum(p.numel() for p in model.parameters())
      - 模型配置：model.config
    """
    total = sum(p.numel() for p in model.parameters())
    cfg   = model.config
    print(f"模型名称：{cfg.model_type}")
    print(f"总参数量：{total:,}")
    print(f"Transformer 层数：{cfg.num_hidden_layers}")
    print(f"d_model（hidden_size）：{cfg.hidden_size}")
    print(f"注意力头数：{cfg.num_attention_heads}")


# ── TODO 5：主程序 ────────────────────────────────────────────
def main():
    device = torch.device('cpu')

    # 1. 加载模型
    print("加载 ESM-2 模型（首次运行会自动下载，约 31MB）...")
    tokenizer, model = load_esm2(device=device)
    print_model_info(model)
    print()

    # 2. 提取嵌入
    print(f"提取 {len(TEST_SEQUENCES)} 条蛋白质序列的嵌入...")
    embeddings = extract_embeddings(tokenizer, model, TEST_SEQUENCES, device=device)
    print(f"嵌入矩阵 shape：{embeddings.shape}")
    print(f"嵌入矩阵 dtype：{embeddings.dtype}")
    print()

    # 3. 单条序列测试
    print("── 单条序列测试 ──")
    single = TEST_SEQUENCES[1]   # 胰岛素
    print(f"序列长度：{len(single)}")
    tokens = tokenizer(single, return_tensors="pt")
    print(f"tokenizer 输出 keys：{list(tokens.keys())}")
    print(f"input_ids shape：{tokens['input_ids'].shape}")
    print(f"  ↑ 注意：长度 = 序列长度 + 2（<cls> 和 <eos>）")
    print()

    # 4. 对比 ProteinBERT 和 ESM-2 的嵌入维度
    print("── 嵌入维度对比 ──")
    print(f"ProteinBERT（Week 10）：d_model = 128，参数量 ≈ 470K")
    print(f"ESM-2 8M（Week 11）  ：d_model = 320，参数量 ≈ 8M")
    print()

    # 5. 计算序列两两余弦相似度（前 5 条）
    print("── 前 5 条序列的嵌入余弦相似度矩阵 ──")
    emb5 = embeddings[:5]
    # 归一化
    norms = np.linalg.norm(emb5, axis=1, keepdims=True)
    emb5_norm = emb5 / norms
    sim_matrix = emb5_norm @ emb5_norm.T
    labels = ["血红蛋白", "胰岛素", "GFP", "溶菌酶", "肌红蛋白"]
    print(f"{'':8}", end="")
    for l in labels:
        print(f"{l:8}", end="")
    print()
    for i, l in enumerate(labels):
        print(f"{l:8}", end="")
        for j in range(5):
            print(f"{sim_matrix[i,j]:8.3f}", end="")
        print()


if __name__ == "__main__":
    main()

# 输出
# 加载 ESM-2 模型（首次运行会自动下载，约 31MB）...
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 1720.47it/s]
# [transformers] EsmModel LOAD REPORT from: facebook/esm2_t6_8M_UR50D
# Key                       | Status     | Details
# --------------------------+------------+--------
# lm_head.layer_norm.bias   | UNEXPECTED |
# lm_head.dense.weight      | UNEXPECTED |
# lm_head.layer_norm.weight | UNEXPECTED |
# lm_head.bias              | UNEXPECTED |
# lm_head.dense.bias        | UNEXPECTED |
# pooler.dense.weight       | MISSING    |
# pooler.dense.bias         | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 模型名称：esm
# 总参数量：7,511,801
# Transformer 层数：6
# d_model（hidden_size）：320
# 注意力头数：20

# 提取 10 条蛋白质序列的嵌入...
# 嵌入矩阵 shape：(10, 320)
# 嵌入矩阵 dtype：float32

# ── 单条序列测试 ──
# 序列长度：110
# tokenizer 输出 keys：['input_ids', 'attention_mask']
# input_ids shape：torch.Size([1, 112])
#   ↑ 注意：长度 = 序列长度 + 2（<cls> 和 <eos>）

# ── 嵌入维度对比 ──
# ProteinBERT（Week 10）：d_model = 128，参数量 ≈ 470K
# ESM-2 8M（Week 11）  ：d_model = 320，参数量 ≈ 8M

# ── 前 5 条序列的嵌入余弦相似度矩阵 ──
#         血红蛋白    胰岛素     GFP     溶菌酶     肌红蛋白
# 血红蛋白       1.000   0.875   0.891   0.913   0.953
# 胰岛素        0.875   1.000   0.792   0.885   0.870
# GFP        0.891   0.792   1.000   0.860   0.882
# 溶菌酶        0.913   0.885   0.860   1.000   0.881
# 肌红蛋白       0.953   0.870   0.882   0.881   1.000

# Q1：tokenizer 输出的 input_ids、attention_mask 分别是什么？attention_mask 和 Week 10 的 src_key_padding_mask 有什么关系？
# input_ids 是 token 的索引序列，attention_mask 是对应的注意力掩码，用于指示哪些 token 是有效的，哪些是填充的。
# attention_mask 和 Week 10 的 src_key_padding_mask 类似，都是用于在计算注意力时忽略填充部分。

# Q2：为什么 Mean Pooling 时要去掉 [:, 1:-1, :]（首尾各一个 token）？如果不去掉会有什么影响？
# 因为首尾的 token 分别是 <cls> 和 <eos>，它们不属于实际的蛋白质序列。
# 如果不去掉，会导致平均池化的结果受到这两个特殊 token 的影响，从而影响嵌入的表示。

# Q3：看余弦相似度矩阵，血红蛋白和肌红蛋白的相似度是多少？和其他蛋白质对相比，这个数字说明了什么？
# 0.953，说明血红蛋白和肌红蛋白在嵌入空间中非常接近，可能因为它们在功能上都与氧气运输和储存相关，因此在序列特征上有一定的相似性。