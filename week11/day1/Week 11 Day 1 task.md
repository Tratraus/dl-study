# Week 11 Day 1：加载 ESM-2，提取真实蛋白质嵌入

## 最小必要理论（10 分钟）

### 1. ESM-2 是什么？

ESM-2（Evolutionary Scale Modeling 2）是 Meta AI 在 2022 年发布的蛋白质语言模型系列。它和你在 Week 10 手写的 ProteinBERT 是**同一类架构**（Encoder-only + MLM 预训练），但有两个本质区别：

```text
ProteinBERT（Week 10）          ESM-2（Week 11）
─────────────────────────────────────────────────
训练数据：合成随机序列            训练数据：2.5 亿条真实蛋白质序列
参数量：  470K（mini）           参数量：8M ~ 15B（多个版本）
预训练：  loss 卡在 2.85         预训练：充分收敛，学到真实进化规律
嵌入质量：无法区分化学性质        嵌入质量：可直接用于功能预测
```

今天用的是 `esm2_t6_8M_UR50D`，这是最小版本：
- `t6`：6 层 Transformer
- `8M`：800 万参数
- `UR50D`：在 UniRef50 数据集上训练

---

### 2. HuggingFace 的使用模式

HuggingFace 的 `transformers` 库提供了统一的接口：

```text
AutoTokenizer   →  把氨基酸序列字符串转成 token ids
AutoModel       →  加载预训练权重，提取嵌入
```

和 Week 10 手写 `token2id` 的区别：

```text
Week 10（手写）：
  seq = "MKTAYIAKQRQISFVK"
  ids = [token2id[aa] for aa in seq]   # 手动查表

Week 11（HuggingFace）：
  tokenizer(["MKTAYIAKQRQISFVK"], return_tensors="pt")
  # 自动处理：特殊 token（<cls>/<eos>）、padding、attention_mask
```

---

### 3. ESM-2 的输出结构

调用 `model(**inputs)` 后，返回一个对象，关键字段：

```text
outputs.last_hidden_state   →  shape: (B, L+2, 320)
                                       ↑   ↑+2 是因为有 <cls> 和 <eos> 两个特殊 token
                                       ↑   320 是 ESM-2 8M 的 d_model
```

做 Mean Pooling 时，需要**排除** `<cls>` 和 `<eos>`，只对真实氨基酸位置取均值：

```text
last_hidden_state[:, 1:-1, :]   →  去掉首尾特殊 token
再对 dim=1 取均值               →  (B, 320)
```

---

### 4. 今天的目标

```text
输入：一组真实蛋白质序列（字符串列表）
         ↓
    ESM-2 tokenizer
         ↓
    ESM-2 模型（冻结，只做推理）
         ↓
    Mean Pooling（排除特殊 token）
         ↓
输出：(N, 320) 的嵌入矩阵
```

---

## 代码任务

新建 `week11/day1/esm2_embed.py`：

```python
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
    ...


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
    ...


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
    ...


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
```

---

## 完成标准

1. `load_esm2` 成功加载模型，打印出参数量约 800 万
2. `embed_batch` 返回 shape `(B, 320)`
3. `extract_embeddings` 对 10 条序列返回 `(10, 320)`
4. `input_ids.shape` 的第二维 = 序列长度 + 2（验证特殊 token 的存在）
5. 余弦相似度矩阵对角线全为 1.0

---

## 输出问题

**Q1**：`tokenizer` 输出的 `input_ids`、`attention_mask` 分别是什么？`attention_mask` 和 Week 10 的 `src_key_padding_mask` 有什么关系？

**Q2**：为什么 Mean Pooling 时要去掉 `[:, 1:-1, :]`（首尾各一个 token）？如果不去掉会有什么影响？

**Q3**：看余弦相似度矩阵，血红蛋白和肌红蛋白的相似度是多少？和其他蛋白质对相比，这个数字说明了什么？

---

准备好后提交代码和终端输出（包括相似度矩阵）。