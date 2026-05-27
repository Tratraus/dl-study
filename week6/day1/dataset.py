import torch
from torch.utils.data import Dataset, DataLoader
import random

# ════════════════════════════════════════════════════════════
# 1. 生成模拟 FASTA 文件
# ════════════════════════════════════════════════════════════

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SS_LABELS   = "HEC"

def generate_fasta_files(seq_path, ss_path, n_samples=500, seed=42):
    """
    生成两个 FASTA 格式的文件：
      seq_path : 氨基酸序列文件
      ss_path  : 二级结构标签文件

    格式：
      >SEQ_0001
      MKLVF...
      >SEQ_0002
      ACDEF...
    """
    random.seed(seed)

    with open(seq_path, "w") as f_seq, open(ss_path, "w") as f_ss:
        for i in range(n_samples):
            length = random.randint(20, 100)

            # 生成序列
            seq = "".join(random.choice(AMINO_ACIDS) for _ in range(length))

            # 生成标签（随机，模拟真实数据的复杂性）
            ss  = "".join(random.choice(SS_LABELS)   for _ in range(length))

            seq_id = f"SEQ_{i+1:04d}"
            f_seq.write(f">{seq_id}\n{seq}\n")
            f_ss.write( f">{seq_id}\n{ss}\n")

    print(f"已生成：{seq_path}（{n_samples} 条序列）")
    print(f"已生成：{ss_path}（{n_samples} 条标签）")


# ════════════════════════════════════════════════════════════
# 2. 解析 FASTA 文件
# ════════════════════════════════════════════════════════════

def parse_fasta(filepath):
    """
    解析 FASTA 文件，返回 dict：{seq_id: sequence_str}

    参数：
        filepath : str，FASTA 文件路径

    返回：
        records : dict，key = seq_id，value = 序列字符串

    提示：
        - 遇到以 ">" 开头的行 → 这是 ID 行，去掉 ">" 和换行符
        - 其他行 → 序列内容，拼接到当前 ID 对应的序列上
        - 注意：序列可能跨多行（本任务生成的是单行，但要写成通用形式）
    """
    records = {}
    current_id  = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # TODO: 保存上一条记录（如果有）
                # TODO: 解析新的 ID
                if current_id is not None:
                    records[current_id] = "".join(current_seq)  # 保存上一条记录
                current_id = line[1:]  # 去掉 ">"
                current_seq = []       # 重置序列列表

            else:
                # TODO: 把这一行拼接到当前序列
                current_seq.append(line)

        # TODO: 保存最后一条记录
        if current_id is not None:
            records[current_id] = "".join(current_seq)

    return records


# ════════════════════════════════════════════════════════════
# 3. Dataset 类
# ════════════════════════════════════════════════════════════

AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0
SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {0: "H", 1: "E", 2: "C"}

class ProteinDataset(Dataset):
    def __init__(self, seq_fasta, ss_fasta):
        """
        参数：
            seq_fasta : 氨基酸序列 FASTA 文件路径
            ss_fasta  : 二级结构标签 FASTA 文件路径
        """
        seq_records = parse_fasta(seq_fasta)
        ss_records  = parse_fasta(ss_fasta)

        # 只保留两个文件都有的 ID
        common_ids = sorted(set(seq_records) & set(ss_records))

        self.samples = []
        for seq_id in common_ids:
            seq = seq_records[seq_id]
            ss  = ss_records[seq_id]
            if len(seq) == len(ss):   # 长度必须一致
                self.samples.append((seq, ss))

        print(f"加载完成：{len(self.samples)} 条有效样本")

    def __len__(self):
        # TODO: 返回样本数量
        return len(self.samples)

    def __getitem__(self, i):
        """
        返回第 i 条样本：(seq_str, ss_str)
        不在这里做 padding，交给 collate_fn 统一处理
        """
        # TODO: 返回 self.samples[i]
        return self.samples[i]


# ════════════════════════════════════════════════════════════
# 4. collate_fn（变长序列 → 对齐的 batch）
# ════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    把一个 batch 的 (seq_str, ss_str) 转成 tensor

    参数：
        batch : List of (seq_str, ss_str)

    返回：
        tokens : (batch, max_len)  LongTensor
        labels : (batch, max_len)  LongTensor，padding 位置 = -1
        mask   : (batch, max_len)  BoolTensor，True = padding

    提示：
        - max_len = 这个 batch 里最长序列的长度
        - tokens 初始化为 0（PAD token）
        - labels 初始化为 -1（ignore_index）
        - mask   初始化为 True（全是 padding）
        - 然后逐条填入真实数据
    """
    # TODO: 实现 collate_fn
    seqs = [item[0] for item in batch]
    ss_labels = [item[1] for item in batch]

    max_len = max(len(seq) for seq in seqs)
    batch_size = len(batch)

    tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)
    mask   = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (seq, ss) in enumerate(zip(seqs, ss_labels)):
        for j, aa in enumerate(seq):
            tokens[i, j] = AA_TO_IDX.get(aa, 0)  # 不认识的 AA 当 PAD 处理
            labels[i, j] = SS_TO_IDX.get(ss[j], -1)  # 不认识的标签当 ignore_index 处理
        mask[i, :len(seq)] = False  # 前 len(seq) 个位置不是 padding

    return tokens, labels, mask

# ════════════════════════════════════════════════════════════
# 5. 测试
# ════════════════════════════════════════════════════════════

def test():
    # 生成文件
    generate_fasta_files("seq.fasta", "ss.fasta", n_samples=100)

    # 解析并建立 Dataset
    dataset = ProteinDataset("seq.fasta", "ss.fasta")
    print(f"数据集大小：{len(dataset)}")
    print(f"第 0 条：seq={dataset[0][0][:10]}...  ss={dataset[0][1][:10]}...")

    # 建立 DataLoader
    loader = DataLoader(
        dataset,
        batch_size  = 8,
        shuffle     = True,
        collate_fn  = collate_fn,
    )

    # 取第一个 batch 检查
    tokens, labels, mask = next(iter(loader))
    print(f"\n第一个 batch：")
    print(f"  tokens shape : {tokens.shape}")   # (8, max_len)
    print(f"  labels shape : {labels.shape}")   # (8, max_len)
    print(f"  mask   shape : {mask.shape}")     # (8, max_len)
    print(f"  tokens[0][:10] : {tokens[0][:10].tolist()}")
    print(f"  labels[0][:10] : {labels[0][:10].tolist()}")
    print(f"  mask[0][:10]   : {mask[0][:10].tolist()}")

    # 验证：labels 中 -1 的位置应该和 mask=True 的位置一致
    assert (labels == -1).all(dim=0).eq(mask.all(dim=0)).all(), \
        "labels 的 padding 位置和 mask 不一致！"
    print("\n✅ 通过")

test()
