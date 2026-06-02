import torch
import torch.nn as nn
import esm
from torch.utils.data import Dataset, DataLoader
import random

# ════════════════════════════════════════════════════════════
# 1. 数据生成（复用 Week 6 的规则）
# ════════════════════════════════════════════════════════════

HELIX_AA   = set("AVILM")
STRAND_AA  = set("FYWST")
SS_TO_IDX  = {"H": 0, "E": 1, "C": 2}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def make_ss_label(seq):
    labels = ["C"] * len(seq)
    i = 0
    while i < len(seq):
        if seq[i] in HELIX_AA:
            j = i
            while j < len(seq) and seq[j] in HELIX_AA:
                j += 1
            if j - i >= 3:
                for k in range(i, j):
                    labels[k] = "H"
            i = j
        elif seq[i] in STRAND_AA:
            j = i
            while j < len(seq) and seq[j] in STRAND_AA:
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    labels[k] = "E"
            i = j
        else:
            i += 1
    return "".join(labels)

def generate_data(n=300, min_len=20, max_len=60, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        seq = "".join(random.choice(AMINO_ACIDS) for _ in range(length))
        ss  = make_ss_label(seq)
        data.append((seq, ss))
    return data


# ════════════════════════════════════════════════════════════
# 2. Dataset + collate_fn
# ════════════════════════════════════════════════════════════

class SSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]   # (seq_str, ss_str)


def collate_fn(batch):
    seqs   = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_len = max(len(l) for l in labels)
    padded_labels = []
    for l in labels:
        padded = [SS_TO_IDX[c] for c in l] + [-1] * (max_len - len(l))
        padded_labels.append(padded)

    return seqs, torch.LongTensor(padded_labels)


# ════════════════════════════════════════════════════════════
# 3. 模型
# ════════════════════════════════════════════════════════════

class FrozenESMClassifier(nn.Module):
    def __init__(self, esm_model, d_model, num_classes=3):
        super().__init__()
        self.esm = esm_model

        # TODO 1：冻结 ESM-2 的所有参数
        # 一行 for 循环即可
        for param in self.esm.parameters():
            param.requires_grad = False

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch_tokens):
        # TODO 2：前向传播，分三步
        # 步骤 1：用 self.esm 提取最后一层表示
        #   with torch.no_grad():
        #       results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        #   token_repr = results["representations"][self.esm.num_layers]
        #   # shape: (batch, seq_len+2, d_model)
        #
        # 步骤 2：去掉 <cls>（第 0 位），保留 index 1 以后
        #   x = token_repr[:, 1:, :]
        #   # shape: (batch, seq_len+1, d_model)
        #   # 注意：这里保留了 <eos>，但它对应的标签位置是 -1，loss 会忽略
        #
        # 步骤 3：过分类头
        #   logits = self.classifier(x)
        #   return logits
        with torch.no_grad():
            results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        token_repr = results["representations"][self.esm.num_layers]
        x = token_repr[:, 1:, :]
        logits = self.classifier(x)
        return logits


# ════════════════════════════════════════════════════════════
# 4. 参数统计工具
# ════════════════════════════════════════════════════════════

def print_param_stats(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  总参数量：    {total:>10,}")
    print(f"  可训练参数：  {trainable:>10,}  ({trainable/total:.2%})")
    print(f"  冻结参数：    {frozen:>10,}  ({frozen/total:.2%})")


# ════════════════════════════════════════════════════════════
# 5. 训练 + 验证
# ════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, criterion,
              batch_converter, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            batch_data   = [("", s) for s in seqs]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            labels       = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits  = model(batch_tokens)        # (batch, seq_len+1, 3)
            seq_len = labels.shape[1]
            logits  = logits[:, :seq_len, :]     # 对齐标签长度

            loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))

            if train:
                loss.backward()
                optimizer.step()

            mask  = labels.reshape(-1) != -1
            preds = logits.reshape(-1, 3).argmax(dim=-1)
            total_correct += (preds[mask] == labels.reshape(-1)[mask]).sum().item()
            total_tokens  += mask.sum().item()
            total_loss    += loss.item()

    return total_loss / len(loader), total_correct / total_tokens


# ════════════════════════════════════════════════════════════
# 6. Main
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    # 加载 ESM
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)

    # 数据
    all_data   = generate_data(n=300)
    train_data = all_data[:240]
    val_data   = all_data[240:]

    train_loader = DataLoader(SSDataset(train_data), batch_size=16,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(SSDataset(val_data),   batch_size=16,
                              shuffle=False, collate_fn=collate_fn)

    # 模型
    model = FrozenESMClassifier(
        esm_model, d_model=esm_model.embed_dim
    ).to(device)

    print("── 参数统计 ──")
    print_param_stats(model)
    print()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Acc':>8}")
    print("─" * 45)

    for epoch in range(1, 11):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion,
            batch_converter, device, train=True
        )
        _, val_acc = run_epoch(
            model, val_loader, optimizer, criterion,
            batch_converter, device, train=False
        )
        if epoch % 2 == 0:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>10.2%}  {val_acc:>8.2%}")

main()

# 可训练参数量是多少？占总参数量的比例是多少？（参数统计那一栏）
# 总参数量：     7,513,437
# 可训练参数：         963  (0.01%)
# 冻结参数：     7,512,474  (99.99%)
# 10 轮后 Train Acc 和 Val Acc 大概是多少？
#  Epoch  Train Loss   Train Acc   Val Acc
# ─────────────────────────────────────────────
#      2      0.6654      84.50%    86.48%
#      4      0.5070      84.52%    86.48%
#      6      0.4594      84.52%    86.48%
#      8      0.4264      84.53%    86.44%
#     10      0.4017      84.55%    86.35%
# Train Acc 和 Val Acc 差距大吗？你觉得这说明了什么？
# 差距不大，说明模型被冻结参数后，几乎没有参数的更新。
# Answer:
# Train Acc ≈ 84.5%
# Val   Acc ≈ 86.4%

# Val Acc 反而略高于 Train Acc，这说明：

# 1. 没有过拟合
#    → 只有 963 个参数，根本没有能力"记住"训练集
#    → 模型的泛化能力完全来自 ESM-2 的预训练表示

# 2. Val 略高的原因
#    → 验证集只有 60 条，样本少，存在随机波动
#    → 不代表验证集"更容易"，只是统计噪声

# 3. 准确率从第 2 轮就稳定在 84-86%，之后几乎不动
#    → 说明 963 个参数的分类头很快就收敛了
#    → ESM-2 的表示已经足够好，线性分类头轻松学会
