import torch
import torch.nn as nn
import esm
from torch.utils.data import Dataset, DataLoader
import random

# ════════════════════════════════════════════════════════════
# 数据部分
# ════════════════════════════════════════════════════════════

HELIX_AA    = set("AVILM")
STRAND_AA   = set("FYWST")
SS_TO_IDX   = {"H": 0, "E": 1, "C": 2}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class SSDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch):
    seqs   = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_len = max(len(l) for l in labels)
    padded  = [[SS_TO_IDX[c] for c in l] + [-1]*(max_len-len(l)) for l in labels]
    return seqs, torch.LongTensor(padded)


def generate_realistic_data(n=300, seed=42):
    random.seed(seed)
    AA_FREQ = "AAAAACDDDEEEFFFGGHHIIKKLLLLMMNNPPQQRRRRSSSSTTTVVVWYY"
    samples = []
    for _ in range(n):
        length = random.randint(50, 200)
        seq = "".join(random.choice(AA_FREQ) for _ in range(length))

        # 生成基础标签（Day 3 的规则）
        labels = ["C"] * length
        i = 0
        while i < length:
            if seq[i] in HELIX_AA:
                j = i
                while j < length and seq[j] in HELIX_AA: j += 1
                if j - i >= 3:
                    for k in range(i, j): labels[k] = "H"
                i = j
            elif seq[i] in STRAND_AA:
                j = i
                while j < length and seq[j] in STRAND_AA: j += 1
                if j - i >= 2:
                    for k in range(i, j): labels[k] = "E"
                i = j
            else:
                i += 1

        # TODO 1：加入 10% 标签噪声
        for k in range(length):
            if random.random() < 0.1:
                labels[k] = random.choice(["H", "E", "C"])

        samples.append((seq, "".join(labels)))
    return samples


# ════════════════════════════════════════════════════════════
# 模型（复用 Day 4）
# ════════════════════════════════════════════════════════════

class PartialESMClassifier(nn.Module):
    def __init__(self, esm_model, d_model, num_classes=3, unfreeze_last_n=2):
        super().__init__()
        self.esm = esm_model
        self.unfreeze_last_n = unfreeze_last_n
        for param in self.esm.parameters():
            param.requires_grad = False
        for layer in self.esm.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch_tokens):
        results    = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        token_repr = results["representations"][self.esm.num_layers]
        x = token_repr[:, 1:, :]
        return self.classifier(x)


def run_epoch(model, loader, optimizer, criterion,
              batch_converter, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            batch_data = [("", s) for s in seqs]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            labels = labels.to(device)
            if train: optimizer.zero_grad()
            logits  = model(batch_tokens)
            seq_len = labels.shape[1]
            logits  = logits[:, :seq_len, :]
            loss    = criterion(logits.reshape(-1, 3), labels.reshape(-1))
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
# Main
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)

    all_data   = generate_realistic_data(n=300)
    train_data = all_data[:240]
    val_data   = all_data[240:]

    print(f"训练集：{len(train_data)} 条，平均长度：{sum(len(s[0]) for s in train_data)//len(train_data)} aa")
    print(f"验证集：{len(val_data)} 条\n")

    # TODO 2：打印标签分布
    print("训练集标签分布：")
    total = sum(len(s[1]) for s in train_data)
    for ss_type in ["H", "E", "C"]:
        count = sum(s[1].count(ss_type) for s in train_data)
        print(f"  {ss_type}: {count:>6,}  ({count/total:.1%})")
    print()

    train_loader = DataLoader(SSDataset(train_data), batch_size=8,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(SSDataset(val_data),   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    model = PartialESMClassifier(
        esm_model, d_model=esm_model.embed_dim, unfreeze_last_n=2
    ).to(device)

    optimizer = torch.optim.Adam([
        {"params": list(model.classifier.parameters()), "lr": 1e-3},
        {"params": [p for layer in model.esm.layers[-2:]
                    for p in layer.parameters() if p.requires_grad], "lr": 1e-4},
    ])
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


# 训练集 H/E/C 分布是多少？和 Day 3-4 的均匀合成数据相比有什么不同？
# 训练集标签分布：
#   H:  2,744  (9.6%)
#   E:  3,811  (13.3%)
#   C: 22,093  (77.1%)

# 加噪声后 Val Acc 最终是多少？和 Day 4 的 97.58% 相比下降了多少？
#  Epoch  Train Loss   Train Acc   Val Acc
# ─────────────────────────────────────────────
#      2      0.5656      78.25%    81.26%
#      4      0.4227      87.99%    86.88%
#      6      0.3401      91.54%    90.70%
#      8      0.2878      93.07%    91.61%
#     10      0.2525      93.47%    91.41%

# Train Acc 和 Val Acc 的差距变大了还是变小了？你觉得说明了什么？
# 差距略有缩小，说明模型对真实数据的迁移能力更强
# Answer:
# Day 4 的 Train 99% 是"虚高"：
#   短序列 + 无噪声 + 规则极简
#   → 模型几乎把训练集"背"下来了
#   → 但 Val 也高，说明规则太简单，背下来就够用

# Day 5 的情况更健康：
#   Train 93% 没有继续逼近 100%
#   → 噪声阻止了模型过度拟合
#   → 这反而是一种"正则化效果"

# Train 和 Val 同步在 91-93% 附近收敛
#   → 说明模型学到的是真实规律，而不是在背答案
