import torch
import torch.nn as nn
import esm
from torch.utils.data import Dataset, DataLoader
import random

# ════════════════════════════════════════════════════════════
# 数据部分（与 Day 3 完全相同，直接复制）
# ════════════════════════════════════════════════════════════

HELIX_AA    = set("AVILM")
STRAND_AA   = set("FYWST")
SS_TO_IDX   = {"H": 0, "E": 1, "C": 2}
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
                for k in range(i, j): labels[k] = "H"
            i = j
        elif seq[i] in STRAND_AA:
            j = i
            while j < len(seq) and seq[j] in STRAND_AA:
                j += 1
            if j - i >= 2:
                for k in range(i, j): labels[k] = "E"
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
        data.append((seq, make_ss_label(seq)))
    return data

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


# ════════════════════════════════════════════════════════════
# 模型：支持解冻最后 N 层
# ════════════════════════════════════════════════════════════

class PartialESMClassifier(nn.Module):
    def __init__(self, esm_model, d_model, num_classes=3, unfreeze_last_n=2):
        super().__init__()
        self.esm = esm_model
        self.unfreeze_last_n = unfreeze_last_n

        # 第一步：先全部冻结（和 Day 3 一样）
        for param in self.esm.parameters():
            param.requires_grad = False

        # TODO 1：解冻最后 unfreeze_last_n 层
        #
        # ESM-2 的层存放在 self.esm.layers 里，是一个 ModuleList
        # 可以用负索引取最后 N 层：self.esm.layers[-unfreeze_last_n:]
        #
        # 提示：
        #   for layer in self.esm.layers[-unfreeze_last_n:]:
        #       for param in layer.parameters():
        #           param.requires_grad = True
        for layer in self.esm.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch_tokens):
        # TODO 2：前向传播
        # 注意：这里不能用 with torch.no_grad()！
        # 因为解冻的层需要计算梯度，no_grad 会阻止梯度流动
        #
        # 直接调用：
        #   results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        #   token_repr = results["representations"][self.esm.num_layers]
        #   x = token_repr[:, 1:, :]
        #   return self.classifier(x)
        results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        token_repr = results["representations"][self.esm.num_layers]
        x = token_repr[:, 1:, :]
        return self.classifier(x)


# ════════════════════════════════════════════════════════════
# 参数统计（与 Day 3 相同）
# ════════════════════════════════════════════════════════════

def print_param_stats(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  总参数量：    {total:>10,}")
    print(f"  可训练参数：  {trainable:>10,}  ({trainable/total:.2%})")
    print(f"  冻结参数：    {frozen:>10,}  ({frozen/total:.2%})")


# ════════════════════════════════════════════════════════════
# 构建分层学习率 optimizer
# ════════════════════════════════════════════════════════════

def build_optimizer(model, lr_head=1e-3, lr_top_layers=1e-4):
    # TODO 3：构建分层学习率 optimizer
    #
    # 需要把参数分成两组：
    #   组 1：分类头参数         → lr = lr_head
    #   组 2：ESM 解冻层的参数   → lr = lr_top_layers
    #
    # 提示：
    #   head_params      = list(model.classifier.parameters())
    #   esm_top_params   = [p for p in model.esm.layers[-model.unfreeze_last_n:].parameters()
    #                       if p.requires_grad]
    #   注意：ModuleList 不能直接用 [-n:]，需要先转成列表或用 for 循环
    #
    #   esm_top_params = []
    #   for layer in model.esm.layers[-model.unfreeze_last_n:]:
    #       esm_top_params += [p for p in layer.parameters() if p.requires_grad]
    #
    #   optimizer = torch.optim.Adam([
    #       {"params": head_params,    "lr": lr_head},
    #       {"params": esm_top_params, "lr": lr_top_layers},
    #   ])
    #   return optimizer
    head_params      = list(model.classifier.parameters())

    esm_top_params = []
    for layer in model.esm.layers[-model.unfreeze_last_n:]:
        esm_top_params += [p for p in layer.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": head_params,    "lr": lr_head},
        {"params": esm_top_params, "lr": lr_top_layers},
    ])
    return optimizer



# ════════════════════════════════════════════════════════════
# 训练循环（与 Day 3 相同）
# ════════════════════════════════════════════════════════════

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
            labels       = labels.to(device)

            if train: optimizer.zero_grad()

            logits  = model(batch_tokens)
            seq_len = labels.shape[1]
            logits  = logits[:, :seq_len, :]

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
# Main
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)

    all_data   = generate_data(n=300)
    train_data = all_data[:240]
    val_data   = all_data[240:]

    train_loader = DataLoader(SSDataset(train_data), batch_size=16,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(SSDataset(val_data),   batch_size=16,
                              shuffle=False, collate_fn=collate_fn)

    model = PartialESMClassifier(
        esm_model, d_model=esm_model.embed_dim, unfreeze_last_n=2
    ).to(device)

    print("── 参数统计 ──")
    print_param_stats(model)
    print()

    optimizer = build_optimizer(model, lr_head=1e-3, lr_top_layers=1e-4)
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


# 可训练参数量变成多少了？和 Day 3 的 963 相比增加了多少？
# ── 参数统计 ──
#   总参数量：     7,513,437
#   可训练参数：   2,466,883  (32.83%)
#   冻结参数：     5,046,554  (67.17%)

# Val Acc 有没有超过 Day 3 的 86.4%？
#  Epoch  Train Loss   Train Acc   Val Acc
# ─────────────────────────────────────────────
#      2      0.4924      84.52%    86.48%
#      4      0.2994      87.45%    89.80%
#      6      0.1678      93.60%    93.62%
#      8      0.0823      97.65%    96.34%
#     10      0.0347      99.41%    97.58%

# TODO 2 里为什么不能用 with torch.no_grad()？用自己的话解释一下。
# with torch.no_grad()会禁止所有梯度更新，如果使用了他，会导致非冻结的参数无法更新。