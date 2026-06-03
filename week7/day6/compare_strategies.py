import torch
import torch.nn as nn
import esm
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import random
import numpy as np

# ════════════════════════════════════════════════════════════
# 数据（复用 Day 5）
# ════════════════════════════════════════════════════════════

HELIX_AA  = set("AVILM")
STRAND_AA = set("FYWST")
SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
AA_FREQ   = "AAAAACDDDEEEFFFGGHHIIKKLLLLMMNNPPQQRRRRSSSSTTTVVVWYY"

def generate_data(n=300, seed=42, noise=0.1):
    random.seed(seed)
    samples = []
    for _ in range(n):
        length = random.randint(50, 200)
        seq    = "".join(random.choice(AA_FREQ) for _ in range(length))
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
        if noise > 0:
            for k in range(length):
                if random.random() < noise:
                    labels[k] = random.choice(["H", "E", "C"])
        samples.append((seq, "".join(labels)))
    return samples

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
# 模型：统一接口，通过 unfreeze_last_n 控制策略
# ════════════════════════════════════════════════════════════

class ESMClassifier(nn.Module):
    def __init__(self, esm_model, d_model, num_classes=3, unfreeze_last_n=0):
        super().__init__()
        self.esm = esm_model
        self.unfreeze_last_n = unfreeze_last_n

        # 先全冻结
        for param in self.esm.parameters():
            param.requires_grad = False

        # 按需解冻
        if unfreeze_last_n > 0:
            for layer in self.esm.layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch_tokens):
        if self.unfreeze_last_n == 0:
            # 全冻结：可以用 no_grad 节省显存
            with torch.no_grad():
                results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        else:
            # 部分解冻：需要梯度流过
            results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])

        token_repr = results["representations"][self.esm.num_layers]
        x = token_repr[:, 1:, :]
        return self.classifier(x)


# ════════════════════════════════════════════════════════════
# 评估：返回 Acc + 每类 F1
# ════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, batch_converter, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for seqs, labels in loader:
            batch_data = [("", s) for s in seqs]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            labels = labels.to(device)

            logits  = model(batch_tokens)
            seq_len = labels.shape[1]
            logits  = logits[:, :seq_len, :]
            loss    = criterion(logits.reshape(-1, 3), labels.reshape(-1))
            total_loss += loss.item()

            mask   = labels.reshape(-1) != -1
            preds  = logits.reshape(-1, 3).argmax(dim=-1)
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels.reshape(-1)[mask].cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # TODO 1：计算每类 F1
    # 提示：
    #   f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1,2])
    #   返回长度为 3 的数组：[F1_H, F1_E, F1_C]
    f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1,2])


    return acc, f1, total_loss / len(loader)


# ════════════════════════════════════════════════════════════
# 训练一个策略，返回结果
# ════════════════════════════════════════════════════════════

def train_strategy(strategy_name, unfreeze_n,
                   train_loader, val_loader,
                   esm_model, alphabet, device,
                   epochs=10):

    batch_converter = alphabet.get_batch_converter()
    model = ESMClassifier(
        esm_model, d_model=esm_model.embed_dim,
        unfreeze_last_n=unfreeze_n
    ).to(device)

    # TODO 2：构建 optimizer
    # 规则：
    #   分类头始终用 lr=1e-3
    #   解冻的 ESM 层用 lr=1e-4
    #   如果 unfreeze_n == 0，只有分类头一个 param_group
    # 提示：
    #   if unfreeze_n == 0:
    #       optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    #   else:
    #       esm_params = [p for layer in model.esm.layers[-unfreeze_n:]
    #                     for p in layer.parameters() if p.requires_grad]
    #       optimizer = torch.optim.Adam([
    #           {"params": list(model.classifier.parameters()), "lr": 1e-3},
    #           {"params": esm_params, "lr": 1e-4},
    #       ])
    if unfreeze_n == 0:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    else:
        esm_params = [p for layer in model.esm.layers[-unfreeze_n:]
                      for p in layer.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": list(model.classifier.parameters()), "lr": 1e-3},
            {"params": esm_params, "lr": 1e-4},
        ])

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*50}")
    print(f"策略：{strategy_name}  |  可训练参数：{trainable:,}")
    print(f"{'─'*50}")
    print(f"{'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>8}  {'Time/ep':>8}")

    epoch_times = []
    for epoch in range(1, epochs + 1):
        # TODO 3：记录每个 epoch 的训练时间
        # 提示：
        #   t0 = time.time()
        #   ... 训练 ...
        #   elapsed = time.time() - t0
        t0 = time.time()

        elapsed = time.time() - t0

        model.train()
        total_correct, total_tokens = 0, 0

        t0 = time.time()  # ← 已给出，下面补训练循环
        for seqs, labels in train_loader:
            batch_data = [("", s) for s in seqs]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits  = model(batch_tokens)
            seq_len = labels.shape[1]
            logits  = logits[:, :seq_len, :]
            loss    = criterion(logits.reshape(-1, 3), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            mask  = labels.reshape(-1) != -1
            preds = logits.reshape(-1, 3).argmax(dim=-1)
            total_correct += (preds[mask] == labels.reshape(-1)[mask]).sum().item()
            total_tokens  += mask.sum().item()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        train_acc = total_correct / total_tokens
        val_acc, val_f1, _ = evaluate(
            model, val_loader, criterion, batch_converter, device
        )

        if epoch % 2 == 0:
            print(f"{epoch:>6}  {train_acc:>10.2%}  {val_acc:>8.2%}  {elapsed:>7.1f}s")

    # 最终评估
    final_acc, final_f1, _ = evaluate(
        model, val_loader, criterion, batch_converter, device
    )
    avg_time = sum(epoch_times) / len(epoch_times)

    return {
        "name":       strategy_name,
        "trainable":  trainable,
        "val_acc":    final_acc,
        "f1_H":       final_f1[0] if final_f1 is not None else 0,
        "f1_E":       final_f1[1] if final_f1 is not None else 0,
        "f1_C":       final_f1[2] if final_f1 is not None else 0,
        "avg_time":   avg_time,
    }


# ════════════════════════════════════════════════════════════
# Main：跑三种策略，输出对比表
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 注意：三种策略共享同一个 esm_model 实例会有问题
    # 每次都重新加载，保证参数干净
    all_data   = generate_data(n=300, noise=0.1)
    train_data = all_data[:240]
    val_data   = all_data[240:]

    train_loader = DataLoader(SSDataset(train_data), batch_size=8,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(SSDataset(val_data),   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    strategies = [
        ("A: 全冻结",       0),
        ("B: 解冻后2层",    2),
        ("C: 解冻后4层",    4),
    ]

    results = []
    for name, unfreeze_n in strategies:
        # 每次重新加载 ESM，保证权重干净
        esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        esm_model = esm_model.to(device)

        result = train_strategy(
            name, unfreeze_n,
            train_loader, val_loader,
            esm_model, alphabet, device,
            epochs=10
        )
        results.append(result)

    # 打印对比表
    print(f"\n\n{'═'*75}")
    print(f"{'策略':<14} {'可训练参数':>12} {'Val Acc':>8} {'F1-H':>6} {'F1-E':>6} {'F1-C':>6} {'秒/epoch':>9}")
    print(f"{'─'*75}")
    for r in results:
        print(f"{r['name']:<14} {r['trainable']:>12,} {r['val_acc']:>8.2%} "
              f"{r['f1_H']:>6.3f} {r['f1_E']:>6.3f} {r['f1_C']:>6.3f} "
              f"{r['avg_time']:>8.1f}s")
    print(f"{'═'*75}")

main()



# 对比表的完整输出（三行数据）
# ──────────────────────────────────────────────────
# 策略：A: 全冻结  |  可训练参数：963
# ──────────────────────────────────────────────────
#  Epoch   Train Acc   Val Acc   Time/ep
#      2      77.12%    77.06%      0.2s
#      4      77.10%    77.03%      0.2s
#      6      77.10%    77.03%      0.2s
#      8      77.30%    77.13%      0.2s
#     10      77.80%    77.57%      0.2s

# ──────────────────────────────────────────────────
# 策略：B: 解冻后2层  |  可训练参数：2,466,883
# ──────────────────────────────────────────────────
#  Epoch   Train Acc   Val Acc   Time/ep
#      2      78.26%    81.84%      0.3s
#      4      88.22%    87.75%      0.3s
#      6      91.84%    90.46%      0.3s
#      8      93.10%    91.83%      0.3s
#     10      93.50%    91.74%      0.3s

# ──────────────────────────────────────────────────
# 策略：C: 解冻后4层  |  可训练参数：4,932,803
# ──────────────────────────────────────────────────
#  Epoch   Train Acc   Val Acc   Time/ep
#      2      81.71%    86.30%      0.3s
#      4      92.21%    91.70%      0.3s
#      6      93.46%    92.05%      0.4s
#      8      94.01%    91.10%      0.4s
#     10      94.87%    90.14%      0.4s


# ═══════════════════════════════════════════════════════════════════════════
# 策略                    可训练参数  Val Acc   F1-H   F1-E   F1-C   秒/epoch
# ───────────────────────────────────────────────────────────────────────────
# A: 全冻结                  963   77.57%  0.003  0.177  0.872      0.3s
# B: 解冻后2层          2,466,883   91.74%  0.713  0.827  0.952      0.3s
# C: 解冻后4层          4,932,803   90.14%  0.688  0.796  0.941      0.4s
# ═══════════════════════════════════════════════════════════════════════════

# F1-H 和 F1-E 和 F1-C 的差距——哪类最难预测？为什么？

# F1-H，可能是因为螺旋结构本身的特性，相较于β折叠，α螺旋的组成更为多样
# Answer: 更根本的原因是α螺旋的数据量最小，有类别不平衡的影响。此外，生物学结构上，螺旋的边界很难判断

# 策略 C（解冻4层）比策略 B（解冻2层）更好吗？从哪个指标能看出来？

# 不一定更好，从 Val Acc可以看出来，解冻4层在延长了训练时间+导致更多参数的同时，效果却不如策略B