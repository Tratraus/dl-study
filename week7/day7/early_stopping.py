import torch
import torch.nn as nn
import esm
import time, os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import random

# ════════════════════════════════════════════════════════════
# 数据 + 模型（复用 Day 6，直接复制）
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

class ESMClassifier(nn.Module):
    def __init__(self, esm_model, d_model, num_classes=3, unfreeze_last_n=4):
        super().__init__()
        self.esm = esm_model
        self.unfreeze_last_n = unfreeze_last_n
        for param in self.esm.parameters():
            param.requires_grad = False
        if unfreeze_last_n > 0:
            for layer in self.esm.layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch_tokens):
        if self.unfreeze_last_n == 0:
            with torch.no_grad():
                results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        else:
            results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
        token_repr = results["representations"][self.esm.num_layers]
        return self.classifier(token_repr[:, 1:, :])


# ════════════════════════════════════════════════════════════
# Early Stopping 类
# ════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, save_path="week7/day7/best_model.pt"):
        self.patience      = patience
        self.min_delta     = min_delta
        self.save_path     = save_path
        self.best_val      = 0.0
        self.counter       = 0
        self.should_stop   = False

    def step(self, val_acc, model):
        # TODO 1：实现 Early Stopping 逻辑
        #
        # 如果 val_acc 比 best_val 提升了超过 min_delta：
        #   → 更新 best_val
        #   → 保存模型权重到 self.save_path
        #   → 重置 counter = 0
        # 否则：
        #   → counter += 1
        #   → 如果 counter >= patience，设 should_stop = True
        #
        # 保存模型：torch.save(model.state_dict(), self.save_path)
        if val_acc - self.best_val > self.min_delta:
            self.best_val = val_acc
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


    def load_best(self, model):
        # TODO 2：加载最好的模型权重
        # torch.load_state_dict(torch.load(self.save_path))
        # 注意正确写法：model.load_state_dict(torch.load(self.save_path))
        model.load_state_dict(torch.load(self.save_path))


# ════════════════════════════════════════════════════════════
# 训练循环（加入 Early Stopping）
# ════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, batch_converter, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
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
            mask  = labels.reshape(-1) != -1
            preds = logits.reshape(-1, 3).argmax(dim=-1)
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels.reshape(-1)[mask].cpu().numpy())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
    return acc, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)

    all_data   = generate_data(n=300, noise=0.1)
    train_data = all_data[:240]
    val_data   = all_data[240:]

    train_loader = DataLoader(SSDataset(train_data), batch_size=8,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(SSDataset(val_data),   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    # 用策略 C（解冻4层）来演示 Early Stopping 的效果
    model = ESMClassifier(
        esm_model, d_model=esm_model.embed_dim, unfreeze_last_n=4
    ).to(device)

    esm_params = [p for layer in model.esm.layers[-4:]
                  for p in layer.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": list(model.classifier.parameters()), "lr": 1e-3},
        {"params": esm_params, "lr": 1e-4},
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # TODO 3：初始化 EarlyStopping
    # early_stopping = EarlyStopping(patience=3, min_delta=0.001,
    #                                save_path="best_model.pt")
    early_stopping = EarlyStopping(
        patience=3, min_delta=0.001, save_path="week7/day7/best_model.pt"
    )  # 替换这行

    print(f"{'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>8}  {'Counter':>8}  {'Best':>8}")
    print("─" * 55)

    for epoch in range(1, 31):   # 最多跑 30 轮
        # 训练
        model.train()
        total_correct, total_tokens = 0, 0
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

        train_acc = total_correct / total_tokens
        val_acc, _ = evaluate(model, val_loader, criterion,
                               batch_converter, device)

        # TODO 4：调用 early_stopping.step()，并打印 counter 和 best_val
        # early_stopping.step(val_acc, model)
        # print(f"{epoch:>6}  {train_acc:>10.2%}  {val_acc:>8.2%}  "
        #       f"{early_stopping.counter:>8}  {early_stopping.best_val:>8.2%}")
        # if early_stopping.should_stop:
        #     print(f"\nEarly stopping 触发！在第 {epoch} 轮停止")
        #     break

        early_stopping.step(val_acc, model)
        print(f"{epoch:>6}  {train_acc:>10.2%}  {val_acc:>8.2%}  "
              f"{early_stopping.counter:>8}  {early_stopping.best_val:>8.2%}")
        if early_stopping.should_stop:
            print(f"\nEarly stopping 触发！在第 {epoch} 轮停止")
            break

    # TODO 5：加载最好的模型，做最终评估
    # early_stopping.load_best(model)
    # final_acc, final_f1 = evaluate(model, val_loader, criterion,
    #                                 batch_converter, device)
    # print(f"\n最好模型的 Val Acc：{final_acc:.2%}")
    # print(f"F1-H: {final_f1[0]:.3f}  F1-E: {final_f1[1]:.3f}  F1-C: {final_f1[2]:.3f}")
    early_stopping.load_best(model)
    final_acc, final_f1 = evaluate(model, val_loader, criterion,
                                    batch_converter, device)
    print(f"\n最好模型的 Val Acc：{final_acc:.2%}")
    print(f"F1-H: {final_f1[0]:.3f}  F1-E: {final_f1[1]:.3f}  F1-C: {final_f1[2]:.3f}")

main()


# Early Stopping 在第几轮触发？
# 第8轮

# 加载最好模型后的 Val Acc，比第10轮的结果高多少？
# 92.02%，无第十轮

# 思考题：patience=3 这个值怎么选？太小和太大分别有什么问题？
# 太小可能导致模型还没充分训练就停止，太大可能导致训练时间过长且可能过拟合