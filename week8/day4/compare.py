import torch
import torch.nn as nn
import esm
import random
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

# ── AdapterLayer（复用）────────────────────────────────────
class AdapterLayer(nn.Module):
    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.act       = nn.GELU()
        self.up_proj   = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return x + self.up_proj(self.act(self.down_proj(x)))


# ── 基础模型（所有策略共用同一个 ESM 主干结构）──────────────
class ESMClassifier(nn.Module):
    """
    通用分类器，策略通过外部控制哪些参数可训练。
    """
    def __init__(self):
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        d_model = self.esm.embed_dim  # 320

        # 先全部冻结
        for param in self.esm.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(d_model, 3)
        self.adapters   = nn.ModuleList()  # 默认为空，按需填充
        self._hooks     = []               # 存储 hook handle，方便清理

    def forward(self, tokens):
        results = self.esm(tokens, repr_layers=[self.esm.num_layers])
        x = results["representations"][self.esm.num_layers]
        return self.classifier(x)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 四种策略的构建函数 ──────────────────────────────────────

def build_frozen(esm_model_cls):
    """策略 A：全冻结，只训练 classifier"""
    model = esm_model_cls()
    # ESM 已全部冻结，classifier 默认可训练
    return model


def build_unfreeze(esm_model_cls, n_layers=2):
    """策略 B：解冻最后 n_layers 层"""
    model = esm_model_cls()
    num_layers = len(model.esm.layers)
    for i in range(num_layers - n_layers, num_layers):
        for param in model.esm.layers[i].parameters():
            param.requires_grad = True
    return model


def build_adapter(esm_model_cls, bottleneck=64, n_layers=2):
    """策略 C/D：插入 Adapter"""
    model = esm_model_cls()
    d_model    = model.esm.embed_dim
    num_layers = len(model.esm.layers)

    for i in range(n_layers):
        adapter   = AdapterLayer(d_model, bottleneck)
        layer_idx = num_layers - n_layers + i
        model.adapters.append(adapter)

        # TODO 1：注册 hook（复用 Day 3 的写法）
        def hook(module, input, output, adapter=adapter):
            hidden = adapter(output[0])
            return (hidden,) + output[1:]
        model.esm.layers[layer_idx].register_forward_hook(hook)

    return model


# ── 数据（复用）────────────────────────────────────────────
AA     = list("ACDEFGHIKLMNPQRSTVWY")
SS_CLS = ["H", "E", "C"]
SS2IDX = {"H": 0, "E": 1, "C": 2}

def generate_data(n=300, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        length = random.randint(50, 150)
        seq    = "".join(random.choices(AA, k=length))
        labels = []
        for k in range(length):
            if seq[k] in "AVILM" and k+2 < length and seq[k+1] in "AVILM" and seq[k+2] in "AVILM":
                labels.append("H")
            elif seq[k] in "FYW":
                labels.append("E")
            else:
                labels.append("C")
        for k in range(length):
            if random.random() < 0.1:
                labels[k] = random.choice(SS_CLS)
        data.append((seq, "".join(labels)))
    return data

class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# ── 训练一轮 ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_tokens = 0, 0, 0

    for seqs, labels_list in loader:
        batch_data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = model.batch_converter(batch_data)
        tokens = tokens.to(device)

        logits = model(tokens)[:, 1:-1, :]

        max_len      = max(len(lb) for lb in labels_list)
        label_tensor = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(labels_list):
            label_tensor[i, :len(lb)] = torch.tensor([SS2IDX[c] for c in lb])
        label_tensor = label_tensor.to(device)

        loss = criterion(logits.reshape(-1, 3), label_tensor.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask = label_tensor != -100
        total_correct += (logits.argmax(-1)[mask] == label_tensor[mask]).sum().item()
        total_tokens  += mask.sum().item()
        total_loss    += loss.item()

    return total_loss / len(loader), total_correct / total_tokens


# ── 评估一轮（含 F1）────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_tokens = 0, 0, 0
    all_preds, all_labels = [], []

    for seqs, labels_list in loader:
        batch_data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = model.batch_converter(batch_data)
        tokens = tokens.to(device)

        logits = model(tokens)[:, 1:-1, :]

        max_len      = max(len(lb) for lb in labels_list)
        label_tensor = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(labels_list):
            label_tensor[i, :len(lb)] = torch.tensor([SS2IDX[c] for c in lb])
        label_tensor = label_tensor.to(device)

        loss  = criterion(logits.reshape(-1, 3), label_tensor.reshape(-1))
        mask  = label_tensor != -100
        preds = logits.argmax(-1)

        total_correct += (preds[mask] == label_tensor[mask]).sum().item()
        total_tokens  += mask.sum().item()
        total_loss    += loss.item()

        # TODO 2：收集预测和标签，用于 F1 计算
        # 提示：mask 是 bool 张量，用 [mask] 取出有效位置
        # 注意：需要 .cpu().numpy() 转换

        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(label_tensor[mask].cpu().numpy())

    # TODO 3：用 sklearn 计算每类 F1
    f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1,2])  # shape: (3,)，分别对应 H/E/C

    acc = total_correct / total_tokens
    return total_loss / len(loader), acc, f1


# ── 设置随机种子 ────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 单次实验 ────────────────────────────────────────────────
def run_experiment(name, model, train_loader, val_loader, device,
                   lr=1e-3, epochs=10):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    print(f"\n{'='*60}")
    print(f"策略：{name}  |  可训练参数：{model.count_trainable():,}")
    print(f"{'='*60}")

    epoch_times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - start
        epoch_times.append(elapsed)

        val_loss, val_acc, f1 = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:2d} | "
              f"Train {train_acc:.4f} | "
              f"Val {val_acc:.4f} | "
              f"F1-H {f1[0]:.3f} F1-E {f1[1]:.3f} F1-C {f1[2]:.3f} | "
              f"{elapsed:.1f}s")

    avg_time = np.mean(epoch_times)
    return {
        "name":       name,
        "params":     model.count_trainable(),
        "val_acc":    val_acc,
        "f1_H":       f1[0],
        "f1_E":       f1[1],
        "f1_C":       f1[2],
        "sec_epoch":  avg_time,
    }


# ── 主函数 ──────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    all_data     = generate_data(n=300)
    train_raw    = all_data[:240]
    val_raw      = all_data[240:]
    train_loader = DataLoader(ProteinDataset(train_raw), batch_size=8,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(ProteinDataset(val_raw),   batch_size=8,
                              shuffle=False, collate_fn=collate_fn)

    results = []

    # 策略 A：全冻结
    set_seed(42)
    model_A = build_frozen(ESMClassifier).to(device)
    results.append(run_experiment("A. 全冻结", model_A,
                                  train_loader, val_loader, device))

    # 策略 B：解冻2层
    set_seed(42)
    model_B = build_unfreeze(ESMClassifier, n_layers=2).to(device)
    results.append(run_experiment("B. 解冻2层", model_B,
                                  train_loader, val_loader, device))

    # 策略 C：Adapter b=64
    set_seed(42)
    model_C = build_adapter(ESMClassifier, bottleneck=64).to(device)
    results.append(run_experiment("C. Adapter b=64", model_C,
                                  train_loader, val_loader, device))

    # 策略 D：Adapter b=256
    set_seed(42)
    model_D = build_adapter(ESMClassifier, bottleneck=256).to(device)
    results.append(run_experiment("D. Adapter b=256", model_D,
                                  train_loader, val_loader, device))

    # TODO 4：打印汇总表格
    print("\n" + "="*80)
    print("汇总对比表")
    print("="*80)
    print(f"{'策略':<20} {'参数量':>10} {'Val Acc':>8} "
          f"{'F1-H':>6} {'F1-E':>6} {'F1-C':>6} {'秒/epoch':>9}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<20} {r['params']:>10,} {r['val_acc']:>8.4f} "
              f"{r['f1_H']:>6.3f} {r['f1_E']:>6.3f} {r['f1_C']:>6.3f} "
              f"{r['sec_epoch']:>9.1f}")

if __name__ == "__main__":
    main()

# 设备：cuda

# ============================================================
# 策略：A. 全冻结  |  可训练参数：963
# ============================================================
# Epoch  1 | Train 0.5860 | Val 0.7815 | F1-H 0.000 F1-E 0.000 F1-C 0.877 | 0.7s
# Epoch  2 | Train 0.7860 | Val 0.7818 | F1-H 0.000 F1-E 0.004 F1-C 0.877 | 0.2s
# Epoch  3 | Train 0.7975 | Val 0.8092 | F1-H 0.000 F1-E 0.281 F1-C 0.892 | 0.2s
# Epoch  4 | Train 0.8389 | Val 0.8632 | F1-H 0.000 F1-E 0.646 F1-C 0.920 | 0.2s
# Epoch  5 | Train 0.8866 | Val 0.9033 | F1-H 0.000 F1-E 0.821 F1-C 0.944 | 0.2s
# Epoch  6 | Train 0.9123 | Val 0.9165 | F1-H 0.000 F1-E 0.868 F1-C 0.952 | 0.2s
# Epoch  7 | Train 0.9182 | Val 0.9192 | F1-H 0.000 F1-E 0.877 F1-C 0.953 | 0.2s
# Epoch  8 | Train 0.9191 | Val 0.9196 | F1-H 0.000 F1-E 0.878 F1-C 0.954 | 0.2s
# Epoch  9 | Train 0.9193 | Val 0.9199 | F1-H 0.000 F1-E 0.880 F1-C 0.954 | 0.2s
# Epoch 10 | Train 0.9193 | Val 0.9199 | F1-H 0.000 F1-E 0.880 F1-C 0.954 | 0.2s

# ============================================================
# 策略：B. 解冻2层  |  可训练参数：2,466,883
# ============================================================
# Epoch  1 | Train 0.8614 | Val 0.9210 | F1-H 0.061 F1-E 0.878 F1-C 0.954 | 0.3s
# Epoch  2 | Train 0.9248 | Val 0.9250 | F1-H 0.212 F1-E 0.880 F1-C 0.957 | 0.2s
# Epoch  3 | Train 0.9316 | Val 0.9339 | F1-H 0.476 F1-E 0.880 F1-C 0.962 | 0.2s
# Epoch  4 | Train 0.9356 | Val 0.9307 | F1-H 0.428 F1-E 0.877 F1-C 0.960 | 0.2s
# Epoch  5 | Train 0.9442 | Val 0.9175 | F1-H 0.428 F1-E 0.854 F1-C 0.952 | 0.2s
# Epoch  6 | Train 0.9616 | Val 0.9216 | F1-H 0.394 F1-E 0.864 F1-C 0.955 | 0.2s
# Epoch  7 | Train 0.9761 | Val 0.9245 | F1-H 0.409 F1-E 0.869 F1-C 0.957 | 0.2s
# Epoch  8 | Train 0.9870 | Val 0.9266 | F1-H 0.430 F1-E 0.864 F1-C 0.958 | 0.2s
# Epoch  9 | Train 0.9924 | Val 0.9194 | F1-H 0.368 F1-E 0.858 F1-C 0.955 | 0.2s
# Epoch 10 | Train 0.9929 | Val 0.9122 | F1-H 0.376 F1-E 0.855 F1-C 0.949 | 0.2s

# ============================================================
# 策略：C. Adapter b=64  |  可训练参数：83,651
# ============================================================
# Epoch  1 | Train 0.7642 | Val 0.9199 | F1-H 0.000 F1-E 0.880 F1-C 0.954 | 0.2s
# Epoch  2 | Train 0.9194 | Val 0.9199 | F1-H 0.000 F1-E 0.880 F1-C 0.954 | 0.2s
# Epoch  3 | Train 0.9194 | Val 0.9200 | F1-H 0.007 F1-E 0.880 F1-C 0.954 | 0.2s
# Epoch  4 | Train 0.9200 | Val 0.9219 | F1-H 0.099 F1-E 0.880 F1-C 0.955 | 0.2s
# Epoch  5 | Train 0.9221 | Val 0.9248 | F1-H 0.238 F1-E 0.880 F1-C 0.957 | 0.2s
# Epoch  6 | Train 0.9248 | Val 0.9248 | F1-H 0.238 F1-E 0.880 F1-C 0.957 | 0.2s
# Epoch  7 | Train 0.9265 | Val 0.9263 | F1-H 0.292 F1-E 0.880 F1-C 0.958 | 0.2s
# Epoch  8 | Train 0.9269 | Val 0.9286 | F1-H 0.401 F1-E 0.880 F1-C 0.959 | 0.2s
# Epoch  9 | Train 0.9292 | Val 0.9288 | F1-H 0.424 F1-E 0.880 F1-C 0.959 | 0.2s
# Epoch 10 | Train 0.9313 | Val 0.9296 | F1-H 0.381 F1-E 0.880 F1-C 0.960 | 0.2s

# ============================================================
# 策略：D. Adapter b=256  |  可训练参数：329,795
# ============================================================
# Epoch  1 | Train 0.8256 | Val 0.9199 | F1-H 0.000 F1-E 0.880 F1-C 0.954 | 0.2s
# Epoch  2 | Train 0.9195 | Val 0.9202 | F1-H 0.014 F1-E 0.880 F1-C 0.954 | 0.3s
# Epoch  3 | Train 0.9225 | Val 0.9245 | F1-H 0.275 F1-E 0.880 F1-C 0.957 | 0.2s
# Epoch  4 | Train 0.9250 | Val 0.9248 | F1-H 0.230 F1-E 0.880 F1-C 0.957 | 0.2s
# Epoch  5 | Train 0.9278 | Val 0.9280 | F1-H 0.358 F1-E 0.880 F1-C 0.959 | 0.2s
# Epoch  6 | Train 0.9308 | Val 0.9320 | F1-H 0.436 F1-E 0.880 F1-C 0.961 | 0.2s
# Epoch  7 | Train 0.9328 | Val 0.9333 | F1-H 0.463 F1-E 0.880 F1-C 0.962 | 0.2s
# Epoch  8 | Train 0.9335 | Val 0.9313 | F1-H 0.412 F1-E 0.879 F1-C 0.961 | 0.2s
# Epoch  9 | Train 0.9330 | Val 0.9317 | F1-H 0.468 F1-E 0.880 F1-C 0.961 | 0.2s
# Epoch 10 | Train 0.9349 | Val 0.9326 | F1-H 0.454 F1-E 0.878 F1-C 0.962 | 0.2s

# ================================================================================
# 汇总对比表
# ================================================================================
# 策略                          参数量  Val Acc   F1-H   F1-E   F1-C   秒/epoch
# --------------------------------------------------------------------------------
# A. 全冻结                      963   0.9199  0.000  0.880  0.954       0.2
# B. 解冻2层               2,466,883   0.9122  0.376  0.855  0.949       0.2
# C. Adapter b=64          83,651   0.9296  0.381  0.880  0.960       0.2
# D. Adapter b=256        329,795   0.9326  0.454  0.878  0.962       0.2

# Q1：Adapter b=64 和解冻2层相比，参数量减少了多少倍？Val Acc 差距是多少个百分点？
# 大约30倍，差距1.74个百分点

# Q2：增大 bottleneck 从 64 到 256，Val Acc 提升了多少？参数量增加了多少？你认为这个"性价比"如何？
# val acc仅差距0.003，而参数量提升到了四倍，我认为提升微乎其微

# Q3：四种策略的 秒/epoch 有什么规律？为什么全冻结和 Adapter 的训练时间接近，而解冻2层更慢？
# 可能由于性能过剩，四者的差距未体现出来