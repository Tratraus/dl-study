# Week 8 Day 6：LoRA 注入 ESM-2 训练实验

今天把 `LoRALinear` 真正插入 ESM-2，跑完整训练，然后和 Day 4 的结果汇总对比。

---

## 今天的任务目标

```text
1. 把 LoRALinear 替换 ESM-2 最后 N 层的 Q/V 投影
2. 跑 10 epoch 训练
3. 输出和 Day 4 相同格式的汇总表，加入 LoRA 这一行
```

---

## 最小必要理论

### ESM-2 的注意力层结构

你需要知道 ESM-2 的 Q/V 投影在哪里：

```python
# ESM-2 的每一层
model.esm.layers[i]                    # TransformerLayer
model.esm.layers[i].self_attn          # MultiheadAttention
model.esm.layers[i].self_attn.q_proj   # nn.Linear(320, 320)
model.esm.layers[i].self_attn.v_proj   # nn.Linear(320, 320)
```

LoRA 通常只替换 Q 和 V，不替换 K，这是原论文的默认设置。

---

### 替换方式

不用 hook，直接把属性替换掉：

```python
original_q = layer.self_attn.q_proj          # 取出原始 Linear
lora_q     = LoRALinear(original_q, rank=8)  # 包装成 LoRALinear
layer.self_attn.q_proj = lora_q              # 替换回去
```

替换后，`layer.self_attn.q_proj` 就是你的 `LoRALinear`，
原始 ESM-2 的 forward 代码不需要任何修改，自动调用你的 `forward`。

这是 LoRA 比 Adapter 更优雅的地方：

```text
Adapter：需要 hook 拦截输出，侵入性更强
LoRA：  直接替换 Linear 属性，forward 逻辑完全不变
```

---

### 参数量估算

ESM-2 8M，替换最后 2 层的 Q/V：

```text
每个 LoRALinear：
  A: rank × 320
  B: 320 × rank

替换 Q 和 V，共 2 个 Linear，最后 2 层：
  总 LoRA 参数 = 4 × (rank × 320 + 320 × rank)
               = 4 × 2 × rank × 320
               = 8 × rank × 320

rank=8：
  = 8 × 8 × 320 = 20,480

加上 classifier：
  320 × 3 + 3 = 963

合计：≈ 21,443
```

---

## 代码任务

新建文件：`week8/day6/lora_esm.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score


# ── LoRALinear（复用 Day 5）────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base_linear = base_linear
        self.rank  = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features  = base_linear.in_features
        out_features = base_linear.out_features

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out    = self.base_linear(x)
        lora_hidden = F.linear(x, self.lora_A)
        lora_out    = F.linear(lora_hidden, self.lora_B)
        return base_out + self.scale * lora_out


# ── 模型 ───────────────────────────────────────────────────
import esm

class ESMWithLoRA(nn.Module):
    def __init__(self, rank=8, alpha=16, n_lora_layers=2):
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        d_model = self.esm.embed_dim  # 320

        # 先全部冻结
        for param in self.esm.parameters():
            param.requires_grad = False

        # TODO 1：替换最后 n_lora_layers 层的 Q/V 投影
        # 提示：
        #   num_layers = len(self.esm.layers)
        #   对 range(num_layers - n_lora_layers, num_layers) 中的每一层：
        #     取出 layer.self_attn.q_proj 和 v_proj
        #     用 LoRALinear 包装后替换回去
        num_layers = len(self.esm.layers)
        for i in range(num_layers - n_lora_layers, num_layers):
            layer = self.esm.layers[i]
            ...

        self.classifier = nn.Linear(d_model, 3)

    def forward(self, tokens):
        results = self.esm(tokens, repr_layers=[self.esm.num_layers])
        x = results["representations"][self.esm.num_layers]
        return self.classifier(x)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
            if seq[k] in "AVILM" and k+2 < length \
               and seq[k+1] in "AVILM" and seq[k+2] in "AVILM":
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


# ── 训练 / 评估（复用）─────────────────────────────────────
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
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        mask = label_tensor != -100
        total_correct += (logits.argmax(-1)[mask] == label_tensor[mask]).sum().item()
        total_tokens  += mask.sum().item()
        total_loss    += loss.item()
    return total_loss / len(loader), total_correct / total_tokens

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
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(label_tensor[mask].cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
    acc = total_correct / total_tokens
    return total_loss / len(loader), acc, f1

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ── 主函数 ──────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    all_data     = generate_data(n=300)
    train_loader = DataLoader(ProteinDataset(all_data[:240]),
                              batch_size=8, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(ProteinDataset(all_data[240:]),
                              batch_size=8, shuffle=False, collate_fn=collate_fn)

    set_seed(42)
    model = ESMWithLoRA(rank=8, alpha=16, n_lora_layers=2).to(device)

    # TODO 2：打印可训练参数量，并验证替换是否成功
    # 提示：打印 model.esm.layers[-1].self_attn.q_proj 的类型
    print(f"可训练参数量：{model.count_trainable():,}")
    print(f"q_proj 类型：{type(model.esm.layers[-1].self_attn.q_proj)}")
    print(f"v_proj 类型：{type(model.esm.layers[-1].self_attn.v_proj)}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    epoch_times = []
    for epoch in range(1, 11):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        epoch_times.append(time.time() - t0)
        val_loss, val_acc, f1 = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:2d} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
              f"F1-H {f1[0]:.3f} F1-E {f1[1]:.3f} F1-C {f1[2]:.3f} | "
              f"{epoch_times[-1]:.1f}s")

    avg_time = np.mean(epoch_times[1:])

    # TODO 3：打印最终汇总（把 Day 4 的结果也手动填进来对比）
    print("\n" + "="*80)
    print("Week 8 完整对比表（含 LoRA）")
    print("="*80)
    print(f"{'策略':<22} {'参数量':>10} {'Val Acc':>8} "
          f"{'F1-H':>6} {'F1-E':>6} {'F1-C':>6} {'秒/epoch':>9}")
    print("-"*80)

    # Day 4 结果（手动填入）
    day4_results = [
        ("A. 全冻结",        963,       0.9199, 0.000, 0.880, 0.954, 0.2),
        ("B. 解冻2层",  2_466_883,       0.9122, 0.376, 0.855, 0.949, 0.2),
        ("C. Adapter b=64",  83_651,    0.9296, 0.381, 0.880, 0.960, 0.2),
        ("D. Adapter b=256", 329_795,   0.9326, 0.454, 0.878, 0.962, 0.2),
    ]
    for name, params, acc, fh, fe, fc, t in day4_results:
        print(f"{name:<22} {params:>10,} {acc:>8.4f} "
              f"{fh:>6.3f} {fe:>6.3f} {fc:>6.3f} {t:>9.1f}")

    # LoRA 结果（今天跑出来的）
    print(f"{'E. LoRA r=8':<22} {model.count_trainable():>10,} {val_acc:>8.4f} "
          f"{f1[0]:>6.3f} {f1[1]:>6.3f} {f1[2]:>6.3f} {avg_time:>9.1f}")


if __name__ == "__main__":
    main()
```

---

## 完成标准

1. `TODO 1` 完成后，打印出：

```text
q_proj 类型：<class '__main__.LoRALinear'>
v_proj 类型：<class '__main__.LoRALinear'>
```

2. 可训练参数量在 **21,000 ~ 22,000** 之间（根据你的估算）

3. 跑完 10 epoch，打印完整汇总表

---

## 输出问题

**Q1**：你实际跑出的 LoRA 可训练参数量是多少？和上面的估算（21,443）是否一致？如果有差异，差在哪里？

**Q2**：对比 Adapter b=64（83,651 参数）和 LoRA r=8（约 21,443 参数），Val Acc 和 F1-H 各差多少？你认为 LoRA 的参数效率如何？

**Q3**：LoRA 替换的是 Q/V 投影，而不是 K 投影。你认为为什么通常不替换 K？（提示：从注意力机制的角度思考 Q、K、V 各自的作用）

---

完成后把代码、完整运行输出和三个问题的回答发给我。