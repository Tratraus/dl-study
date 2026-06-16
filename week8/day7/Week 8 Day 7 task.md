# Week 8 Day 7：多任务学习初探

Week 8 的最后一天。今天把共享主干 + 多任务头的框架跑通。

---

## 最小必要理论

### 1. 多任务学习的核心思想

```text
单任务：
  ESM-2 → [SS3 Head] → 预测 H/E/C

多任务：
  ESM-2 → [SS3 Head]  → 预测 H/E/C
        → [Hydro Head] → 预测疏水/亲水
```

两个任务共享同一个 ESM-2 主干，各自有独立的任务头。

训练时两个任务的 loss 加权求和：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{SS3} + \lambda_2 \mathcal{L}_{Hydro}$$

今天用最简单的设置：$$\lambda_1 = \lambda_2 = 1.0$$

---

### 2. 疏水性标签怎么定义

疏水氨基酸（标准定义）：

```python
HYDROPHOBIC = set("AVILMFYW")  # 8 种
# 其余 12 种为亲水/极性
```

标签：

```text
疏水 → 1
亲水 → 0
```

这是一个 token-level 的二分类任务，和 SS3 结构完全一样，只是类别数从 3 变成 2。

---

### 3. 多任务 loss 的梯度流向

```text
L_total = L_SS3 + L_Hydro

L_total.backward()

→ 梯度同时流向：
    SS3 Head 的参数
    Hydro Head 的参数
    共享主干（ESM-2 可训练部分）的参数

共享主干收到的是两个任务梯度的叠加。
```

这就是多任务的"正则化效果"来源：

```text
单任务时：主干只被 SS3 的梯度塑造
多任务时：主干被 SS3 + Hydro 的梯度共同塑造
          两个任务的共同特征会被强化
          某一个任务的过拟合方向会被另一个任务抑制
```

---

## 代码任务

新建文件：`week8/day7/multitask.py`

```python
import torch
import torch.nn as nn
import random
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import esm


# ── 标签定义 ───────────────────────────────────────────────
AA       = list("ACDEFGHIKLMNPQRSTVWY")
SS_CLS   = ["H", "E", "C"]
SS2IDX   = {"H": 0, "E": 1, "C": 2}
HYDRO    = set("AVILMFYW")   # 疏水氨基酸


# ── 数据生成（新增疏水标签）────────────────────────────────
def generate_data(n=300, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        length = random.randint(50, 150)
        seq    = "".join(random.choices(AA, k=length))

        # SS3 标签（和之前一样）
        ss_labels = []
        for k in range(length):
            if seq[k] in "AVILM" and k+2 < length \
               and seq[k+1] in "AVILM" and seq[k+2] in "AVILM":
                ss_labels.append("H")
            elif seq[k] in "FYW":
                ss_labels.append("E")
            else:
                ss_labels.append("C")
        for k in range(length):
            if random.random() < 0.1:
                ss_labels[k] = random.choice(SS_CLS)

        # TODO 1：疏水性标签
        # 对序列中每个氨基酸：
        #   如果在 HYDRO 中 → 1，否则 → 0
        # 结果存为整数列表 hydro_labels
        hydro_labels = ...

        data.append((seq, "".join(ss_labels), hydro_labels))
    return data


class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    seqs        = [b[0] for b in batch]
    ss_labels   = [b[1] for b in batch]
    hydro_labels= [b[2] for b in batch]
    return seqs, ss_labels, hydro_labels


# ── 模型 ───────────────────────────────────────────────────
class MultiTaskESM(nn.Module):
    """
    共享主干（ESM-2，Adapter 微调）+ 两个任务头。
    """
    def __init__(self, bottleneck=64, n_adapter_layers=2):
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        d_model = self.esm.embed_dim  # 320

        # 冻结主干
        for param in self.esm.parameters():
            param.requires_grad = False

        # Adapter（复用 Week 8 Day 3 的方式）
        self.adapters = nn.ModuleList()
        num_layers    = len(self.esm.layers)
        for i in range(n_adapter_layers):
            adapter   = AdapterLayer(d_model, bottleneck)
            layer_idx = num_layers - n_adapter_layers + i
            self.adapters.append(adapter)
            def hook(module, input, output, adapter=adapter):
                hidden = adapter(output[0])
                return (hidden,) + output[1:]
            self.esm.layers[layer_idx].register_forward_hook(hook)

        # TODO 2：定义两个任务头
        # ss3_head：Linear(d_model, 3)
        # hydro_head：Linear(d_model, 2)
        self.ss3_head   = ...
        self.hydro_head = ...

    def forward(self, tokens):
        results = self.esm(tokens, repr_layers=[self.esm.num_layers])
        x = results["representations"][self.esm.num_layers]
        # x shape: (batch, seq_len+2, d_model)
        # 去掉首尾 special token
        x = x[:, 1:-1, :]  # (batch, seq_len, d_model)

        # TODO 3：分别通过两个任务头，返回两个 logits
        ss3_logits   = ...   # (batch, seq_len, 3)
        hydro_logits = ...   # (batch, seq_len, 2)
        return ss3_logits, hydro_logits

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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


# ── 训练一轮 ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    ss_correct = ss_tokens = hydro_correct = hydro_tokens = 0
    total_loss = 0

    for seqs, ss_list, hydro_list in loader:
        batch_data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = model.batch_converter(batch_data)
        tokens = tokens.to(device)

        ss3_logits, hydro_logits = model(tokens)

        # SS3 label tensor
        max_len   = max(len(lb) for lb in ss_list)
        ss_tensor = torch.full((len(ss_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(ss_list):
            ss_tensor[i, :len(lb)] = torch.tensor([SS2IDX[c] for c in lb])
        ss_tensor = ss_tensor.to(device)

        # TODO 4：构建 hydro_tensor
        # 和 ss_tensor 类似，但标签是 0/1 整数列表
        # padding 同样用 -100
        hydro_tensor = torch.full((len(hydro_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(hydro_list):
            ...
        hydro_tensor = hydro_tensor.to(device)

        # TODO 5：计算两个 loss，加权求和
        # L_total = L_ss3 + L_hydro
        loss_ss3   = criterion(ss3_logits.reshape(-1, 3),   ss_tensor.reshape(-1))
        loss_hydro = ...
        loss       = ...

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计 SS3 accuracy
        mask = ss_tensor != -100
        ss_correct += (ss3_logits.argmax(-1)[mask] == ss_tensor[mask]).sum().item()
        ss_tokens  += mask.sum().item()

        # 统计 Hydro accuracy
        hmask = hydro_tensor != -100
        hydro_correct += (hydro_logits.argmax(-1)[hmask] == hydro_tensor[hmask]).sum().item()
        hydro_tokens  += hmask.sum().item()

        total_loss += loss.item()

    return (total_loss / len(loader),
            ss_correct / ss_tokens,
            hydro_correct / hydro_tokens)


# ── 评估一轮 ────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    ss_correct = ss_tokens = hydro_correct = hydro_tokens = 0
    all_ss_preds = []; all_ss_labels = []
    all_h_preds  = []; all_h_labels  = []

    for seqs, ss_list, hydro_list in loader:
        batch_data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = model.batch_converter(batch_data)
        tokens = tokens.to(device)

        ss3_logits, hydro_logits = model(tokens)

        max_len   = max(len(lb) for lb in ss_list)
        ss_tensor = torch.full((len(ss_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(ss_list):
            ss_tensor[i, :len(lb)] = torch.tensor([SS2IDX[c] for c in lb])
        ss_tensor = ss_tensor.to(device)

        hydro_tensor = torch.full((len(hydro_list), max_len), -100, dtype=torch.long)
        for i, lb in enumerate(hydro_list):
            hydro_tensor[i, :len(lb)] = torch.tensor(lb)
        hydro_tensor = hydro_tensor.to(device)

        mask  = ss_tensor    != -100
        hmask = hydro_tensor != -100

        ss_preds    = ss3_logits.argmax(-1)
        hydro_preds = hydro_logits.argmax(-1)

        ss_correct    += (ss_preds[mask]    == ss_tensor[mask]).sum().item()
        ss_tokens     += mask.sum().item()
        hydro_correct += (hydro_preds[hmask] == hydro_tensor[hmask]).sum().item()
        hydro_tokens  += hmask.sum().item()

        all_ss_preds.extend(ss_preds[mask].cpu().numpy())
        all_ss_labels.extend(ss_tensor[mask].cpu().numpy())
        all_h_preds.extend(hydro_preds[hmask].cpu().numpy())
        all_h_labels.extend(hydro_tensor[hmask].cpu().numpy())

    ss_f1    = f1_score(all_ss_labels, all_ss_preds,
                        average=None, labels=[0, 1, 2])
    hydro_f1 = f1_score(all_h_labels,  all_h_preds,
                        average="binary")

    return (ss_correct / ss_tokens,
            ss_f1,
            hydro_correct / hydro_tokens,
            hydro_f1)


# ── 主函数 ──────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    all_data     = generate_data(n=300)
    train_loader = DataLoader(ProteinDataset(all_data[:240]),
                              batch_size=8, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(ProteinDataset(all_data[240:]),
                              batch_size=8, shuffle=False, collate_fn=collate_fn)

    set_seed(42)
    model = MultiTaskESM(bottleneck=64, n_adapter_layers=2).to(device)
    print(f"可训练参数量：{model.count_trainable():,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | "
          f"{'SS3 Acc':>8} {'Hydro Acc':>10} | "
          f"{'Val SS3':>8} {'F1-H':>6} | "
          f"{'Val Hydro':>10} {'F1-Hydro':>9}")
    print("-" * 90)

    for epoch in range(1, 11):
        t0 = time.time()
        train_loss, ss_acc, hydro_acc = train_epoch(
            model, train_loader, optimizer, device)
        val_ss_acc, ss_f1, val_hydro_acc, hydro_f1 = eval_epoch(
            model, val_loader, device)
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} | "
              f"{ss_acc:>8.4f} {hydro_acc:>10.4f} | "
              f"{val_ss_acc:>8.4f} {ss_f1[0]:>6.3f} | "
              f"{val_hydro_acc:>10.4f} {hydro_f1:>9.3f} | "
              f"{elapsed:.1f}s")

    # TODO 6：打印最终对比
    print("\n" + "="*60)
    print("Week 8 Day 7 最终结果")
    print("="*60)
    print(f"SS3  任务：Val Acc = {val_ss_acc:.4f}，F1-H = {ss_f1[0]:.3f}")
    print(f"Hydro任务：Val Acc = {val_hydro_acc:.4f}，F1   = {hydro_f1:.3f}")
    print(f"\n对比单任务 Adapter b=64（Day 4）：")
    print(f"  SS3 Val Acc：0.9296 → {val_ss_acc:.4f}  "
          f"({'↑' if val_ss_acc > 0.9296 else '↓'}"
          f"{abs(val_ss_acc - 0.9296):.4f})")
    print(f"  F1-H：       0.381  → {ss_f1[0]:.3f}  "
          f"({'↑' if ss_f1[0] > 0.381 else '↓'}"
          f"{abs(ss_f1[0] - 0.381):.3f})")

if __name__ == "__main__":
    main()
```

---

## 完成标准

1. 四个 TODO 全部填完，无报错
2. 打印出每轮的双任务指标
3. 最终对比打印出来

---

## 输出问题

**Q1**：多任务模型的可训练参数量是多少？和单任务 Adapter b=64（83,651）相比，多了多少？多出来的参数是哪里的？

**Q2**：多任务训练后，SS3 任务的 Val Acc 和 F1-H 相比单任务 Adapter b=64 是升了还是降了？你认为原因是什么？

**Q3**：Hydro 任务的 Val Acc 大概是多少？这个任务难吗？为什么？（提示：想想疏水性标签是怎么生成的，和序列有什么关系）

---

完成后把代码、完整运行输出和三个问题的回答发给我。