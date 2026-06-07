# Week 8 Day 3：将 Adapter 插入 ESM-2

## 今天的任务目标

**冻结 ESM-2 所有原始参数，在最后 N 层后插入 Adapter，只训练 Adapter 参数，并验证原始权重未被修改。**

---

## 最小必要理论

### 1. 插入策略：用 forward hook

ESM-2 的每一层是 `esm.layers[i]`，它的 `forward` 我们不能直接改。

PyTorch 提供了 **register_forward_hook**：

```python
# hook 会在某个模块 forward 之后自动被调用
# 输入：module, 模块输入, 模块输出
# 返回值：如果返回非 None，则替换该模块的输出

def hook_fn(module, input, output):
 # output 是该层的原始输出
 # 我们在这里接上 Adapter
 return adapter(output)

layer.register_forward_hook(hook_fn)
```

这样不需要修改 ESM-2 的任何代码，只需要"挂钩子"。

---

### 2. ESM-2 每层的输出格式

ESM-2 的每一层（`TransformerLayer`）输出是一个**元组**：

```python
output = (hidden_states, ...)
# output[0] 才是我们需要的 hidden_states，shape: (batch, seq_len, d_model)
# output[1:] 是 attention weights 等附加信息
```

所以 hook 里需要这样处理：

```python
def hook_fn(module, input, output):
 hidden = output[0] # 取出 hidden_states
 hidden = adapter(hidden) # 过 Adapter
 return (hidden,) + output[1:] # 重新打包成元组返回
```

---

### 3. 冻结策略回顾

```python
# 冻结所有参数
for param in model.parameters():
 param.requires_grad = False

# 只解冻 Adapter 参数（Adapter 是新建的，默认 requires_grad=True）
# 不需要额外操作，新建的 nn.Module 参数默认可训练
```

---

### 4. 权重未被修改的验证方法

```python
# 训练前：保存某一层权重的快照
weight_before = model.esm.layers[-1].self_attn.q_proj.weight.data.clone()

# 训练若干步后：
weight_after = model.esm.layers[-1].self_attn.q_proj.weight.data

# 验证：
assert torch.equal(weight_before, weight_after), "ESM-2 权重被修改了！"
```

---

## 代码任务

新建文件：`week8/day3/esm_with_adapter.py`

```python
import torch
import torch.nn as nn
import esm
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

# ── 复用 Day 2 的 AdapterLayer ──────────────────────────────
class AdapterLayer(nn.Module):
 def __init__(self, d_model: int, bottleneck: int):
 super().__init__()
 self.down_proj = nn.Linear(d_model, bottleneck)
 self.act = nn.GELU()
 self.up_proj = nn.Linear(bottleneck, d_model)
 nn.init.zeros_(self.up_proj.weight)
 nn.init.zeros_(self.up_proj.bias)

 def forward(self, x):
 return x + self.up_proj(self.act(self.down_proj(x)))

 def count_parameters(self):
 return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── ESM + Adapter 主模型 ────────────────────────────────────
class ESMWithAdapter(nn.Module):
 def __init__(self, num_adapter_layers: int = 2, bottleneck: int = 64):
 super().__init__()

 # 加载 ESM-2 8M
 self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
 self.batch_converter = self.alphabet.get_batch_converter()
 d_model = self.esm.embed_dim # 320

 # TODO 1：冻结 ESM-2 所有参数
 ...

 # TODO 2：创建 Adapter 列表
 # 为最后 num_adapter_layers 层各创建一个 AdapterLayer
 # 存入 nn.ModuleList（不能用普通 list，否则参数不会被注册）
 self.adapters = nn.ModuleList([
 ...
 ])

 # TODO 3：注册 forward hook
 # 对最后 num_adapter_layers 层，各挂一个 hook
 # hook 里调用对应的 AdapterLayer
 num_layers = len(self.esm.layers)
 for i, adapter in enumerate(self.adapters):
 layer_idx = num_layers - num_adapter_layers + i
 # TODO 3a：写 hook 函数（注意闭包捕获 adapter）
 # TODO 3b：注册 hook
 ...

 # 分类头（3类：H/E/C）
 self.classifier = nn.Linear(d_model, 3)

 def forward(self, tokens):
 # 通过 ESM-2（hook 会自动在对应层后触发）
 results = self.esm(tokens, repr_layers=[self.esm.num_layers])
 x = results["representations"][self.esm.num_layers]
 # x shape: (batch, seq_len, d_model)
 logits = self.classifier(x)
 return logits

 def count_trainable_parameters(self):
 return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 合成数据（复用 Week 7）──────────────────────────────────
AA = list("ACDEFGHIKLMNPQRSTVWY")
SS = ["H", "E", "C"]

def generate_data(n=240, seed=42):
 random.seed(seed)
 data = []
 for _ in range(n):
 length = random.randint(50, 150)
 seq = "".join(random.choices(AA, k=length))
 labels = []
 for k in range(length):
 if seq[k] in "AVILM" and k+2 < length and seq[k+1] in "AVILM" and seq[k+2] in "AVILM":
 labels.append("H")
 elif seq[k] in "FYW":
 labels.append("E")
 else:
 labels.append("C")
 # 10% 标签噪声
 for k in range(length):
 if random.random() < 0.1:
 labels[k] = random.choice(SS)
 data.append((seq, "".join(labels)))
 return data

SS2IDX = {"H": 0, "E": 1, "C": 2}

class ProteinDataset(Dataset):
 def __init__(self, data):
 self.data = data
 def __len__(self):
 return len(self.data)
 def __getitem__(self, idx):
 return self.data[idx]


# ── 训练与验证 ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
 model.train()
 total_loss, total_correct, total_tokens = 0, 0, 0
 criterion = nn.CrossEntropyLoss()

 for seqs, labels_list in loader:
 batch_data = [(str(i), s) for i, s in enumerate(seqs)]
 _, _, tokens = model.batch_converter(batch_data)
 tokens = tokens.to(device)

 logits = model(tokens)
 # logits: (batch, seq_len+2, 3) ← ESM 会加 <cls> 和 <eos>
 # 去掉首尾特殊 token
 logits = logits[:, 1:-1, :]

 # 构建标签张量
 max_len = max(len(lb) for lb in labels_list)
 label_tensor = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
 for i, lb in enumerate(labels_list):
 idxs = torch.tensor([SS2IDX[c] for c in lb])
 label_tensor[i, :len(lb)] = idxs
 label_tensor = label_tensor.to(device)

 loss = criterion(logits.reshape(-1, 3), label_tensor.reshape(-1))
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 mask = label_tensor != -100
 preds = logits.argmax(-1)
 total_correct += (preds[mask] == label_tensor[mask]).sum().item()
 total_tokens += mask.sum().item()
 total_loss += loss.item()

 return total_loss / len(loader), total_correct / total_tokens


@torch.no_grad()
def eval_epoch(model, loader, device):
 model.eval()
 total_correct, total_tokens = 0, 0
 criterion = nn.CrossEntropyLoss()
 total_loss = 0

 for seqs, labels_list in loader:
 batch_data = [(str(i), s) for i, s in enumerate(seqs)]
 _, _, tokens = model.batch_converter(batch_data)
 tokens = tokens.to(device)

 logits = model(tokens)
 logits = logits[:, 1:-1, :]

 max_len = max(len(lb) for lb in labels_list)
 label_tensor = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
 for i, lb in enumerate(labels_list):
 idxs = torch.tensor([SS2IDX[c] for c in lb])
 label_tensor[i, :len(lb)] = idxs
 label_tensor = label_tensor.to(device)

 loss = criterion(logits.reshape(-1, 3), label_tensor.reshape(-1))
 mask = label_tensor != -100
 preds = logits.argmax(-1)
 total_correct += (preds[mask] == label_tensor[mask]).sum().item()
 total_tokens += mask.sum().item()
 total_loss += loss.item()

 return total_loss / len(loader), total_correct / total_tokens


def main():
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f"使用设备：{device}")

 # 数据
 all_data = generate_data(n=300)
 train_raw = all_data[:240]
 val_raw = all_data[240:]

 def collate_fn(batch):
 seqs = [b[0] for b in batch]
 labels = [b[1] for b in batch]
 return seqs, labels

 train_loader = DataLoader(ProteinDataset(train_raw), batch_size=8,
 shuffle=True, collate_fn=collate_fn)
 val_loader = DataLoader(ProteinDataset(val_raw), batch_size=8,
 shuffle=False, collate_fn=collate_fn)

 # 模型
 model = ESMWithAdapter(num_adapter_layers=2, bottleneck=64).to(device)

 # TODO 4：打印可训练参数量，并验证只有 Adapter 和 classifier 在训练
 print(f"可训练参数量：{model.count_trainable_parameters():,}")
 print("可训练模块列表：")
 for name, param in model.named_parameters():
 if param.requires_grad:
 print(f" {name:60s} {param.numel():>8,}")

 # TODO 5：保存训练前的 ESM-2 权重快照（用于事后验证）
 weight_before = ...

 # 优化器（只优化可训练参数）
 optimizer = torch.optim.Adam(
 filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
 )

 # 训练 10 轮
 print("\n开始训练...")
 for epoch in range(1, 11):
 train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
 val_loss, val_acc = eval_epoch(model, val_loader, device)
 print(f"Epoch {epoch:2d} | "
 f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
 f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

 # TODO 6：验证 ESM-2 原始权重未被修改
 weight_after = ...
 assert torch.equal(weight_before, weight_after), "❌ ESM-2 权重被修改了！"
 print("\n✅ 验证通过：ESM-2 原始权重在训练后未发生变化")

if __name__ == "__main__":
 main()
```

---

## 完成标准

1. 可训练参数列表里**只出现** `adapters.*` 和 `classifier.*`，没有 `esm.*`
2. 10 轮训练正常收敛，Val Acc 有上升趋势
3. 最后的权重验证打印 `✅ 验证通过`

---

## 输出问题

**Q1**：你的可训练参数列表里有哪些模块？总参数量是多少？和 Day 1 的估算（82,688 + classifier）对上了吗？

**Q2**：hook 函数里为什么要用闭包捕获 `adapter`？如果直接写 `self.adapters[i]` 会有什么问题？

**Q3**：训练 10 轮后，Val Acc 是多少？和 Week 7 解冻2层（Val Acc ≈ 92%）相比如何？

---

提交代码和运行结果后我来验收。
