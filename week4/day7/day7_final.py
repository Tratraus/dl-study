# day7_final.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 基础设置（全部自己写，不提供）
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE  = len(AMINO_ACIDS)
CHARGED     = set("KRHDE")
MAX_LEN     = 80
MIN_LEN     = 20

char2idx = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}

def onehot_encode_padded(seq, max_len=MAX_LEN):
    # 返回 (matrix, real_len)
    # matrix 形状：(max_len, VOCAB_SIZE)，填充部分全零
    matrix = np.zeros((max_len, VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
            matrix[i, char2idx[ch]] = 1.0
    return matrix, len(seq)


# ============================================================
# Part 2: 数据集生成
# ============================================================

def generate_multitask_dataset(n_samples=3000, random_state=42):
    """
    每条样本包含：
      seq     : 原始序列字符串（长度 20~80）
      label_a : 0/1，是否含信号肽
                规则：序列前5个氨基酸中，第一个是 M，
                      且后续4个中至少3个属于疏水氨基酸 AVILMFYW
      label_b : list[int]，每个位置是否带电荷（长度 = len(seq)）
    """
    rng = np.random.RandomState(random_state)
    HYDROPHOBIC = set("AVILMFYW")

    data = []
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # 正样本（含信号肽）
    for _ in range(n_pos):
        seq_len = rng.randint(MIN_LEN, MAX_LEN + 1)
        seq     = list(rng.choice(AMINO_ACIDS, size=seq_len))
        # 构造信号肽头部
        seq[0] = 'M'
        hydro_positions = rng.choice([1,2,3,4], size=3, replace=False)
        for p in hydro_positions:
            seq[p] = rng.choice(list(HYDROPHOBIC))
        seq_str = ''.join(seq)
        label_b = [1 if ch in CHARGED else 0 for ch in seq_str]
        data.append((seq_str, 1, label_b))

    # 负样本（不含信号肽）
    for _ in range(n_neg):
        while True:
            seq_len = rng.randint(MIN_LEN, MAX_LEN + 1)
            seq     = list(rng.choice(AMINO_ACIDS, size=seq_len))
            # 确保不满足信号肽条件
            seq[0]  = rng.choice([a for a in AMINO_ACIDS if a != 'M'])
            seq_str = ''.join(seq)
            label_b = [1 if ch in CHARGED else 0 for ch in seq_str]
            data.append((seq_str, 0, label_b))
            break

    rng.shuffle(data)
    return data


data = generate_multitask_dataset(n_samples=3000)
print(f"数据集大小：{len(data)}")
print(f"任务A 类别分布：pos={sum(d[1] for d in data)}, "
      f"neg={sum(1-d[1] for d in data)}")
print(f"示例序列（前20）：{data[0][0][:20]}...")
print(f"  label_a={data[0][1]}, label_b前5={data[0][2][:5]}")


# ============================================================
# Part 3: Dataset
# ============================================================

class MultiTaskDataset(Dataset):
    """
    你来实现：
    __init__：
      - 对每条序列调用 onehot_encode_padded，得到 (matrix, real_len)
      - label_b 也需要 padding 到 MAX_LEN（用 -1 填充，表示忽略）
      - 存储 self.X, self.lengths, self.label_a, self.label_b

    __getitem__：
      返回 (X[idx], label_a[idx], label_b[idx], lengths[idx])
    """
    def __init__(self, data):
      encoded = [onehot_encode_padded(d[0]) for d in data]

      # ✅ 先转 numpy array 再转 tensor，避免 UserWarning
      self.X       = torch.tensor(np.array([e[0] for e in encoded]), dtype=torch.float32)  # (N, MAX_LEN, VOCAB_SIZE)
      self.lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)               # (N,)
      self.label_a = torch.tensor([d[1] for d in data], dtype=torch.float32)               # (N,)

      # ✅ 先全填 -1，再按真实长度写入有效标签
      n = len(data)
      self.label_b = torch.full((n, MAX_LEN), -1.0, dtype=torch.float32)                  # (N, MAX_LEN)
      for i, d in enumerate(data):
          seq_len = self.lengths[i].item()                                                 # ✅ 0-d tensor → Python int
          self.label_b[i, :seq_len] = torch.tensor(d[2], dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.label_a[idx], self.label_b[idx], self.lengths[idx]


# ============================================================
# Part 4: 模型
# ============================================================

class MultiTaskBiLSTM(nn.Module):
    """
    结构：
      共享编码器：双向 LSTM（带 pack_padded_sequence）
      Head A：Linear(hidden*2, 1) + Sigmoid → 序列级别分类
      Head B：Linear(hidden*2, 1) + Sigmoid → 残基级别分类（每个位置）

    你来实现 forward：
      输入：x (batch, MAX_LEN, VOCAB_SIZE), lengths (batch,)
      输出：
        out_a : (batch,)         序列级别预测
        out_b : (batch, MAX_LEN) 残基级别预测
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(VOCAB_SIZE, hidden_size,
                            batch_first=True, bidirectional=True)
        self.head_a = nn.Linear(hidden_size * 2, 1)   # 序列分类
        self.head_b = nn.Linear(hidden_size * 2, 1)   # 残基分类

    def forward(self, x, lengths):
        # 你来写
        h0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)  # (num_layers*2, batch, hidden)
        c0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)  # (num_layers*2, batch, hidden)

        # 1. pack：让 LSTM 忽略 padding 位置的计算
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 2. 过 BiLSTM
        packed_out, (hn, cn) = self.lstm(packed, (h0, c0))

        # 3. unpack：还原为 (batch, MAX_LEN, hidden*2)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=MAX_LEN)

        # ── Head A：序列级别 ──────────────────────────────────────
        # 取前向最后一步 hn[0] 和后向最后一步 hn[1]，拼接
        last_hidden = torch.cat([hn[0], hn[1]], dim=-1)   # (batch, hidden*2)
        out_a = self.head_a(last_hidden).squeeze(-1)       # (batch,)
        out_a = torch.sigmoid(out_a)

        # ── Head B：残基级别 ──────────────────────────────────────
        out_b = self.head_b(out).squeeze(-1)               # (batch, MAX_LEN)
        out_b = torch.sigmoid(out_b)

        return out_a, out_b

# ============================================================
# Part 5: 损失函数（多任务）
# ============================================================

def multitask_loss(out_a, out_b, label_a, label_b, alpha=0.5):
    """
    out_a   : (batch,)          序列级别预测
    out_b   : (batch, MAX_LEN)  残基级别预测
    label_a : (batch,)          序列级别标签
    label_b : (batch, MAX_LEN)  残基级别标签（-1 表示 padding，需要忽略）
    alpha   : 两个任务的权重平衡

    你来实现：
    1. loss_a = BCELoss(out_a, label_a)
    2. loss_b：只计算 label_b != -1 的位置
    3. return alpha * loss_a + (1 - alpha) * loss_b
    """
    # 1. 任务 A 的损失
    loss_a = nn.BCELoss()(out_a, label_a)

    # 2. 任务 B 的损失（只算有效位置）
    mask = (label_b != -1)  # (batch, MAX_LEN)，True 表示有效位置
    if mask.sum() > 0:
        loss_b = nn.BCELoss()(out_b[mask], label_b[mask])
    else:
        loss_b = torch.tensor(0.0, device=out_b.device)

    # 3. 综合损失
    return alpha * loss_a + (1 - alpha) * loss_b


# ============================================================
# Part 6: 训练循环
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_a_correct = total_a_samples = 0
    total_b_correct = total_b_samples = 0

    for xb, la, lb, lengths in loader:
        optimizer.zero_grad()
        out_a, out_b = model(xb, lengths)
        loss = multitask_loss(out_a, out_b, la, lb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(la)

        # 任务 A 准确率
        pred_a = (out_a > 0.5).float()
        total_a_correct += (pred_a == la).sum().item()
        total_a_samples += len(la)

        # 任务 B 准确率（只算非 padding 位置）
        mask = (lb != -1)
        pred_b = (out_b > 0.5).float()
        total_b_correct += ((pred_b == lb) & mask).sum().item()
        total_b_samples += mask.sum().item()

    n = total_a_samples
    return (total_loss / n,
            total_a_correct / total_a_samples,
            total_b_correct / total_b_samples)


def evaluate(model, loader):
    # 你来写（参考 train_one_epoch，去掉梯度和 optimizer）
    model.eval()
    total_loss, total_a_correct, total_a_samples, total_b_correct, total_b_samples = 0.0, 0, 0, 0, 0

    with torch.no_grad():
        for xb, la, lb, lengths in loader:
            out_a, out_b = model(xb, lengths)
            loss = multitask_loss(out_a, out_b, la, lb)
            total_loss += loss.item() * len(la)

            # 任务 A 准确率
            pred_a = (out_a > 0.5).float()
            total_a_correct += (pred_a == la).sum().item()
            total_a_samples += len(la)

            # 任务 B 准确率（只算非 padding 位置）
            mask = (lb != -1)
            pred_b = (out_b > 0.5).float()
            total_b_correct += ((pred_b == lb) & mask).sum().item()
            total_b_samples += mask.sum().item()

    n = total_a_samples
    return (total_loss / n,
            total_a_correct / total_a_samples,
            total_b_correct / total_b_samples)


# ============================================================
# Part 7: 主训练流程
# ============================================================

def main():
    # 数据集划分
    train_data, temp_data = train_test_split(data, test_size=0.3,
                                              random_state=42)
    val_data,   test_data = train_test_split(temp_data, test_size=0.5,
                                              random_state=42)

    train_loader = DataLoader(MultiTaskDataset(train_data),
                              batch_size=32, shuffle=True)
    val_loader   = DataLoader(MultiTaskDataset(val_data),
                              batch_size=32, shuffle=False)
    test_loader  = DataLoader(MultiTaskDataset(test_data),
                              batch_size=32, shuffle=False)

    model     = MultiTaskBiLSTM(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping
    best_val_acc_a = 0.0
    patience       = 10
    counter        = 0
    best_state     = None

    print("开始训练 MultiTaskBiLSTM")
    print("=" * 65)

    for epoch in range(1, 81):
        tr_loss, tr_acc_a, tr_acc_b = train_one_epoch(
            model, train_loader, optimizer)
        val_loss, val_acc_a, val_acc_b = evaluate(model, val_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | "
                  f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
                  f"AccA {tr_acc_a:.4f}/{val_acc_a:.4f} | "
                  f"AccB {tr_acc_b:.4f}/{val_acc_b:.4f}")

        if val_acc_a > best_val_acc_a + 0.001:
            best_val_acc_a = val_acc_a
            counter        = 0
            best_state     = {k: v.clone()
                              for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                print(f"  ← Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, test_acc_a, test_acc_b = evaluate(model, test_loader)
    print(f"\n最终结果：")
    print(f"  任务A（信号肽检测）Test Acc：{test_acc_a:.4f}")
    print(f"  任务B（带电荷残基）Test Acc：{test_acc_b:.4f}")


if __name__ == "__main__":
    main()



# ============================================================
# Questions:
# ============================================================
# 问题 1： multitask_loss 里，为什么 label_b 的 padding 位置要用 -1 而不是 0？用 0 会有什么问题？
# 因为 0 是一个有效的标签值，如果用 0 作为 padding，会导致计算损失时把 padding 位置也算进去，影响模型训练。

# 问题 2： Head A 用的是 hn（最终隐藏状态），Head B 用的是 output（每个时间步的输出）。为什么这样设计？能不能反过来？
# 任务A 是序列级别分类，最终隐藏状态 hn 包含了整个序列的信息，适合做全局判断；
# 任务 B 是残基级别分类，每个时间步的 output 包含了当前位置的信息，更适合做逐位置判断。
# 反过来设计会导致信息不匹配，性能可能会下降。

# 问题 3： alpha=0.5 表示两个任务权重相等。如果任务 A 比任务 B 更重要，应该怎么调整 alpha？调大还是调小？
# 如果任务 A 更重要，应该调大 alpha（比如 0.7），让 loss_a 在总损失中占更大比例；如果任务 B 更重要，则调小 alpha（比如 0.3）。
