# week4/review.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE  = len(AMINO_ACIDS)
CHARGED     = set("KRHDE")
MAX_LEN     = 50
char2idx    = {ch: i for i, ch in enumerate(AMINO_ACIDS)}

# ============================================================
# Part 1：onehot_encode_padded
# 输入：seq（字符串），max_len
# 输出：(matrix, real_len)
#       matrix 形状 (max_len, VOCAB_SIZE)，padding 部分全零
# ============================================================

def onehot_encode_padded(seq, max_len=MAX_LEN):
    matrix = np.zeros((max_len, VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        if ch in char2idx:
            matrix[i, char2idx[ch]] = 1.0
    return matrix, len(seq)



# ============================================================
# Part 2：ReviewDataset
# 每条数据是 (seq_str, label_a, label_b_list)
#   label_a : 0/1 整数
#   label_b_list : 长度 = len(seq) 的 0/1 列表
#
# __init__ 需要：
#   self.X       : (N, MAX_LEN, VOCAB_SIZE)
#   self.lengths : (N,)
#   self.label_a : (N,)  float32
#   self.label_b : (N, MAX_LEN)  float32，padding 用 -1
#
# __getitem__ 返回：
#   (X[idx], label_a[idx], label_b[idx], lengths[idx])
# ============================================================

class ReviewDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.lengths = []
        self.label_a = []
        self.label_b = []

        for seq, la, lb in data:
            x, length = onehot_encode_padded(seq, MAX_LEN)
            self.X.append(x)
            self.lengths.append(length)
            self.label_a.append(la)
            lb_padded = lb + [-1] * (MAX_LEN - len(lb))
            self.label_b.append(lb_padded)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.lengths = torch.tensor(self.lengths, dtype=torch.int64)
        self.label_a = torch.tensor(self.label_a, dtype=torch.float32)
        self.label_b = torch.tensor(self.label_b, dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.label_a[idx], self.label_b[idx], self.lengths[idx]


# ============================================================
# Part 3：ReviewBiLSTM
# 共享编码器（双向 LSTM）+ 两个 Head
#
# forward 输入：x (batch, MAX_LEN, VOCAB_SIZE), lengths (batch,)
# forward 输出：
#   out_a : (batch,)        序列级别概率
#   out_b : (batch, MAX_LEN) 残基级别概率
#
# 要求：
#   - 使用 pack_padded_sequence / pad_packed_sequence
#   - Head A 用 hn 拼接
#   - Head B 用 output 每个位置
# ============================================================

class ReviewBiLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=VOCAB_SIZE, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.head_a = nn.Linear(hidden_size * 2, 1)  # 双向，所以乘以 2
        self.head_b = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        # 第一步：初始化 h0, c0
        h0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)  # (num_layers*2, batch, hidden)
        c0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)  # (num_layers*2, batch, hidden)

        # 第二步：pack
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first = True,enforce_sorted = False)
        # 第三步：过 BiLSTM
        packed_out, (hn, cn) = self.lstm(packed, (h0, c0))
        # 第四步：unpack
        out, _ = pad_packed_sequence(packed_out, batch_first = True,total_length = MAX_LEN)
        # 第五步：Head A，用 hn 拼接
        hn = hn.view(2, -1, self.hidden_size)
        hn = torch.cat([hn[0], hn[1]], dim = 1)
        out_a = torch.sigmoid(self.head_a(hn)).squeeze(-1)

        # 第六步：Head B，用 out 每个位置
        out_b = torch.sigmoid(self.head_b(out)).squeeze(-1)

        return out_a, out_b


# ============================================================
# Part 4：multitask_loss
# 输入：
#   out_a, label_a : (batch,)
#   out_b, label_b : (batch, MAX_LEN)，label_b 含 -1
#   alpha          : float
# 输出：加权总 loss
# ============================================================

def multitask_loss(out_a, out_b, label_a, label_b, alpha=0.5):
    loss_a = nn.BCELoss()(out_a, label_a)
    mask   = (label_b != -1)
    loss_b = nn.BCELoss()(out_b[mask], label_b[mask])
    return alpha * loss_a + (1 - alpha) * loss_b

# ============================================================
# Part 5：冒烟测试（写完后运行，全部通过才算完成）
# ============================================================

def smoke_test():
    import numpy as np

    rng = np.random.RandomState(0)

    # 造 10 条假数据
    fake_data = []
    for _ in range(10):
        seq_len = rng.randint(10, MAX_LEN + 1)
        seq     = ''.join(rng.choice(AMINO_ACIDS, size=seq_len))
        la      = int(rng.randint(0, 2))
        lb      = [int(rng.randint(0, 2)) for _ in range(seq_len)]
        fake_data.append((seq, la, lb))

    # Dataset
    ds     = ReviewDataset(fake_data)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    xb, la, lb, lengths = next(iter(loader))

    print(f"[Dataset]")
    print(f"  X      : {xb.shape}")       # (4, 50, 20)
    print(f"  label_a: {la.shape}")       # (4,)
    print(f"  label_b: {lb.shape}")       # (4, 50)
    print(f"  lengths: {lengths.tolist()}")
    print(f"  label_b[0, :5]   = {lb[0, :5].tolist()}")
    print(f"  label_b[0, -3:]  = {lb[0, -3:].tolist()}")  # 应含 -1

    # Model
    model  = ReviewBiLSTM(hidden_size=32)
    out_a, out_b = model(xb, lengths)
    print(f"\n[Model]")
    print(f"  out_a : {out_a.shape}")     # (4,)
    print(f"  out_b : {out_b.shape}")     # (4, 50)
    assert out_a.shape == (4,),           "out_a 形状错误"
    assert out_b.shape == (4, MAX_LEN),   "out_b 形状错误"
    assert out_a.min() >= 0 and out_a.max() <= 1, "out_a 不在 [0,1]"
    assert out_b.min() >= 0 and out_b.max() <= 1, "out_b 不在 [0,1]"

    # Loss
    loss = multitask_loss(out_a, out_b, la.float(), lb, alpha=0.5)
    print(f"\n[Loss]")
    print(f"  loss  : {loss.item():.4f}")
    assert loss.item() > 0, "loss 应该大于 0"

    print("\n✅ 全部通过")

smoke_test()
