# day5_padding.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# ============================================================
# Part 1: 编码（复用，新增 PAD token）
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN   = "<PAD>"
VOCAB       = AMINO_ACIDS   # PAD 不需要出现在词表里，用全零向量表示
VOCAB_SIZE  = len(AMINO_ACIDS)   # 20
MOTIF       = "LSEE"
MAX_LEN     = 50   # batch 内统一填充到这个长度

char2idx = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}

def onehot_encode_padded(seq, max_len=MAX_LEN):
    """
    把序列编码为 (max_len, VOCAB_SIZE) 的矩阵
    真实部分：正常 one-hot
    填充部分：全零向量（代表 PAD）
    同时返回真实长度
    """
    real_len = len(seq)
    matrix   = np.zeros((max_len, VOCAB_SIZE), dtype=np.float32)
    for i, ch in enumerate(seq):
        matrix[i, char2idx[ch]] = 1.0
    # 填充部分保持全零，不需要额外操作
    return matrix, real_len

# 测试
test_seq = "ATGC"
mat, length = onehot_encode_padded(test_seq, max_len=8)
print("编码测试（max_len=8）：")
print(f"  序列：{test_seq}，真实长度：{length}")
print(f"  矩阵形状：{mat.shape}")
print(f"  前5行：\n{mat[:5]}")   # 前4行有值，第5行全零


# ============================================================
# Part 2: 生成变长数据集
# ============================================================

def generate_variable_dataset(n_samples=2000, min_len=10, max_len=MAX_LEN,
                               motif=MOTIF, random_state=42):
    """
    生成变长蛋白质序列
    序列长度从 [min_len, max_len] 均匀采样
    正负样本各 50%
    """
    rng = np.random.RandomState(random_state)
    sequences, labels = [], []
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # 生成正样本
    for _ in range(n_pos):
        while True:
            seq_len = rng.randint(min_len, max_len + 1)   # 随机长度
            # motif 必须能放进去
            if seq_len < len(motif):
                continue
            seq = list(rng.choice(AMINO_ACIDS, size=seq_len))
            insert_pos = rng.randint(0, seq_len - len(motif) + 1)
            for j, ch in enumerate(motif):
                seq[insert_pos + j] = ch
            seq_str = ''.join(seq)
            if motif in seq_str:
                sequences.append(seq_str)
                labels.append(1)
                break

    # 生成负样本
    for _ in range(n_neg):
        while True:
            seq_len = rng.randint(min_len, max_len + 1)
            seq = ''.join(rng.choice(AMINO_ACIDS, size=seq_len))
            if motif not in seq:
                sequences.append(seq)
                labels.append(0)
                break

    combined = list(zip(sequences, labels))
    rng.shuffle(combined)
    sequences, labels = zip(*combined)
    return list(sequences), list(labels)


sequences, labels = generate_variable_dataset(n_samples=2000)
lengths_all = [len(s) for s in sequences]
print(f"\n数据集大小：{len(sequences)}")
print(f"序列长度分布：min={min(lengths_all)}, "
      f"max={max(lengths_all)}, "
      f"mean={np.mean(lengths_all):.1f}")
print(f"类别分布：0={labels.count(0)}, 1={labels.count(1)}")


# ============================================================
# Part 3: Dataset（需要返回长度信息）
# ============================================================

class VarLenProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels    = torch.tensor(labels, dtype=torch.float32)
        # 预编码所有序列
        encoded = [onehot_encode_padded(seq) for seq in sequences]
        matrices, lengths = zip(*encoded)
        self.X       = torch.tensor(np.array(matrices), dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        # 注意：需要同时返回 X、label 和 length
        return self.X[idx], self.labels[idx], self.lengths[idx]


seq_train, seq_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.3, random_state=42, stratify=labels)
seq_val, seq_test, y_val, y_test = train_test_split(
    seq_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_loader = DataLoader(VarLenProteinDataset(seq_train, y_train),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(VarLenProteinDataset(seq_val,   y_val),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(VarLenProteinDataset(seq_test,  y_test),
                          batch_size=32, shuffle=False)

# 验证 DataLoader 输出
xb, yb, lb = next(iter(train_loader))
print(f"\n一个 batch 的形状：")
print(f"  xb.shape = {xb.shape}")    # (32, 50, 20)
print(f"  yb.shape = {yb.shape}")    # (32,)
print(f"  lb.shape = {lb.shape}")    # (32,)  ← 长度信息
print(f"  lb（前8个）= {lb[:8]}")    # 应该是 10~50 之间的随机整数


# ============================================================
# Part 4: 带 Padding 处理的 LSTM 模型
# ============================================================

class PaddedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # x 形状：(batch, MAX_LEN, input_size)
        # lengths：每条序列的真实长度，形状 (batch,)

        # 你来写：
        # 1. 初始化 h0, c0（全零，注意 device）
        # 2. 用 pack_padded_sequence 打包 x
        #    packed = pack_padded_sequence(x, lengths.cpu(),
        #                                  batch_first=True,
        #                                  enforce_sorted=False)
        # 3. 把 packed 送入 self.lstm，得到 output_packed 和 (hn, cn)
        # 4. 用 hn.squeeze(0) 取最后隐藏状态
        # 5. 经过 fc + sigmoid，返回 (batch,) 形状
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output_packed, (hn, cn) = self.lstm(packed, (h0, c0))
        hn = hn.squeeze(0)
        out = torch.sigmoid(self.fc(hn)).squeeze(1)
        return out


# 测试
model = PaddedLSTMClassifier(VOCAB_SIZE, hidden_size=64)
xb, yb, lb = next(iter(train_loader))
out = model(xb, lb)
print(f"\nPaddedLSTMClassifier 输出形状：{out.shape}")   # (32,)
print(f"输出值范围：[{out.min():.3f}, {out.max():.3f}]") # 0~1 之间


# ============================================================
# Part 5: 训练函数（需要处理三元组 batch）
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, use_lengths=False):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch in loader:
        if use_lengths:
            xb, yb, lb = batch
            output = model(xb, lb)
        else:
            xb, yb = batch[:2]
            output = model(xb)
        optimizer.zero_grad()
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(yb)
        preds = (output > 0.5).float()
        total_correct += (preds == yb).sum().item()
        total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion, use_lengths=False):
# 你来写（参考 train_one_epoch，加上 use_lengths 参数）
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if use_lengths:
                xb, yb, lb = batch
                output = model(xb, lb)
            else:
                xb, yb = batch[:2]
                output = model(xb)
            loss = criterion(output, yb)
            total_loss    += loss.item() * len(yb)
            preds = (output > 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_samples += len(yb)
    return total_loss / total_samples, total_correct / total_samples

# ============================================================
# Part 6: Early Stopping（直接复用）
# ============================================================

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None

    def step(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ============================================================
# Part 7: 对比实验
# ============================================================

# 模型 A：不处理 padding（直接用 Day 4 的 LSTMClassifier）
class NaiveLSTM(nn.Module):
    """不处理 padding，把 PAD 当成真实输入"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size, device=x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return torch.sigmoid(self.fc(hn.squeeze(0))).squeeze(1)

def run_experiment(model, name, epochs=60, lr=0.0005,
                   patience=8, use_lengths=False):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper   = EarlyStopping(patience=patience)

    print(f"\n{'='*60}\n训练 {name}\n{'='*60}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, use_lengths)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, use_lengths)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f}")
        if stopper.step(val_acc, model):
            print(f"  ← Early stop! epoch={epoch}, "
                  f"best val acc={stopper.best_acc:.4f}")
            break
    stopper.restore(model)
    _, test_acc = evaluate(model, test_loader, criterion, use_lengths)
    print(f"→ Test Acc: {test_acc:.4f}")
    return test_acc

results = {}
results['NaiveLSTM (无padding处理)'] = run_experiment(
    NaiveLSTM(VOCAB_SIZE, 64), 'NaiveLSTM',
    use_lengths=False)
results['PaddedLSTM (有padding处理)'] = run_experiment(
    PaddedLSTMClassifier(VOCAB_SIZE, 64), 'PaddedLSTM',
    use_lengths=True)

print("\n" + "=" * 45)
print("对比结果")
print("=" * 45)
for name, acc in results.items():
    print(f"  {name:<30} {acc:.4f}")


# ============================================================
# Questions
# ============================================================

# 问题 1：
# pack_padded_sequence 需要传入 lengths.cpu()，为什么要加 .cpu()？
# 如果不加会发生什么？

# 你的回答：未指定设备，默认在 CPU 上，但如果输入 x 在 GPU 上，就会报错。
# 加上 .cpu() 可以确保 lengths 在 CPU 上，避免设备不匹配的错误。
# Answer: pack_padded_sequence 内部使用 lengths 做 CPU 端的索引计算
# （决定每个时间步处理哪些样本），这个操作必须在 CPU 上执行。

# 即使你的模型在 GPU 上训练，lengths 也必须是 CPU tensor。
# 如果不加 .cpu()，在 GPU 训练时会报：

#   RuntimeError: lengths must be on CPU

# 在 CPU 训练时不会报错（因为本来就在 CPU 上），
# 这也是为什么你今天没遇到这个错误——你在 CPU 上跑的。

# 问题 2：
# NaiveLSTM 把 PAD（全零向量）当成真实输入处理。
# 全零向量经过 LSTM 的输入门和遗忘门后，会对隐藏状态产生什么影响？
# 为什么短序列受到的影响比长序列更大？

# 你的回答：
# 会导致隐藏状态含有无效计算
# 短序列会比长序列添加更多的PAD，因此在后续时间步中，输入全零向量会导致更多的无效计算，可能会干扰模型对真实输入的学习和记忆。而长序列相对来说受到的影响较小，因为它们包含更多的真实输入，PAD 的比例较低。

# Answer:
# 全零向量输入 LSTM 时，各个门的行为：

#   输入门  i_t = σ(W_i · [h_{t-1}, 0] + b_i)
#              ≈ σ(W_i · h_{t-1} + b_i)   ← 只受上一步隐藏状态影响

#   遗忘门  f_t = σ(W_f · [h_{t-1}, 0] + b_f)
#              ≈ σ(W_f · h_{t-1} + b_f)   ← 同上

#   候选值  g_t = tanh(W_g · [h_{t-1}, 0] + b_g)
#              ≈ tanh(W_g · h_{t-1} + b_g) ← 偏置项主导

# 结果：
#   每一个 PAD 时间步都会让细胞状态 c_t 发生一次"漂移"
#   → 遗忘门会持续衰减之前记住的信息
#   → 经过 40 个 PAD 步后，原始序列的记忆几乎被冲走

# 短序列（比如长度 10）：
#   读完 10 个真实氨基酸后，还要经过 40 个 PAD 步
#   → 最终 h_T 里真实信息所剩无几

# 长序列（比如长度 45）：
#   只有 5 个 PAD 步
#   → h_T 里真实信息基本保留



# 问题 3：
# 如果一个 batch 里最长序列是 50，最短序列是 10，
# 不用 pack_padded_sequence，NaiveLSTM 要多做多少次无效计算？
# （用百分比表示）

# 你的回答：
# 最长序列是 50，最短序列是 10，那么每条序列平均有 (50 - 10) / 2 = 20 个 PAD。
# 也就是说，每条序列中有 20 / 50 = 40% 的时间步是无效计算。
# 因此，NaiveLSTM 要多做 40% 的无效计算。

# Answer:
# 你的算法：
#   平均 PAD 数 = (50 - 10) / 2 = 20
#   无效比例   = 20 / 50 = 40%

# 问题：
#   这里假设序列长度均值是 (50+10)/2 = 30，
#   但题目问的是"这一个 batch"，
#   batch 里最长 50、最短 10，
#   问的是对"最短序列"多做了多少无效计算。

# 正确算法（针对最短序列）：
#   最短序列真实长度 = 10
#   NaiveLSTM 强迫它跑满 50 步
#   无效步数 = 50 - 10 = 40 步
#   无效比例 = 40 / 50 = 80%

# 如果问的是整个 batch 平均：
#   平均序列长度 ≈ 30（均匀分布 [10,50] 的期望）
#   平均无效步数 = 50 - 30 = 20
#   平均无效比例 = 20 / 50 = 40%  ← 你算的这个

# 两种问法都有意义，但要区分清楚。
