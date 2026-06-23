# Week 10 Day 6：Fine-tuning 与对比实验

## 最小必要理论（10 分钟）

### 1. 今天要回答的核心问题

Day 5 的任务太简单，无法区分"预训练有没有用"。今天设计一个**更难的任务**，然后做三组对比：

```text
实验组 A：随机初始化 Encoder + 冻结 + 分类头
实验组 B：预训练 Encoder   + 冻结 + 分类头   ← Day 5 的做法
实验组 C：预训练 Encoder   + Fine-tuning      ← 今天新增
```

通过对比 A vs B，看预训练有没有带来更好的特征表示。
通过对比 B vs C，看 Fine-tuning 能否进一步提升性能。

---

### 2. 更难的分类任务：Motif 检测

**任务定义**：

```text
正样本：序列中存在连续的疏水氨基酸簇（≥ 3 个连续疏水 AA）
负样本：序列中不存在这样的簇

疏水氨基酸：A, V, I, L, M, F, W, P
```

为什么这个任务更难？

- **不能靠频率统计**：正负样本中疏水氨基酸的总比例可以相同，但排列方式不同
- **需要局部上下文**：模型必须理解"相邻位置"的关系，而不只是单个 token 的信息
- **Transformer 的注意力机制天然适合这类任务**

---

### 3. Fine-tuning 的实现要点

Fine-tuning 和冻结 Encoder 的代码差异只有两处：

```python
# 冻结 Encoder（Day 5）
for param in encoder.parameters():
    param.requires_grad = False
optimizer = Adam(classifier.parameters(), lr=1e-3)

# Fine-tuning（今天）
for param in encoder.parameters():
    param.requires_grad = True          # ← 解冻
optimizer = Adam([
    {'params': encoder.parameters(),   'lr': 1e-4},   # ← Encoder 用小学习率
    {'params': classifier.parameters(),'lr': 1e-3},   # ← 分类头用大学习率
])
```

**为什么 Encoder 用更小的学习率？**

预训练 Encoder 已经学到了有用的特征，用大学习率更新会"破坏"这些特征（称为**灾难性遗忘**，Catastrophic Forgetting）。用小学习率可以在保留预训练知识的同时，让 Encoder 微调以适应新任务。

---

### 4. 评估指标

除了 accuracy，今天还加一个指标：**每组实验的收敛速度**（达到 80% accuracy 需要多少步）。

---

## 代码任务

新建 `week10/day6/finetune.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day5'))

from mlm_data import AA_IDS, token2id, PAD
from protein_bert import ProteinBERT
from extract_embedding import mean_pooling
from classify import ClassificationHead


# ── 常量 ──────────────────────────────────────────────────────
HYDROPHOBIC = {'A','V','I','L','M','F','W','P'}
HYDROPHOBIC_IDS = {token2id[aa] for aa in HYDROPHOBIC}
MOTIF_LEN = 3   # 连续疏水氨基酸的最短长度


# ── TODO 1：Motif 任务数据生成 ────────────────────────────────
def has_hydrophobic_motif(seq: list[int]) -> bool:
    """
    判断序列中是否存在长度 ≥ MOTIF_LEN 的连续疏水氨基酸簇。

    提示：用滑动窗口，检查是否有连续 MOTIF_LEN 个位置都是疏水 AA。
    """
    ...


def make_motif_batch(
    batch_size: int,
    seq_len:    int,
    device:     torch.device,
):
    """
    生成 Motif 检测任务的数据。

    正样本生成策略：
      - 随机生成序列
      - 在随机位置强制插入一段长度为 MOTIF_LEN 的疏水簇
      - 确保 has_hydrophobic_motif 返回 True

    负样本生成策略：
      - 随机生成序列，但疏水 AA 的权重设为 0（完全排除疏水 AA）
      - 确保 has_hydrophobic_motif 返回 False

    返回：(src, labels)
      src:    (batch_size, seq_len)  dtype=torch.long
      labels: (batch_size,)          dtype=torch.long
    """
    half = batch_size // 2

    # 正样本
    pos_seqs = []
    for _ in range(half):
        seq = random.choices(AA_IDS, k=seq_len)
        # 在随机位置插入疏水簇
        insert_pos = random.randint(0, seq_len - MOTIF_LEN)
        for j in range(MOTIF_LEN):
            seq[insert_pos + j] = random.choice(list(HYDROPHOBIC_IDS))
        pos_seqs.append(seq)

    # 负样本
    weights_neg = [0 if i in HYDROPHOBIC_IDS else 1 for i in AA_IDS]
    neg_seqs = [random.choices(AA_IDS, weights=weights_neg, k=seq_len)
                for _ in range(half)]

    all_seqs   = pos_seqs + neg_seqs
    all_labels = [1] * half + [0] * half

    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)

    src    = torch.tensor(list(all_seqs),   dtype=torch.long, device=device)
    labels = torch.tensor(list(all_labels), dtype=torch.long, device=device)
    return src, labels


# ── TODO 2：通用训练函数 ──────────────────────────────────────
def train_model(
    encoder:    ProteinBERT,
    finetune:   bool  = False,
    num_steps:  int   = 500,
    batch_size: int   = 64,
    seq_len:    int   = 40,
    log_every:  int   = 50,
) -> tuple[ClassificationHead, list[float]]:
    """
    通用训练函数，支持冻结和 Fine-tuning 两种模式。

    finetune=False（冻结模式）：
      - 冻结 encoder 所有参数
      - optimizer 只优化 classifier
      - encoder 提取嵌入时用 torch.no_grad()

    finetune=True（Fine-tuning 模式）：
      - encoder 参数解冻（requires_grad=True）
      - optimizer 使用分组学习率：
          encoder:    lr = 1e-4
          classifier: lr = 1e-3
      - encoder 需要参与梯度计算（不用 no_grad）
      - encoder.train() 模式

    两种模式共用：
      - make_motif_batch 生成数据
      - mean_pooling 提取嵌入
      - ClassificationHead 分类
      - cross_entropy loss
      - 每 log_every 步记录 loss 和 accuracy

    返回：(classifier, acc_history)
    """
    device = next(encoder.parameters()).device
    classifier = ClassificationHead().to(device)

    if not finetune:
        # 冻结模式
        ...
    else:
        # Fine-tuning 模式
        ...

    acc_history = []

    for step in range(num_steps + 1):
        src, labels = make_motif_batch(batch_size, seq_len, device)

        if not finetune:
            with torch.no_grad():
                embeddings = mean_pooling(encoder, src)
        else:
            embeddings = mean_pooling(encoder, src)

        logits = classifier(embeddings)
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc   = (preds == labels).float().mean().item()
        acc_history.append(acc)

        if step % log_every == 0:
            mode = "FT" if finetune else "FR"
            print(f"[{mode}] Step {step:4d} | loss: {loss:.4f} | acc: {acc:.4f}")

    return classifier, acc_history


# ── TODO 3：三组对比实验 ──────────────────────────────────────
def run_experiments():
    """
    运行三组对比实验并汇总结果。

    实验组：
      A. 随机初始化 Encoder + 冻结
      B. 预训练 Encoder     + 冻结
      C. 预训练 Encoder     + Fine-tuning

    每组实验结束后，在独立的测试集上评估准确率。
    最后打印汇总表格：

      ┌─────────────────────────────────────────────────────┐
      │              Motif 检测任务对比实验结果               │
      ├──────┬──────────────────────────┬───────────────────┤
      │ 实验 │ 配置                     │ 测试集准确率       │
      ├──────┼──────────────────────────┼───────────────────┤
      │  A   │ 随机 Encoder + 冻结      │ 0.XXXX            │
      │  B   │ 预训练 Encoder + 冻结    │ 0.XXXX            │
      │  C   │ 预训练 Encoder + FT      │ 0.XXXX            │
      └──────┴──────────────────────────┴───────────────────┘
    """
    device = torch.device('cpu')
    ckpt_path = 'week10/day3/protein_bert_mlm.pt'

    # ── 实验 A：随机 Encoder + 冻结 ──
    print("=" * 50)
    print("实验 A：随机初始化 Encoder + 冻结")
    print("=" * 50)
    encoder_A = ProteinBERT().to(device)   # 不加载 checkpoint
    clf_A, acc_A = train_model(encoder_A, finetune=False)

    # ── 实验 B：预训练 Encoder + 冻结 ──
    print("\n" + "=" * 50)
    print("实验 B：预训练 Encoder + 冻结")
    print("=" * 50)
    encoder_B = ProteinBERT().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_B.load_state_dict(ckpt['model_state_dict'])
    clf_B, acc_B = train_model(encoder_B, finetune=False)

    # ── 实验 C：预训练 Encoder + Fine-tuning ──
    print("\n" + "=" * 50)
    print("实验 C：预训练 Encoder + Fine-tuning")
    print("=" * 50)
    encoder_C = ProteinBERT().to(device)
    encoder_C.load_state_dict(ckpt['model_state_dict'])
    clf_C, acc_C = train_model(encoder_C, finetune=True)

    # ── 测试集评估 ──
    def test_acc(encoder, clf, n=20):
        encoder.eval(); clf.eval()
        correct = total = 0
        for _ in range(n):
            src, labels = make_motif_batch(64, 40, device)
            with torch.no_grad():
                emb    = mean_pooling(encoder, src)
                logits = clf(emb)
                preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return correct / total

    acc_test_A = test_acc(encoder_A, clf_A)
    acc_test_B = test_acc(encoder_B, clf_B)
    acc_test_C = test_acc(encoder_C, clf_C)

    # ── 打印汇总表格 ──
    print("\n")
    print("┌─────────────────────────────────────────────────────┐")
    print("│              Motif 检测任务对比实验结果               │")
    print("├──────┬──────────────────────────┬───────────────────┤")
    print("│ 实验 │ 配置                     │ 测试集准确率       │")
    print("├──────┼──────────────────────────┼───────────────────┤")
    print(f"│  A   │ 随机 Encoder + 冻结      │ {acc_test_A:.4f}            │")
    print(f"│  B   │ 预训练 Encoder + 冻结    │ {acc_test_B:.4f}            │")
    print(f"│  C   │ 预训练 Encoder + FT      │ {acc_test_C:.4f}            │")
    print("└──────┴──────────────────────────┴───────────────────┘")

    return {
        'A': (acc_A, acc_test_A),
        'B': (acc_B, acc_test_B),
        'C': (acc_C, acc_test_C),
    }


if __name__ == "__main__":
    results = run_experiments()
```

---

## 完成标准

1. `has_hydrophobic_motif` 能正确检测连续疏水簇
2. `make_motif_batch` 生成的正样本 `has_hydrophobic_motif` 全部为 True
3. 三组实验都能跑通，打印汇总表格
4. 观察并记录三组实验的准确率差异

---

## 输出问题

**Q1**：Fine-tuning 时为什么 Encoder 用 `lr=1e-4`，分类头用 `lr=1e-3`？如果两者都用 `1e-3` 会有什么风险？

**Q2**：实验 A（随机 Encoder）和实验 B（预训练 Encoder）的准确率差距是多少？你如何解释这个差距（或者没有差距）？

**Q3**：实验 B（冻结）和实验 C（Fine-tuning）相比，哪个更好？为什么 Fine-tuning 不一定总是更好？

---

准备好后提交代码、终端输出和三个问题的回答。