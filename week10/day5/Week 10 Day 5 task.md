# Week 10 Day 5：下游分类任务

## 最小必要理论（10 分钟）

### 1. 迁移学习的两种范式

今天的核心问题是：**预训练好的 ProteinBERT，怎么用于分类？**

有两种做法：

```text
范式 A：冻结 Encoder（Frozen）
  ProteinBERT.encode()  →  固定不变，不更新梯度
        ↓
  MLP 分类头  →  只训练这部分

范式 B：微调（Fine-tuning）
  ProteinBERT.encode()  →  参与梯度更新
        ↓
  MLP 分类头  →  也训练
```

今天先实现**范式 A（冻结 Encoder）**，Day 6 对比两种范式的差异。

---

### 2. 为什么先做冻结 Encoder？

| | 冻结 Encoder | Fine-tuning |
|--|--|--|
| 训练速度 | 快（只更新分类头） | 慢（更新全部参数） |
| 数据需求 | 少（分类头参数少） | 多（防止过拟合） |
| 适用场景 | 标注数据少 | 标注数据充足 |
| 调试难度 | 低（问题容易定位） | 高（Encoder 和 Head 互相影响） |

冻结 Encoder 是一个很好的**基线**：如果冻结后分类效果已经很好，说明预训练嵌入质量高；如果效果差，再考虑 Fine-tuning。

---

### 3. 构造有可学习信号的分类任务

Day 3 的教训：**合成随机数据没有上下文依赖，模型学不到有用信息**。

今天的分类任务需要有**明确的可学习信号**。

设计如下：

```text
规则：序列中 "富含 K/R（带正电荷氨基酸）" → 标签 1（正电荷蛋白）
      否则                                 → 标签 0（普通蛋白）

具体：统计序列中 K 和 R 的比例
      比例 > 0.2 → 标签 1
      比例 ≤ 0.2 → 标签 0
```

这个任务是**可学习的**，因为：
- 序列中 K/R 的频率和标签直接相关
- 模型只需要学会"关注 K/R 出现的频率"就能分类

---

### 4. 分类头的结构

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
```

输入：`(batch, d_model)` = `(batch, 128)`
输出：`(batch, num_classes)` = `(batch, 2)`

---

### 5. 冻结参数的方法

```python
# 冻结 ProteinBERT 的所有参数
for param in encoder.parameters():
    param.requires_grad = False

# 只让分类头的参数参与优化
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
```

冻结后，`encoder` 的参数不会出现在梯度图里，反向传播到 Encoder 就停止。

---

## 代码任务

新建 `week10/day5/classify.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day4'))

from mlm_data import PAD, AA_IDS, token2id
from protein_bert import ProteinBERT
from extract_embedding import mean_pooling


# ── 常量 ──────────────────────────────────────────────────────
KR_IDS  = {token2id['K'], token2id['R']}   # 带正电荷氨基酸的 id
KR_THRESHOLD = 0.2                          # 比例阈值


# ── TODO 1：数据生成 ──────────────────────────────────────────
def make_classification_batch(
    batch_size: int,
    seq_len:    int,
    device:     torch.device,
):
    """
    生成带标签的分类数据。

    规则：
      - 标签 1（正电荷）：序列中 K/R 比例 > KR_THRESHOLD
        生成方式：从 AA_IDS 中随机采样，但 K/R 的权重设为 5，其余为 1
      - 标签 0（普通）：序列中 K/R 比例 ≤ KR_THRESHOLD
        生成方式：从 AA_IDS 中随机采样，但 K/R 的权重设为 0，其余为 1

    步骤：
      1. 各生成 batch_size // 2 条正样本和负样本
      2. 拼接后打乱顺序
      3. 返回 (src, labels)
         src:    (batch_size, seq_len)  dtype=torch.long
         labels: (batch_size,)          dtype=torch.long

    提示：
      weights_pos = [5 if i in KR_IDS else 1 for i in AA_IDS]
      weights_neg = [0 if i in KR_IDS else 1 for i in AA_IDS]
    """
    ...


# ── TODO 2：分类头 ────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    两层 MLP 分类头。

    结构：
      Linear(d_model, 64) → ReLU → Dropout(0.1) → Linear(64, num_classes)

    输入：(batch, d_model)
    输出：(batch, num_classes)
    """
    def __init__(self, d_model: int = 128, num_classes: int = 2):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


# ── TODO 3：训练循环 ──────────────────────────────────────────
def train_classifier(
    encoder:    ProteinBERT,
    num_steps:  int   = 500,
    batch_size: int   = 64,
    seq_len:    int   = 30,
    lr:         float = 1e-3,
    log_every:  int   = 50,
) -> tuple[ClassificationHead, list]:
    """
    冻结 Encoder，只训练分类头。

    步骤：
      1. 冻结 encoder 的所有参数（requires_grad = False）
      2. 初始化 ClassificationHead
      3. optimizer 只优化 classifier.parameters()
      4. 训练循环：
         a. make_classification_batch 生成数据
         b. encoder 提取嵌入（mean_pooling，用 torch.no_grad()）
         c. 分类头前向传播
         d. cross_entropy loss
         e. 反向传播 + 更新
      5. 每 log_every 步打印 loss 和 accuracy
      6. 返回 (classifier, loss_history)

    注意：
      - encoder 提取嵌入时用 torch.no_grad()，节省显存和计算
      - accuracy = 预测正确的样本数 / 总样本数
    """
    device = next(encoder.parameters()).device
    encoder.eval()

    # 冻结 encoder
    ...

    classifier = ClassificationHead().to(device)
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=lr)

    loss_history = []
    acc_history  = []

    for step in range(num_steps + 1):
        src, labels = make_classification_batch(batch_size, seq_len, device)

        # 提取嵌入（冻结，不需要梯度）
        with torch.no_grad():
            embeddings = mean_pooling(encoder, src)   # (batch, d_model)

        # 分类头前向传播
        ...

        if step % log_every == 0:
            print(f"Step {step:4d} | loss: {loss:.4f} | acc: {acc:.4f}")

    return classifier, loss_history


# ── TODO 4：测试集评估 ────────────────────────────────────────
def evaluate(
    encoder:    ProteinBERT,
    classifier: ClassificationHead,
    num_batches: int = 20,
    batch_size:  int = 64,
    seq_len:     int = 30,
) -> float:
    """
    在测试集上评估分类准确率。

    步骤：
      1. encoder 和 classifier 都切换到 eval 模式
      2. 生成 num_batches 批测试数据
      3. 统计总准确率
      4. 返回 accuracy（float）
    """
    encoder.eval()
    classifier.eval()
    ...


# ── 主程序 ────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')

    # 加载预训练 Encoder
    model = ProteinBERT().to(device)
    ckpt  = torch.load('week10/day3/protein_bert_mlm.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Encoder 加载成功，final_loss = {ckpt['final_loss']:.4f}\n")

    # 训练分类头
    classifier, loss_history = train_classifier(model)

    # 测试集评估
    test_acc = evaluate(model, classifier)
    print(f"\n✅ 测试集准确率：{test_acc:.4f}")
```

---

## 完成标准

1. `make_classification_batch` 能生成有效的正负样本，标签分布接近 50/50
2. `ClassificationHead` 结构正确，输出 shape `(batch, 2)`
3. 训练 500 步后 accuracy 达到 **0.85 以上**
4. 测试集准确率同样达到 **0.85 以上**（说明没有过拟合）

---

## 输出问题

**Q1**：冻结 Encoder 后，`optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)` 只优化分类头。如果误写成 `optimizer = torch.optim.Adam(model.parameters(), lr=lr)`，会发生什么？

**Q2**：为什么提取嵌入时要用 `torch.no_grad()`？不加会有什么后果？

**Q3**：你的最终测试集准确率是多少？训练集 accuracy 和测试集 accuracy 差距大吗？说明了什么？

---

准备好后提交代码、终端输出和三个问题的回答。