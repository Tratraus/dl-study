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
    weights_pos = [5 if i in KR_IDS else 1 for i in AA_IDS]
    weights_neg = [0 if i in KR_IDS else 1 for i in AA_IDS]

    half = batch_size // 2
    pos_seqs = [random.choices(AA_IDS, weights=weights_pos, k=seq_len) for _ in range(half)]
    neg_seqs = [random.choices(AA_IDS, weights=weights_neg, k=seq_len) for _ in range(half)]

    labels_pos = [1] * half
    labels_neg = [0] * half

    all_seqs = pos_seqs + neg_seqs
    all_labels = labels_pos + labels_neg

    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)

    src = torch.tensor(all_seqs, dtype=torch.long).to(device)
    labels = torch.tensor(all_labels, dtype=torch.long).to(device)
    return src, labels


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
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    for param in encoder.parameters():
        param.requires_grad = False

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
        logits = classifier(embeddings)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

        loss_history.append(loss.item())
        acc_history.append(acc)

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

    correct = 0
    total = 0

    for _ in range(num_batches):
        src, labels = make_classification_batch(batch_size, seq_len, device)

        with torch.no_grad():
            embeddings = mean_pooling(encoder, src)
            logits = classifier(embeddings)
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


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

# 输出

# ✅ Encoder 加载成功，final_loss = 2.8625

# Step    0 | loss: 0.7359 | acc: 0.5000
# Step   50 | loss: 0.1371 | acc: 0.9688
# Step  100 | loss: 0.0272 | acc: 1.0000
# Step  150 | loss: 0.0113 | acc: 1.0000
# Step  200 | loss: 0.0043 | acc: 1.0000
# Step  250 | loss: 0.0043 | acc: 1.0000
# Step  300 | loss: 0.0034 | acc: 1.0000
# Step  350 | loss: 0.0047 | acc: 1.0000
# Step  400 | loss: 0.0031 | acc: 1.0000
# Step  450 | loss: 0.0021 | acc: 1.0000
# Step  500 | loss: 0.0005 | acc: 1.0000

# ✅ 测试集准确率：0.9992


# Q1：冻结 Encoder 后，optimizer = torch.optim.Adam(classifier.parameters(), lr=lr) 只优化分类头。
# 如果误写成 optimizer = torch.optim.Adam(model.parameters(), lr=lr)，会发生什么？
# 会导致整个模型（包括 encoder）都参与优化，虽然 encoder 的参数 requires_grad=False，
# 但 optimizer 仍然会尝试更新它们。这可能会浪费计算资源，并且可能会引入不必要的噪声，影响分类头的训练效果。

# Q2：为什么提取嵌入时要用 torch.no_grad()？不加会有什么后果？
# 使用 torch.no_grad() 可以告诉 PyTorch 不需要计算梯度，从而节省显存和计算资源。
# 如果不加，PyTorch 会为 encoder 的输出计算梯度，虽然 encoder 的参数被冻结，但这会占用额外的显存，并且可能会导致训练速度变慢。

# Q3：你的最终测试集准确率是多少？训练集 accuracy 和测试集 accuracy 差距大吗？说明了什么？
# 最终测试集准确率约为 0.9992，训练集 accuracy 在训练过程中逐渐提升，最终达到接近 1.0。
# 训练集和测试集的 accuracy 非常接近，说明模型没有过拟合，能够很好地泛化到未见过的数据。这可能是因为数据生成规则简单，模型容量足够，或者训练步骤足够多。