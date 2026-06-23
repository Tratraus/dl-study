import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day2'))

from mlm_data import (
    VOCAB_SIZE, PAD, IGNORE_INDEX,
    make_mlm_batch
)
from protein_bert import ProteinBERT


# ── TODO 1：实现单步训练函数 ──────────────────────────────────
def train_step(
    model: ProteinBERT,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> float:
    """
    执行一步 MLM 训练。

    步骤：
      1. 生成一批 MLM 数据（make_mlm_batch）
      2. 构造 src_key_padding_mask
      3. 前向传播，得到 logits
      4. 计算 MLM loss（cross_entropy，ignore_index=IGNORE_INDEX）
      5. 反向传播 + optimizer.step() + 梯度清零

    返回：
      loss.item()（float）

    注意：
      - logits shape: (batch, seq_len, vocab_size)
      - labels shape: (batch, seq_len)
      - cross_entropy 需要 (N, C) 格式，记得 .view()
    """
    masked_src, labels = make_mlm_batch(batch_size, seq_len, device)
    src_key_padding_mask = (masked_src == PAD)
    logits = model(masked_src, src_key_padding_mask=src_key_padding_mask)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=IGNORE_INDEX)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── TODO 2：实现训练主循环 ────────────────────────────────────
def train(
    num_steps:  int   = 1000,
    batch_size: int   = 32,
    seq_len:    int   = 50,
    lr:         float = 1e-3,
    log_every:  int   = 100,
    save_path:  str   = 'week10/day3/protein_bert_mlm.pt',
):
    """
    MLM 预训练主循环。

    步骤：
      1. 初始化设备、模型、optimizer（用 Adam）
      2. 循环 num_steps 步，每步调用 train_step
      3. 每 log_every 步打印当前 step 和 loss
      4. 训练结束后保存 checkpoint

    checkpoint 格式（用 torch.save 保存 dict）：
      {
        'model_state_dict': model.state_dict(),
        'step': num_steps,
        'final_loss': last_loss,
      }

    打印格式示例：
      Step    0 | loss: 3.2154
      Step  100 | loss: 2.8732
      Step  200 | loss: 2.4501
      ...
      Step 1000 | loss: 1.3204
      ✅ 训练完成，checkpoint 已保存至 protein_bert_mlm.pt
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProteinBERT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for step in range(num_steps + 1):
        loss = train_step(model, optimizer, batch_size, seq_len, device)
        loss_history.append(loss)

        if step % log_every == 0:
            print(f"Step {step:5d} | loss: {loss:.4f}")

    # 保存 checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': num_steps,
        'final_loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✅ 训练完成，checkpoint 已保存至 {save_path}")
    return loss_history


# ── TODO 3：loss 曲线可视化 ───────────────────────────────────
def plot_loss_curve(loss_history: list[float], save_path: str = 'week10/day3/loss_curve.png'):
    """
    绘制 loss 曲线并保存。

    要求：
      - x 轴：训练步数
      - y 轴：loss
      - 标题：MLM Pre-training Loss
      - 加一条水平虚线标注 log(25) ≈ 3.22（理论初始 loss）
      - 保存为 loss_curve.png
    """
    import matplotlib.pyplot as plt
    import math

    fig, ax = plt.subplots(figsize=(8, 4))

    # TODO：
    # 1. 画 loss 曲线
    # 2. 画 y = log(25) 的水平虚线，标注 "random baseline (log 25)"
    # 3. 设置标题、x/y 轴标签
    # 4. 保存图片
    ax.plot(loss_history, label='MLM Loss')
    ax.axhline(y=math.log(25), color='r', linestyle='--', label='random baseline (log 25)')
    ax.set_title('MLM Pre-training Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()        # 自动调整间距
    fig.savefig(save_path)    # 保存为 loss_curve.png
    print(f"✅ Loss 曲线已保存至 {save_path}")


if __name__ == "__main__":
    # 先跑训练，收集 loss history
    # 再画曲线

    # 提示：可以修改 train() 让它返回 loss_history
    loss_history = train()          # 拿到 loss 列表
    plot_loss_curve(loss_history)   # 画图


# 输出
# Step     0 | loss: 3.4143
# Step   100 | loss: 2.8815
# Step   200 | loss: 2.9201
# Step   300 | loss: 2.7925
# Step   400 | loss: 2.8041
# Step   500 | loss: 2.8651
# Step   600 | loss: 2.8725
# Step   700 | loss: 2.8665
# Step   800 | loss: 2.7670
# Step   900 | loss: 2.8006
# Step  1000 | loss: 2.8625
# ✅ 训练完成，checkpoint 已保存至 week10/day3/protein_bert_mlm.pt
# ✅ Loss 曲线已保存至 week10/day3/loss_curve.png

# Q1：logits.view(-1, vocab_size) 和 labels.view(-1) 做了什么？为什么 cross_entropy 需要这个格式？
# logits.view(-1, vocab_size) 将 logits 张量展平成二维张量，形状为 (batch_size * seq_len, vocab_size)，每行对应一个 token 的预测分布。
# labels.view(-1) 将 labels 张量展平成一维张量，形状为 (batch_size * seq_len)，每个元素对应一个 token 的真实标签。
# cross_entropy 需要这个格式是因为它期望输入的 logits 是 (N, C) 形式，其中 N 是样本数量（这里是 batch_size * seq_len），C 是类别数量（这里是 vocab_size）。

# Q2：你的初始 loss 是多少？和理论值 log(25)≈3.22 相比如何？
# 3.4143，略高于理论值 log(25)≈3.22，说明模型在初始阶段的预测能力较差，接近随机猜测。

# Q3：训练 1000 步后 loss 降到了多少？loss 曲线的下降趋势是什么样的（匀速下降、先快后慢、还是其他）？
# 2.8625，loss 曲线呈现先快后慢的下降趋势，说明模型在初始阶段学习较快，随着训练的进行，学习速度逐渐减慢，趋于收敛。