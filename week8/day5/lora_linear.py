import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    用 LoRA 包装一个已有的 nn.Linear。

    原始 Linear:
        y = x W0^T + b

    LoRA:
        y = x W0^T + b + scale * x (B A)^T

    其中：
        A: (r, in_features)
        B: (out_features, r)
        BA: (out_features, in_features)
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()

        assert isinstance(base_linear, nn.Linear), "base_linear 必须是 nn.Linear"

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # TODO 1：冻结原始 Linear 参数
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # TODO 2：定义 LoRA A 和 B
        # A shape: (rank, in_features)
        # B shape: (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # TODO 3：初始化
        # A 用 Kaiming uniform 或正态小随机数
        # B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始 frozen linear 输出
        base_out = self.base_linear(x)

        # TODO 4：LoRA 分支
        # 方法一：
        #   lora_hidden = F.linear(x, self.lora_A)
        #   lora_out    = F.linear(lora_hidden, self.lora_B)
        #   return base_out + self.scale * lora_out
        lora_hidden = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return base_out + self.scale * lora_out

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merge_weights(self):
        """
        推理时可以把 LoRA 权重合并进 base_linear.weight：
            W_merged = W0 + scale * B @ A

        注意：调用后 LoRA 分支理论上不应再重复加，否则会加两次。
        今天只实现计算，不真正修改 forward 逻辑。
        """
        with torch.no_grad():
            delta_w = self.scale * (self.lora_B @ self.lora_A)
            merged_weight = self.base_linear.weight + delta_w
        return merged_weight


def verify_lora_linear():
    torch.manual_seed(42)

    in_features = 320
    out_features = 320
    rank = 8
    alpha = 16

    # 原始 Linear
    base = nn.Linear(in_features, out_features)

    # 保存一份原始输出
    x = torch.randn(4, 100, in_features)
    with torch.no_grad():
        base_out = base(x)

    # 包装成 LoRA Linear
    lora = LoRALinear(base, rank=rank, alpha=alpha)

    # 验证 1：原始权重已冻结
    assert lora.base_linear.weight.requires_grad == False
    assert lora.base_linear.bias.requires_grad == False
    print("✅ 原始 Linear 参数已冻结")

    # 验证 2：输出形状
    out = lora(x)
    assert out.shape == base_out.shape
    print(f"✅ 输出形状正确：{out.shape}")

    # 验证 3：初始化时输出应与原始 Linear 完全一致
    max_diff = (out - base_out).abs().max().item()
    print(f"✅ 初始化最大输出偏差：{max_diff:.2e}")

    # 验证 4：参数量
    trainable = lora.count_trainable_parameters()
    expected = rank * in_features + out_features * rank
    print(f"✅ LoRA 可训练参数量：{trainable}，预期：{expected}")

    # 验证 5：梯度
    loss = out.sum()
    loss.backward()

    print(f"✅ base weight grad: {lora.base_linear.weight.grad}")
    print(f"✅ LoRA A grad norm: {lora.lora_A.grad.norm():.4f}")
    print(f"✅ LoRA B grad norm: {lora.lora_B.grad.norm():.4f}")

    # 验证 6：merge 权重形状
    merged = lora.merge_weights()
    assert merged.shape == lora.base_linear.weight.shape
    print(f"✅ merged weight shape 正确：{merged.shape}")

    print("\n🎉 LoRALinear 所有验证完成！")


if __name__ == "__main__":
    verify_lora_linear()


# Q1
# 为什么 LoRA 要把 B 初始化为零，而不是把 A 和 B 都随机初始化？
## 这样做的好处是：在训练开始时，LoRA 分支对输出没有影响，模型行为与原始 Linear 完全一致。
# 这有助于稳定训练，避免一开始就引入过大的扰动。随着训练进行，A 和 B 会逐渐学习到有用的调整，从而提升模型性能。

# Q2
# 如果 rank=8, in_features=320, out_features=320，LoRA 参数量是多少？相比原始 Linear 的权重参数量减少了多少倍？
# 5120, 原始 Linear 权重参数量是 320*320=102400，减少了 20 倍。

# Q3
# merge_weights() 的作用是什么？为什么 LoRA 在推理时可以做到几乎不增加延迟？
# merge_weights() 的作用是将 LoRA 分支的权重调整合并到原始 Linear 的权重中，使得推理时不需要额外计算 LoRA 分支。
# LoRA 在推理时几乎不增加延迟的原因是：合并后只需进行一次线性变换，
# 而不需要分别计算原始 Linear 和 LoRA 分支的输出，从而避免了额外的计算开销。

# Q4
# 你观察到的梯度现象是什么？lora_A.grad 和 lora_B.grad 谁一开始为 0？为什么？
# LoRA A 的梯度一开始为 0，而 LoRA B 的梯度不为 0。这是因为在前向传播中，LoRA A 的输出被 LoRA B 线性变换后才影响最终输出，
# 因此在反向传播中，LoRA A 的梯度需要通过 LoRA B 的权重传播回来，而 LoRA B 的梯度直接来自于输出损失的梯度。
