import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        # TODO 1：定义降维线性层
        # 输入维度 d_model，输出维度 bottleneck
        self.down_proj = nn.Linear(d_model, bottleneck)


        # TODO 2：定义激活函数（使用 GELU）
        self.act = nn.GELU()

        # TODO 3：定义升维线性层
        # 输入维度 bottleneck，输出维度 d_model
        self.up_proj = nn.Linear(bottleneck, d_model)

        # TODO 4：近恒等初始化
        # 将 up_proj 的权重初始化为全零
        # 提示：nn.init.zeros_(tensor)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5：实现前向传播
        # 注意：需要残差连接
        down = self.down_proj(x)  # 降维
        act  = self.act(down)     # 激活
        up   = self.up_proj(act)  # 升维
        return x + up              # 残差连接

    def count_parameters(self) -> int:
        # TODO 6：返回该模块的可训练参数数量
        # 写法A：
        # total = 0
        # for p in self.parameters():
        #     if p.requires_grad:
        #         total += p.numel()
        # return total
        # 写法B（更简洁）：
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# ════════════════════════════════════════════════════════════
# 验证代码（不需要修改，完成上面的 TODO 后直接运行）
# ════════════════════════════════════════════════════════════

def verify_adapter():
    d_model    = 320
    bottleneck = 64
    batch_size = 4
    seq_len    = 100

    adapter = AdapterLayer(d_model, bottleneck)

    x = torch.randn(batch_size, seq_len, d_model)

    # 验证 1：输出形状是否正确
    out = adapter(x)
    assert out.shape == x.shape, f"形状错误：{out.shape} != {x.shape}"
    print(f"✅ 输出形状正确：{out.shape}")

    # 验证 2：近恒等初始化——初始输出应与输入几乎相同
    max_diff = (out - x).abs().max().item()
    print(f"✅ 初始最大偏差：{max_diff:.2e}（应接近 0）")

    # 验证 3：参数量
    params = adapter.count_parameters()
    expected = 2 * d_model * bottleneck + d_model + bottleneck
    print(f"✅ 可训练参数量：{params}（预期约 {expected}）")

    # 验证 4：梯度能否正常流动
    loss = out.sum()
    loss.backward()
    grad_down = adapter.down_proj.weight.grad
    grad_up   = adapter.up_proj.weight.grad
    assert grad_down is not None, "down_proj 没有梯度！"
    assert grad_up   is not None, "up_proj 没有梯度！"
    print(f"✅ 梯度正常：down_proj grad norm = {grad_down.norm():.4f}")
    print(f"✅ 梯度正常：up_proj   grad norm = {grad_up.norm():.4f}")

    print("\n🎉 所有验证通过！")

if __name__ == "__main__":
    verify_adapter()


# Q1：验证 2 检查"初始最大偏差接近 0"，你的输出是多少？为什么不是严格的 0？（提示：down_proj 的初始化是什么？）
# 0.00e+00，因为我们初始化时把bias也初始化为0了，所以up_proj的输出在初始状态下是0，
# 如果说没有初始化bias，那么up_proj的输出就不一定是0了，
# 因为默认的Kaiming初始化会给bias一个非零的初始值，这样输出就会有偏差了。

# Q2：如果把 TODO 4 的初始化去掉（让 up_proj 保持默认的 Kaiming 初始化），验证 2 的结果会变成什么？这对训练有什么影响？
# 验证 2 的结果会变成一个较大的偏差，因为 up_proj 的初始输出不再是零，这会导致初始输出与输入的差异较大。


# Q3：count_parameters 里，bias 也是参数，你有没有把它算进去？bias 的维度是多少？
# 包含进去了
# down_proj的bias维度是bottleneck，up_proj的bias维度是d_model，
