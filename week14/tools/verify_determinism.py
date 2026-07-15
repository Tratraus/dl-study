"""
week14/tools/verify_determinism.py
------------------------------------------------------------
每日开工前的强制第一步：验证确定性算子是否真正生效。
做法：用同一个探针函数（含卷积/matmul等常见不确定性算子来源）
跑两次，比较输出哈希是否完全一致。
"""

import hashlib

import torch

from tools.set_seed import set_seed


def _probe_fn(seed: int) -> torch.Tensor:
    set_seed(seed)
    x = torch.randn(64, 128)
    w = torch.randn(128, 64)
    y = torch.matmul(x, w)
    y = torch.relu(y)
    conv = torch.nn.Conv1d(1, 4, kernel_size=3)
    z = conv(y.unsqueeze(1))
    return z


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()


def verify_determinism(seed: int = 42) -> bool:
    out1 = _probe_fn(seed)
    out2 = _probe_fn(seed)
    h1, h2 = _tensor_hash(out1), _tensor_hash(out2)

    ok = h1 == h2
    if ok:
        print(f"[verify_determinism] ✅ 通过！两次运行哈希一致: {h1[:16]}...")
    else:
        print(f"[verify_determinism] ❌ 失败！哈希不一致:\n  第一次: {h1}\n  第二次: {h2}")
    return ok


if __name__ == "__main__":
    passed = verify_determinism()
    exit(0 if passed else 1)
