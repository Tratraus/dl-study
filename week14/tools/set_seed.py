"""
week14/tools/set_seed.py
------------------------------------------------------------
统一种子控制。全项目所有脚本必须在最开始调用 set_seed()，
禁止在各自脚本里散落调用 random.seed()/np.random.seed()/torch.manual_seed()。

设计原则：单一入口(single source of truth)，避免Week13出现的
"某个脚本忘了设置种子导致结果不可复现"问题。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # 强制确定性算子，遇到不支持确定性实现的op直接报错（而不是静默使用非确定性实现）
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[set_seed] 全局种子已固定为 {seed}，确定性算子已强制开启")


def new_generator(seed: int = 42) -> torch.Generator:
    """
    为DataLoader单独创建generator，避免多个DataLoader共享全局generator
    导致的shuffle顺序互相干扰问题。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
