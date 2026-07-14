import torch
import random
import numpy as np

# ⚠️ 关键：先检查环境种子状态，再加载
print("=== 环境种子状态检查 ===")
print(f"torch initial seed: {torch.initial_seed()}")

# 加载 checkpoint
checkpoint = torch.load('/home/tratr/dl-study/week13/day1/baseline_checkpoint.pt', map_location='cuda')

# 打印 checkpoint 里保存了什么，尤其关注是否存了训练时的随机状态
print("\n=== Checkpoint Keys ===")
for k in checkpoint.keys():
    print(k)

# 如果 checkpoint 是 dict 且只存了 model state_dict，检查是否还额外存了 epoch/optimizer/rng_state
