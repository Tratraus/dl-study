def set_seed_fully(seed=42):
    random.seed(seed)                          # Python 内置 random
    np.random.seed(seed)                       # numpy（很多数据增强/采样用这个）
    torch.manual_seed(seed)                    # torch CPU
    torch.cuda.manual_seed(seed)               # torch 单 GPU
    torch.cuda.manual_seed_all(seed)           # torch 多 GPU
    torch.backends.cudnn.deterministic = True  # ⚠️ 常被遗漏
    torch.backends.cudnn.benchmark = False     # ⚠️ 常被遗漏，benchmark=True 会引入不确定性

set_seed_fully(42)
