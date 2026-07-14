import torch
ckpt = torch.load("week13/day1/baseline_checkpoint.pt")  # 换成你实际路径
print("version   :", ckpt.get("version"))
print("test_acc  :", ckpt.get("test_acc"))
print("macro_f1  :", ckpt.get("macro_f1"))
print("seed      :", ckpt.get("seed"))
