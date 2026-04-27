import sys
import torch

print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.randn(3, 4)
print("Tensor shape:", x.shape)
print(x)
