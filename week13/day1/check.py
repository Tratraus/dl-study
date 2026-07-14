import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week11', 'day2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week12', 'day5'))
# 1. 数据 pipeline（W11）
from protein_dataset import ProteinDataset, make_collate_fn

# 2. ESM-2 加载工具（W11）
from esm2_embed import load_esm2   # 或你实际的函数名

# 3. 自实现分类器（W12）
from train_classifier import ProteinClassifier