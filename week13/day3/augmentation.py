import random

# ══════════════════════════════════════════════════════
# 氨基酸保守替换分组
# ══════════════════════════════════════════════════════
CONSERVATIVE_GROUPS = [
    set('AVLIM'),   # 脂肪族疏水
    set('FWY'),     # 芳香族
    set('STNQ'),    # 极性不带电
    set('KRH'),     # 带正电
    set('DE'),      # 带负电
    set('CGP'),     # 特殊结构
]

# 构建 氨基酸 -> 同组其他氨基酸列表 的映射，加速查找
_AA_TO_GROUP_MATES = {}
for group in CONSERVATIVE_GROUPS:
    for aa in group:
        _AA_TO_GROUP_MATES[aa] = list(group - {aa})


def conservative_substitution(seq: str, prob: float = 0.1) -> str:
    """
    保守氨基酸替换：以 prob 概率将每个氨基酸替换为同组的其他氨基酸。

    Args:
        seq:  原始蛋白质序列（大写字母字符串）
        prob: 每个位置被替换的概率

    Returns:
        增强后的序列（长度不变）
    """
    seq_list = list(seq)
    for i, aa in enumerate(seq_list):
        if aa in _AA_TO_GROUP_MATES and random.random() < prob:
            mates = _AA_TO_GROUP_MATES[aa]
            if mates:   # 该组内确实有其他成员可替换
                seq_list[i] = random.choice(mates)
    return ''.join(seq_list)

# task1 ver
# def random_crop(seq: str, min_len: int = 30) -> str:
#     """
#     随机截取子序列。若序列本身长度 <= min_len，则不裁剪，原样返回。

#     Args:
#         seq:     原始蛋白质序列
#         min_len: 裁剪后的最小长度

#     Returns:
#         裁剪后的子序列
#     """
#     seq_len = len(seq)
#     if seq_len <= min_len:
#         return seq

#     # 随机决定本次裁剪后的长度：[min_len, seq_len] 之间
#     crop_len = random.randint(min_len, seq_len)
#     # 随机决定起始位置
#     max_start = seq_len - crop_len
#     start = random.randint(0, max_start)
#     return seq[start:start + crop_len]
def random_crop(seq: str, min_len_ratio: float = 0.7) -> str:
    """
    随机截取子序列（按比例版）。
    保留长度在 [seq_len * min_len_ratio, seq_len] 之间随机取值，
    避免固定长度裁剪对短序列/长序列造成不同程度的信息丢失。

    Args:
        seq:           原始蛋白质序列
        min_len_ratio: 裁剪后保留的最小比例（0~1）

    Returns:
        裁剪后的子序列
    """
    seq_len = len(seq)
    min_len = max(1, int(seq_len * min_len_ratio))

    if min_len >= seq_len:
        return seq

    crop_len  = random.randint(min_len, seq_len)
    max_start = seq_len - crop_len
    start     = random.randint(0, max_start)
    return seq[start:start + crop_len]

# ══════════════════════════════════════════════════════
# 数据集包装器：在原有 ProteinDataset 基础上加增强
# ══════════════════════════════════════════════════════
from torch.utils.data import Dataset
# task1 ver
# class AugmentedProteinDataset(Dataset):
#     """
#     包装原始 ProteinDataset，训练时开启增强，验证/测试时关闭。

#     用法：
#         train_ds = AugmentedProteinDataset(base_train_dataset, augment=True)
#         val_ds   = AugmentedProteinDataset(base_val_dataset,   augment=False)
#     """
#     def __init__(self, base_dataset, augment: bool = False,
#                 sub_prob: float = 0.1, crop_min_len: int = 30):
#         self.base_dataset = base_dataset
#         self.augment      = augment
#         self.sub_prob     = sub_prob
#         self.crop_min_len = crop_min_len

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         seq, label = self.base_dataset[idx]

#         if self.augment:
#             seq = conservative_substitution(seq, prob=self.sub_prob)
#             seq = random_crop(seq, min_len=self.crop_min_len

#         return seq, label

class AugmentedProteinDataset(Dataset):
    def __init__(self, base_dataset, augment: bool = False,
                use_substitution: bool = True, use_crop: bool = True,
                sub_prob: float = 0.1, crop_min_len_ratio: float = 0.7):
        self.base_dataset       = base_dataset
        self.augment            = augment
        self.use_substitution   = use_substitution
        self.use_crop           = use_crop
        self.sub_prob           = sub_prob
        self.crop_min_len_ratio = crop_min_len_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        seq, label = self.base_dataset[idx]

        if self.augment:
            if self.use_substitution:
                seq = conservative_substitution(seq, prob=self.sub_prob)
            if self.use_crop:
                seq = random_crop(seq, min_len_ratio=self.crop_min_len_ratio)

        return seq, label

if __name__ == '__main__':
    # ── 快速自测 ──────────────────────────────────────
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

    print("原始序列长度：", len(test_seq))
    print("原始序列前50位：", test_seq[:50])

    sub_seq = conservative_substitution(test_seq, prob=0.1)
    print("\n保守替换后前50位：", sub_seq[:50])
    diff_count = sum(1 for a, b in zip(test_seq, sub_seq) if a != b)
    print(f"替换位点数：{diff_count} / {len(test_seq)}"
          f"（约 {100*diff_count/len(test_seq):.1f}%）")

    crop_seq = random_crop(test_seq, min_len_ratio=0.7)
    print(f"\n裁剪后长度：{len(crop_seq)}（原长度 {len(test_seq)}）")
    print("裁剪后序列：", crop_seq[:50], "...")
