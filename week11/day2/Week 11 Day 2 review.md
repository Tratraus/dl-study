# Week 11 Day 2 Review：真实蛋白质功能分类数据集构造

## 数据集变更记录

原计划使用 `mila-intel/ProtST-EC`（6 类酶功能分类），但该数据集在 HuggingFace 上不存在/已下架。

**替代方案**：改用 `mila-intel/ProtST-SubcellularLocalization`（10 类亚细胞定位），数据量更大（8388 条 vs ~1000 条），类别更多（10 vs 6），任务更有挑战性。

附带变化：
- 字段名：`protein_sequence`/`label` → `prot_seq`/`localization`
- 序列格式：原始数据氨基酸之间有空格，需要 `.replace(' ', '')`
- 移除了 `make_mock_data` 备用方案（真实数据集已可用）

---

## 代码结构分析

### 6 个函数/类的职责

| 函数/类 | 输入 | 输出 | 职责 |
|---------|------|------|------|
| `load_localization_data()` | — | sequences, labels | 加载 HuggingFace 数据集，过滤短序列 |
| `ProteinDataset` | sequences, labels | Dataset 对象 | 封装序列+标签，`__getitem__` 返回字符串 |
| `split_dataset()` | sequences, labels, ratios, seed | 3 个 Dataset | 打乱 + 按比例切分 train/val/test |
| `make_collate_fn()` | tokenizer | collate_fn 函数 | 闭包捕获 tokenizer，批量 tokenize |
| `mean_pooling_with_mask()` | hidden, mask | (B, 320) | 掩码 Mean Pooling，修复 Day 1 bug |
| `main()` | — | — | 组装所有步骤，验证输出 |

### 调用链

```
main()
  ├─ load_localization_data()  → 8388 条序列 + 标签
  ├─ split_dataset()           → train(5871) / val(1258) / test(1259)
  ├─ load_esm2()               → tokenizer, model
  ├─ make_collate_fn(tokenizer) → collate_fn（闭包）
  ├─ DataLoader(collate_fn)    → (input_ids, attention_mask, labels)
  └─ mean_pooling_with_mask()  → (8, 320)
```

## 数据流 / Shape 变化追踪

```
原始数据：ds["prot_seq"]（带空格的字符串）
  │
  ▼ .replace(' ', '')[:512] + 过滤 len >= 50
sequences: list[str]，8388 条
  │
  ▼ split_dataset → ProteinDataset
train_ds(5871) / val_ds(1258) / test_ds(1259)
  │
  ▼ DataLoader(batch_size=8, collate_fn=make_collate_fn(tokenizer))
batch: [(seq1, label1), ..., (seq8, label8)]
  │
  ▼ collate_fn 内部
tokenizer(8条序列, padding=True, truncation=True, max_length=512)
  │
  ├─ input_ids:      (8, 512)
  ├─ attention_mask:  (8, 512)
  └─ labels:          (8,)
  │
  ▼ model(**inputs)
outputs.last_hidden_state: (8, 512, 320)
  │
  ▼ mean_pooling_with_mask()
hidden[:, 1:-1, :] → (8, 510, 320)
mask[:, 1:-1]      → (8, 510)
mask.unsqueeze(-1)  → (8, 510, 1)
(hidden * mask).sum(dim=1) / mask.sum(dim=1) → (8, 320)
```

## 关键知识点

### 1. 闭包模式（Closure）

```python
def make_collate_fn(tokenizer):       # 外层函数接收 tokenizer
    def collate_fn(batch):            # 内层函数"记住"了 tokenizer
        ...tokenizer(sequences)...    # 使用外层的变量
    return collate_fn                 # 返回内层函数
```

**为什么用闭包？**
- DataLoader 的 `collate_fn` 参数签名必须是 `fn(batch)`，不能传额外参数
- 闭包把 tokenizer "打包"进函数内部，避免使用全局变量
- 每次调用 `make_collate_fn(tokenizer)` 都会创建一个新的闭包，互不干扰

### 2. 为什么 tokenization 放在 collate_fn 里

| 位置 | 效果 |
|------|------|
| `__getitem__` | 每条序列单独 tokenize，长度各异，无法 batch |
| `collate_fn` | 整个 batch 一起 tokenize，自动 padding 到 batch 内最长 |

放在 collate_fn 的好处：
- padding 量更少（只 pad 到 batch 内最长，不是全局最长）
- tokenizer 原生支持批量处理，效率更高

### 3. 掩码 Mean Pooling 的数学

```
Day 1（错误）：
  embedding = mean(hidden[:, 1:-1, :])
  → PAD token 也被算进均值，稀释了有效信息

Day 2（正确）：
  embedding = Σ(hidden[i] * mask[i]) / Σ(mask[i])
  → PAD 位置乘以 0，不参与求和
```

### 4. Day1 vs Day2 差异分析

**最大绝对差异：0.605475** — 这个数字很大！

原因：batch 内序列长度差异大（219 ~ 512，差值 293）。
- 短序列（219）有大量 PAD 位置（512 - 219 = 293 个 PAD）
- Day 1 的简单 mean 把这 293 个 PAD 的隐藏状态也算进去了
- PAD token 的隐藏状态通常接近 0 或有固定偏移，拉低了整体均值

**差异什么时候会更大？**
- batch 内序列长度差异越大 → 差异越大
- 短序列占比越高 → 差异越大
- 如果所有序列等长 → 差异为 0（没有 PAD）

### 5. 数据集类别分布（严重不平衡）

| 类别 | 数量 | 占比 |
|------|------|------|
| Nucleus | 2424 | 28.9% |
| Cytoplasm | 1635 | 19.5% |
| Extracellular | 1155 | 13.8% |
| Mitochondria | 906 | 10.8% |
| Cell membrane | 800 | 9.5% |
| Endoplasmic reticulum | 516 | 6.2% |
| Plastid | 453 | 5.4% |
| Golgi apparatus | 214 | 2.6% |
| Lysosome/Vacuole | 192 | 2.3% |
| Peroxisome | 93 | 1.1% |

最大类（Nucleus 2424）是最小类（Peroxisome 93）的 **26 倍**。Day 3/4 训练时需要注意类别不平衡问题。

## 踩坑 / 易错点

1. **`prot_seq` 带空格**：原始数据中氨基酸之间有空格（"M K T A Y..."），必须 `.replace(' ', '')` 才能给 tokenizer 用。

2. **`zip(*batch)` 返回 tuple**：`sequences, labels = zip(*batch)` 后 sequences 是 tuple 不是 list，传给 tokenizer 前要 `list(sequences)`。

3. **`valid_count + 1e-8` 防除零**：虽然正常序列不会出现 valid_count=0，但防御性编程是好习惯。

4. **闭包 vs 全局变量**：有人会想把 tokenizer 设为全局变量再直接在 collate_fn 里用。能跑但不推荐——闭包更清晰，不污染命名空间。

## 输出问题回顾

**Q1**: collate_fn 签名必须是 `fn(batch)`，不能有额外参数。闭包把 tokenizer 打包进函数内部。

**Q2**: 固定 seed 保证实验可复现。如果每次划分不同，模型性能波动就无法判断是方法差异还是数据差异。

**Q3**: 最大差异 0.605，当 batch 内序列长度差异大时差异更明显（短序列 PAD 多，被 Day1 的简单 mean 算进去了）。
