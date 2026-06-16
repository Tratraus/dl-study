# Week 9 Day 3 Review — 完整 Seq2Seq 模型组装 + Teacher Forcing 训练

## 代码结构分析

```python
# 三层结构清晰：

Encoder(src)                    # src → memory
    ↓
Seq2Seq.forward(src, tgt_input) # 组装 Encoder + Decoder
    ↓
Decoder(tgt_input, memory)      # tgt_input + memory → logits
    ↓
CrossEntropyLoss(logits, tgt)   # 计算 loss
```

Seq2Seq 类把 Day 1 的 Encoder/Decoder 干净地组装起来，forward 只有 3 步：mask → encoder → decoder。

---

## 数据流 / Shape 变化追踪

### 数据生成

| 阶段 | 操作 | Shape |
|------|------|-------|
| src | randint(3, 20) | (64, 10) |
| tgt | [BOS] + flip(src) + [EOS] | (64, 11) |
| tgt_input | tgt[:, :-1] | (64, 10) |
| tgt_output | tgt[:, 1:] | (64, 10) |

### 前向传播

| 阶段 | Shape |
|------|-------|
| src | (64, 10) |
| tgt_input | (64, 10) |
| causal mask | (10, 10) |
| memory = encoder(src) | (64, 10, 64) |
| logits = decoder(tgt_input, memory) | (64, 10, 20) |

### Loss 计算

| 阶段 | Shape |
|------|-------|
| logits.view(-1, 20) | (640, 20) |
| tgt_output.reshape(-1) | (640,) |
| loss | scalar ✅ |

---

## 关键知识点

### 1. Teacher Forcing 的数据切分

```python
tgt_input  = tgt[:, :-1]   # [<BOS>, 4, 2, 5, 3, 1]     输入给 Decoder
tgt_output = tgt[:, 1:]    # [4, 2, 5, 3, 1, <EOS>]      作为 label
```

**关键**：错开一位。Decoder 看到 `<BOS>` 时，要预测 `4`；看到 `<BOS>, 4` 时，要预测 `2`……这就是 Teacher Forcing——训练时给 Decoder 看"正确答案"，而不是它自己的预测。

### 2. Causal Mask 的作用

```python
mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
```

上三角全是 `-inf`，经过 softmax 后变成 0。效果：位置 i 只能 attend 到位置 0~i，不能"偷看未来"。

### 3. ignore_index=PAD 的防御性写法

主人回答正确——这个合成任务没有 PAD，但养成习惯很重要。实际任务中（变长序列），PAD 出现在 label 里会被 loss 忽略，防止模型学"预测 PAD"。

---

## 踩坑易错点

### 易错 1：view vs reshape 报错

主人踩到了这个坑，问得好。

**根本原因**：`view` 要求 tensor 在内存中是**连续的（contiguous）**，`reshape` 不要求。

```python
tgt = torch.randint(0, 20, (64, 11))      # contiguous ✅
tgt_output = tgt[:, 1:]                    # 可能不 contiguous ❌
# tgt 的 stride = (11, 1)
# tgt_output 是原 tensor 的一个"窗口"，stride 还是 (11, 1)
# 但 shape 变成了 (64, 10)，内存布局不再是连续的

tgt_output.view(-1)    # ❌ 报错：view size is not compatible with input tensor's size and stride
tgt_output.reshape(-1)  # ✅ reshape 会自动处理，必要时复制一份
```

**什么时候会不连续？**
- 切片操作 `tensor[:, 1:]`、`tensor[::2]` 等
- `transpose()`、`permute()` 之后
- 某些 `expand()` 操作

**记忆口诀**：切片/transpose 后用 reshape 就对了。

**深入理解：Stride 与内存连续性**

Tensor 在内存中就是一排数字，不管几维，底层内存是一维的。Stride = "每一维走一步，要跳过多少个内存位置"：

```python
x = torch.tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9,10,11,12]])
x.shape    # (3, 4)
x.stride() # (4, 1)  ← dim0 跳 4 格，dim1 跳 1 格
```

访问 `[i, j]` 的内存地址 = `i × stride[0] + j × stride[1]`

连续（contiguous）= stride 和 shape 满足：`stride[i] = stride[i+1] × shape[i+1]`。切片后为什么不连续：

```
原始内存:  [1][2][3][4][5][6][7][8][9][10][11][12]
                ↑               ↑                ↑
              y[0,0]          y[1,0]           y[2,0]
              地址 1           地址 5           地址 9

y = x[:, 1:]  →  shape (3, 3)，stride 还是 (4, 1)
从 y[0,0] 到 y[1,0]：地址 1 → 5，隔了 4 格 → stride 必须是 4
但连续要求 stride[0] = 1 × 3 = 3 ≠ 4 → 不连续
```

**Stride 不能"更新"**——它是对底层内存真实距离的如实描述。切片不移动数据，内存间距没变，所以 stride 也没法变。要让 stride 变小（变连续），唯一办法是**把数据复制到一块新内存里紧密排列**（`contiguous()` 或 `reshape`）。

本质是**时间 vs 空间的 trade-off**：切片不复制 → 省内存但不连续；reshape/contiguous 复制 → 多用内存但连续。

### 易错 2：tgt 的长度

```python
tgt = [BOS] + src.flip() + [EOS]   # 长度 = seq_len + 2
tgt_input  = tgt[:, :-1]            # 长度 = seq_len + 1
tgt_output = tgt[:, 1:]             # 长度 = seq_len + 1
```

注意 `tgt_input` 和 `tgt_output` 长度是 `seq_len + 1`，不是 `seq_len`。但在这个代码里 `src` 长度是 `seq_len=10`，`tgt` 长度是 11，切分后都是 10——和 src 长度恰好一样，所以没有问题。如果 src 长度和 tgt 长度不同，要注意 causal mask 的 size 要匹配 `tgt_input` 的长度。

---

## 输出问题回答评价

### Q1：为什么需要 ignore_index=PAD？

**主人回答**：合成任务没有 PAD，但实际应用中会有，习惯上加上。

**评价**：✅ 正确。补充一点技术细节：如果不过 `ignore_index`，模型会被迫学习"预测 PAD token"，这会：
1. 污染梯度（PAD 位置的梯度会干扰正常 token 的学习）
2. 虚假地降低 loss（PAD 预测好了 loss 看起来更低，但实际能力没提升）

### Q2：为什么要 view？

**主人回答**：CrossEntropyLoss 期望 (N, C) 和 (N,)，不 view 会 shape 不匹配。

**评价**：✅ 完全正确。PyTorch 的 CrossEntropyLoss 有两种输入格式：
- 2D: `(N, C)` + `(N,)` — 标准格式
- 3D: `(N, C, d1, d2, ...)` + `(N, d1, d2, ...)` — 多维格式

其实 PyTorch 也支持 3D 输入（`logits` shape `(batch, vocab, seq)` + `tgt` shape `(batch, seq)`），但需要把 vocab 维度放在第 1 位（不是第 2 位）。所以最安全的做法就是 view 成 2D。

### Q3：loss 降到了多少？

**主人回答**：Step 500 loss = 0.0027

**评价**：✅ 非常好。从初始 ~3.0（log(20)）降到 0.0027，下降了 1000 倍以上，远超"0.5 以下"的完成标准。模型完美学会了序列翻转任务。

---

## 训练曲线分析

```
Step 100, Loss: 0.0597   ← 快速下降阶段
Step 200, Loss: 0.0136   ← 继续下降
Step 300, Loss: 0.0074   ← 趋于平缓
Step 400, Loss: 0.0040   ← 接近收敛
Step 500, Loss: 0.0027   ← 收敛
```

下降曲线典型：前 100 步快速下降（模型学到"翻转"的基本模式），后面逐步精炼。没有震荡、没有发散，训练循环写得很稳。

---

## 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| TODO 完成度 | ✅ 3/3 | 全部完成 |
| Seq2Seq 组装 | ✅ | Encoder + Decoder + causal mask 正确 |
| 训练循环 | ✅ | 数据切分、loss 计算、梯度更新都对 |
| view/reshape 问题 | ✅ | 改用 reshape 解决，理解了原因 |
| Q1-Q3 回答 | ✅ | 准确，Q2 理解到位 |
| 训练结果 | ✅ | loss 0.0027，远超标准 |

**一句话总结**：Day 3 把 Day 1/2 的零件组装成了完整的 Seq2Seq，训练循环跑通，loss 从 3.0 降到 0.0027。view → reshape 的坑是 PyTorch 内存布局的经典问题，记住"切片/transpose 后用 reshape 就对了"。
