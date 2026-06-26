# Week 11 Day 4 Review：测试集评估 + 混淆矩阵

## 代码结构分析

| 函数 | 职责 |
|------|------|
| `get_predictions()` | 遍历 test_loader，收集所有预测/真实标签（np.ndarray） |
| `per_class_report()` | 手动计算每类 TP/FP/FN → Precision/Recall/F1，打印表格 |
| `plot_confusion_matrix()` | 构建混淆矩阵 → 行归一化 → imshow 热图 → 保存 PNG |
| `main()` | 加载数据（seed=42 保证一致）→ 加载 best_classifier.pt → 评估 |

整体结构清晰，三步流水线：收集预测 → 分类报告 → 可视化。

---

## 数据流 / 形状变化

```
test_loader batch:
  input_ids:      (B, max_len)
  attention_mask:  (B, max_len)
  labels:          (B,)
       ↓
esm_model(input_ids, attention_mask)
       ↓
outputs.last_hidden_state: (B, max_len, 320)
       ↓
mean_pooling_with_mask: (B, 320)
       ↓
classifier: (B, 10)  ← logits
       ↓
argmax(dim=1): (B,)  ← predicted labels
       ↓
np.concatenate → (N,)  ← all_preds, all_labels
```

---

## 关键数据对比

### Day 3 vs Day 4 一致性验证

| 指标 | Day 3 (Val) | Day 4 (Test) | 差距 |
|------|-------------|-------------|------|
| Accuracy | 0.634 | 0.664 | +3.0% |

测试集准确率略高于验证集，合理范围内（±5% ✓）。说明模型没有过拟合到验证集。

### 每类别 F1 排行

| 排名 | 类别 | F1 | 支持数 | 特征 |
|------|------|-----|--------|------|
| 🥇 | Extracellular | 0.894 | 160 | 分泌蛋白，序列信号强 |
| 🥈 | Nucleus | 0.753 | 369 | 最多样本，核定位信号明确 |
| 🥉 | Plastid | 0.707 | 76 | 转运肽特征明显 |
| 4 | Cell membrane | 0.683 | 115 | 跨膜螺旋有规律 |
| 5 | Mitochondria | 0.629 | 140 | 线粒体靶向序列 |
| 6 | Cytoplasm | 0.520 | 261 | 最多样本之一但"默认类别"，信号弱 |
| 7 | Endoplasmic reticulum | 0.481 | 78 | ER 信号肽 |
| 8 | Golgi apparatus | 0.338 | 21 | 样本极少 |
| 9 | Lysosome/Vacuole | 0.291 | 26 | 样本极少 |
| 🔴 | Peroxisome | 0.214 | 13 | 最少样本，PTS 信号弱 |

**Macro F1 = 0.551** vs **Overall Accuracy = 0.664**，差距 0.113，说明多数类撑高了准确率，少数类拉低了 F1。

---

## 混淆矩阵分析

从混淆矩阵中可以观察到几个关键混淆模式：

1. **Golgi → Cytoplasm (高比例)**：高尔基体蛋白在细胞质中合成和加工，序列特征相似，模型难以区分
2. **Lysosome → Cytoplasm**：溶酶体蛋白同样在细胞质中加工，共享信号肽特征
3. **Peroxisome → 多类分散**：样本太少（测试集仅 13 条），模型没有学到 PTS (Peroxisomal Targeting Signal) 的有效特征
4. **Cytoplasm → Nucleus**：细胞质和细胞核距离近，部分蛋白有双重定位

---

## 踩坑 / 易错点

1. **`plot_confusion_matrix` 的 `save_path` 默认值**：
   - Task 模板写的默认值是 `"confusion_matrix.png"`（相对路径）
   - 主人改成了 `"week11/day4/confusion_matrix.png"`
   - 实际运行时 `main()` 没有传 `save_path`，用的是默认值
   - 如果从 repo 根目录运行，改后的路径是对的；如果从 `day4/` 目录运行，会保存到 `week11/day4/week11/day4/confusion_matrix.png`
   - **当前输出显示保存成功**，说明是从根目录运行的 ✓

2. **`torch.load` 缺少 `weights_only` 参数**：
   - PyTorch 2.6+ 会 warning，建议加 `weights_only=True`

3. **设备选择**：主人把 task 模板的 `cpu` 改成了 `cuda if available`，这是正确的改进（虽然 ESM-2 8M 在 CPU 上也够快）

---

## 输出问题回答评估

### Q1：混淆矩阵对角线代表什么？哪个类别对角线值最低？为什么？

主人的回答 ✅ 正确：
- 对角线 = 预测正确
- Peroxisome 最低
- 原因：样本太少（93 条，测试集仅 13 条）

补充：不仅是样本数量问题，Peroxisomal Targeting Signal (PTS) 本身变异大、信号弱，即使是人类也难以仅从序列判断。

### Q2：Macro F1 vs Overall Accuracy

主人的回答 ✅ 核心正确：
- Macro F1 更能反映不均衡数据上的真实能力
- 原因：每个类别权重相等，不受样本数量影响

补充：实际数据中 Accuracy=0.664 vs Macro F1=0.551，差距 11.3% 就是不均衡造成的——Nucleus（369 条）和 Cytoplasm（261 条）两个大类拉高了准确率，但 Peroxisome（13 条）和 Golgi（21 条）的惨淡表现被平均掉了。

### Q3：Peroxisome 的保守预测策略

主人的回答 ✅ 逻辑正确：
- 模型只在高置信度时才预测为 Peroxisome → Precision 高、Recall 低

**但实际数据需要修正**：Peroxisome 的 Precision = 0.200，Recall = 0.231，**两个都很低**！这说明模型不仅保守（Recall 低），而且连保守预测的质量也不好（Precision 也低）。真正的情况是：模型几乎不预测 Peroxisome，偶尔预测几次还经常错——这是一个 **被模型"放弃"的类别**。

这比单纯的"保守策略"更严重。解决方案可以考虑：
- 过采样 Peroxisome（SMOTE 或重复采样）
- Focal Loss 替代加权 CrossEntropy
- 数据增强（序列扰动）

---

## 总结

| 维度 | 评价 |
|------|------|
| 代码质量 | ✅ 干净整洁，函数分工明确 |
| 功能完整性 | ✅ 三个 TODO 全部完成 |
| 一致性验证 | ✅ Test Acc 66.4% vs Val Acc 63.4%，差距 3%，合理 |
| Q&A 质量 | ✅ Q1/Q2 准确，Q3 逻辑对但需修正数据细节 |
| 主要发现 | 不均衡问题严重：Macro F1 (0.551) 比 Accuracy (0.664) 低 11.3% |
| 改进方向 | 少数类过采样 / Focal Loss / 更强的分类头 |

---

_Review by Talos | 2026-06-25_
