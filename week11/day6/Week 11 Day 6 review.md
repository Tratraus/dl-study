# Week 11 Day 6 Review：Fine-tuned 模型的完整测试评估

## 代码结构分析

| 函数 | 职责 |
|------|------|
| `get_predictions()` | 与 Day4 相同，遍历 test_loader 收集预测/真实标签 |
| `per_class_report()` | 打印每类 P/R/F1，额外返回 `(acc, mf1, f1_by_class)` 用于对比 |
| `print_comparison()` | 硬编码 Day4 结果，与 Day6 Fine-tuned 结果做逐类对比表 |
| `plot_confusion_matrix()` | 与 Day4 相同，保存为 `confusion_matrix_finetune.png` |
| `main()` | 加载 Day5 best_finetune.pt（ESM + 分类头）→ 测试集评估 → 对比 |

代码复用度高，`get_predictions` 和 `plot_confusion_matrix` 与 Day4 几乎一致。

---

## 核心对比结果

### 整体指标

| 指标 | Day4 Frozen | Day6 Fine-tuned | Δ |
|------|:-----------:|:---------------:|:--:|
| Overall Accuracy | 0.664 | 0.676 | **+0.012** |
| Macro F1 | 0.551 | 0.581 | **+0.030** |

### Val vs Test 提升对比

| 数据集 | 提升幅度 |
|--------|---------|
| Val Acc | +0.043（63.4% → 677%） |
| Test Acc | +0.012（66.4% → 676%） |

**Test 提升（1.2%）远小于 Val 提升（4.3%）**，差距 3.1%，说明 Day5 的 Fine-tuning 在一定程度上**过拟合到了验证集**。这也印证了 Day5 review 中的观察：train/val 差距从 2.2% 扩大到 6.5%。

### 逐类别 F1 变化

| 类别 | Frozen | FT | Δ | 趋势 |
|------|:------:|:--:|:--:|:----:|
| Extracellular | 0.894 | 0.924 | +0.030 | ↑ |
| Plastid | 0.707 | 0.771 | **+0.064** | ↑↑ |
| Mitochondria | 0.629 | 0.688 | **+0.059** | ↑↑ |
| Endoplasmic reticulum | 0.481 | 0.563 | **+0.082** | ↑↑ |
| Golgi apparatus | 0.338 | 0.386 | +0.048 | ↑ |
| Peroxisome | 0.214 | 0.250 | +0.036 | ↑ |
| Cell membrane | 0.683 | 0.694 | +0.011 | ─ |
| Nucleus | 0.753 | 0.744 | -0.009 | ─ |
| Cytoplasm | 0.520 | 0.516 | -0.004 | ─ |
| Lysosome/Vacuole | 0.291 | 0.270 | **-0.021** | ↓ |

**最大提升**：Endoplasmic reticulum (+0.082)、Plastid (+0.064)、Mitochondria (+0.059)
**唯一下降**：Lysosome/Vacuole (-0.021)

---

## 混淆矩阵对比分析

Fine-tuned 混淆矩阵相比 Frozen 的关键变化：

1. **ER、Plastid、Mitochondria 对角线变深** — ESM-2 解冻层学到了更细粒度的序列特征（ER 信号肽、转运肽、线粒体靶向序列），这些是有明确生物学信号的类别

2. **Peroxisome Recall 提升（0.231 → 0.385）** — 从 13 条中正确识别 5 条（vs 之前 3 条），虽然仍然低但有进步

3. **Lysosome/Vacuole Recall 提升但 Precision 下降** — 模型开始更多地预测这个类别，但也引入了更多误判（与 Cytoplasm 的混淆加剧）

4. **Cytoplasm 仍然被"挤压"** — Recall 从 0.475 降到 0.464，Fine-tuning 让更多 Cytoplasm 被误分为其他类别

---

## 踩坑 / 易错点

1. **`unfreeze_last_n_layers` 必须在 `load_state_dict` 之前调用** — 这是因为 `load_state_dict` 按 key 匹配参数，技术上不影响加载，但 `unfreeze_last_n_layers` 同时设置了 `model.train()` 模式，确保 dropout 等层在训练时正确工作。对于纯推理场景（如本 Day），如果只是 eval 模式，实际上不调用也能加载成功，但保持代码一致性是好习惯

2. **对比表硬编码** — `print_comparison` 把 Day4 的结果直接写死在代码里，方便但不灵活。如果 Day4 结果需要修正，Day6 也要同步改。更好的做法是从文件读取或传参

3. **`torch.load` 缺少 `weights_only`** — PyTorch 2.6+ warning，与 Day3/Day5 一致

---

## 输出问题回答评估

### Q1：最大提升 / 下降类别

主人的回答 ✅ 部分正确：
- ER 提升最大（+0.082）✓
- Lysosome/Vacuole 唯一下降 ✓
- 但原因分析需要补充：Lysosome 下降不仅是样本少，更关键的是 Fine-tuning 后**与 Cytoplasm 的混淆加剧**。Lysosome 蛋白在细胞质中合成，Fine-tuning 让 ESM-2 学到了更多 Cytoplasm 特征，反而把 Lysosome "拉"向了 Cytoplasm
- 主人提到 "lysosome 和 vacuole 在生物学上和 ER 有密切联系"——这个说法不太准确。Lysosome/溶酶体确实与 ER 有膜运输关系，但它们的**蛋白质序列特征**差异很大（溶酶体酶有 mannose-6-phosphate 标签，ER 蛋白有 KDEL/HDEL 信号）。导致 Lysosome F1 下降的原因是与 Cytoplasm 混淆，不是与 ER

### Q2：Val vs Test 提升差距

主人的回答 ✅ 正确：
- Test 提升 +1.2% vs Val 提升 +4.3%，差距明显
- 过拟合验证集的判断正确

补充：实际泛化提升只有 Val 提升的 28%（1.2/4.3），这在小数据集 Fine-tuning 中很常见——ESM-2 解冻后有 2.5M 可训练参数，但只有 ~6000 训练样本，模型容量远超数据量

### Q3：为什么必须调用 `unfreeze_last_n_layers`

主人的回答 ⚠️ 逻辑有误：
- 说 "会导致 load_state_dict 时出现参数不匹配的错误" —— **这不对**。Day5 保存的 checkpoint 包含 ESM-2 的全部参数（frozen + unfrozen），`load_state_dict` 按 key 匹配，**技术上不调用也能成功加载**
- 真正的原因是：`unfreeze_last_n_layers` 设置了 `requires_grad=True` 和 `model.train()` 模式。如果不调用：
  - 如果后续有训练：梯度不会流过 ESM-2（因为 `requires_grad=False`），解冻形同虚设
  - 如果纯推理（本 Day）：影响较小，但保持代码一致是好习惯
  - 不会出现 "参数不匹配" 的错误，因为所有参数都存在

---

## 总结

| 维度 | 评价 |
|------|------|
| 代码质量 | ✅ 复用 Day4 结构，对比表设计清晰 |
| 核心结论 | ✅ Fine-tuning 有效，Macro F1 +3.0%，但 Test 提升（+1.2%）远小于 Val 提升（+4.3%） |
| 过拟合信号 | ⚠️ Day5 的 train/val 差距（6.5%）在 Test 上得到验证（实际泛化提升只有 Val 的 28%） |
| 类别分析 | ✅ ER/Plastid/Mito 提升最大（有明确生物信号），Lysosome 唯一下降 |
| 改进方向 | Early Stopping、更多数据、Focal Loss 处理少数类 |

**Week 11 全景回顾**：
- Day1-2：数据准备（ESM-2 嵌入 + 真实数据集）
- Day3：冻结 ESM-2 + MLP 分类头（Val 63.4%）
- Day4：测试集评估 + 混淆矩阵（Test 66.4%）
- Day5：Fine-tuning 最后 2 层（Val 677%）
- Day6：Fine-tuned 测试集评估（Test 676%，Macro F1 0.581）

从冻结到 Fine-tuning，整体提升了 **+1.2% 准确率**和 **+3.0% Macro F1**。提升不大但方向正确，进一步提升需要更多数据或更强的类别平衡策略。

---

_Review by Talos | 2026-06-25_
