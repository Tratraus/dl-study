# Week 14 Day 3 总结与复盘

## 一、运行输出

```text
f1_micro: 0.7272727272727273
f1_macro: 0.7222222222222222
f1_samples: 0.7499999999999999
auroc_macro: 0.9166666666666666
auprc_macro: 0.9444444444444443
valid_auroc_labels: 3
```

```text
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /home/tratr/miniconda3/envs/dl-study/bin/python
cachedir: .pytest_cache
rootdir: /home/tratr/dl-study/week14
plugins: anyio-4.13.0
collecting ... collected 4 items

tests/test_day3_metrics.py::test_perfect_prediction_has_f1_one PASSED    [ 25%]
tests/test_day3_metrics.py::test_logits_to_predictions PASSED            [ 50%]
tests/test_day3_metrics.py::test_shape_mismatch_raises_value_error PASSED [ 75%]
tests/test_day3_metrics.py::test_label_without_both_classes_is_skipped_for_auc PASSED [100%]

============================== 4 passed in 1.36s ===============================
```

Day 3 专项测试 **4/4 通过**。全量测试当前为 `10 passed, 3 failed`，3项失败均来自旧的 `test_day2_loss.py` 中未定义的 `pos_weight`、`loss_fn` 等变量，与 Day 3 实现无关。

---

## 二、输出结果是如何得到的

在 `threshold=0.5` 下，合成样例的二值预测为：

```text
[1, 0, 0]
[1, 0, 0]
[0, 1, 0]
[0, 1, 1]
```

汇总所有样本与标签位置后：

```text
TP = 4
FP = 1
FN = 2
```

因此：

$$
F1_{micro}=\frac{2TP}{2TP+FP+FN}=\frac{8}{11}=0.7273
$$

三个标签各自的 F1 为：

```text
标签1：1.0000
标签2：0.5000
标签3：0.6667
```

因此：

$$
F1_{macro}=\frac{1+0.5+0.6667}{3}=0.7222
$$

四个样本各自的 F1 为：

```text
样本1：0.6667
样本2：0.6667
样本3：1.0000
样本4：0.6667
```

所以：

$$
F1_{samples}=0.7500
$$

这说明三种 F1 的主要差别是聚合维度：

- `micro`：汇总全部标签位置，再计算一次 F1。
- `macro`：先计算每个标签的 F1，再让每个标签平等参与平均。
- `samples`：先计算每条蛋白质的标签集 F1，再对样本平均。

---

## 三、今日问答（修正后）

### 1. 如果 `micro_f1` 明显高于 `macro_f1`，对当前 Top-50 长尾任务意味着什么？

通常说明模型在高频标签上表现较好，但在部分低频标签上表现较差。高频标签贡献更多 TP/FP/FN，因此更能主导 micro F1；macro F1 让每个标签权重相同，会暴露模型对长尾标签的忽视。这不是“整体特征与局部特征”的区别，而是不同频率标签之间的表现是否均衡。

### 2. 为什么“全部预测为0”可能拥有很高的按位 accuracy，却是一个无用模型？

当前每个样本平均只有2.25个正标签，而标签空间有50维，因此绝大多数位置本来就是0。全部预测为0会依靠大量真负例获得很高的按位 accuracy，但所有正标签的 Recall 都是0，没有识别出任何蛋白质功能，因此没有实际价值。

### 3. Macro F1 中某个低频标签的 F1 为0，可能是模型能力问题，也可能是什么数据问题？

除了模型没有学会该标签，也可能是该标签正样本总量太少、切分后验证集只有极少正例甚至没有正例、标签存在缺失或噪声，或者训练集与验证集分布不一致。正样本稀少是数据问题；欠拟合则是模型表现问题，两者不能直接画等号。

### 4. 为什么在当前高度不平衡任务中，AUPRC 通常比 AUROC 更值得关注？

长尾任务中负样本数量非常多。AUROC 中 FPR 的分母包含全部负样本，因此即使模型产生了一定数量的假阳性，FPR 仍可能很小，使 AUROC 看起来较高。AUPRC 直接关注正预测中有多少是真的，以及真实正样本找回了多少，因此更容易暴露稀有标签上的假阳性和漏检。

AUROC 与 AUPRC 回答的问题不同，不能简单认为 AUPRC 在任何场景都优于 AUROC；只是在当前高度不平衡的任务中，AUPRC 通常更具解释力。

### 5. 为什么 Day 3 要固定阈值0.5，而不是现在就选一个让 F1 最高的阈值？

Day 3 的目标是验证评估函数，而不是优化模型的决策规则。固定0.5可以让所有指标基于同一规则计算，避免把“指标实现是否正确”和“阈值是否最优”混在一起。

阈值必须在验证集上搜索。如果为了获得最高 F1 而直接在测试集上选阈值，就会造成测试集信息泄漏，得到过度乐观的结果。

```text
Day 3：固定阈值，校准评估尺子
Day 4：在验证集上搜索阈值
最终测试：冻结阈值后只评估一次
```

---

## 四、代码复盘

Day 3 已完成以下数据流：

```text
logits [N,K]
    ↓ sigmoid
probabilities [N,K]
    ↓ threshold=0.5
multi-hot predictions [N,K]
    ↓ sklearn.metrics
micro / macro / samples / per-label / AUROC / AUPRC
```

已实现：

- `logits_to_predictions()`：logits 转概率与二值预测。
- `_validate_inputs()`：检查维度、shape、multi-hot值和概率范围。
- `_macro_auc_scores()`：逐标签计算 AUROC/AUPRC，并跳过全0或全1标签。
- `compute_multilabel_metrics()`：统一返回 Precision/Recall/F1、per-label F1、AUROC 和 AUPRC。

当前4项基础测试覆盖：

1. 完美预测时三种 F1 均为1。
2. logits 经 sigmoid 与阈值转换正确。
3. `y_true` 与 `y_prob` shape 不匹配时抛出 `ValueError`。
4. 评估集中全0标签不参与 macro AUROC 平均。

后续可扩展但不影响 Day 3 核心完成的测试：

- 全零预测不产生 NaN。
- 构造 `micro_f1 > macro_f1` 的长尾示例。
- 手算并核对 `samples_f1`。

---

## 五、Day 3 完成记录

```text
Week14 Day3 完成记录：
- 实现多标签评估模块，输入为[N,K]真实标签与预测概率。
- 固定threshold=0.5，将每个标签独立二值化。
- 实现Precision/Recall/F1的micro、macro、samples聚合及per-label F1。
- 实现micro/macro AUROC与AUPRC。
- 对评估集中全0或全1的标签跳过AUROC，并记录有效标签数量。
- 合成数据sanity check：micro F1=0.7273，macro F1=0.7222，samples F1=0.7500，macro AUROC=0.9167，macro AUPRC=0.9444。
- Day3基础测试4/4通过。
- 核心认识：micro衡量整体标签决策，macro暴露低频标签失败，samples衡量典型蛋白质标签集表现。
- 高度不平衡任务中，按位accuracy可能被大量真负例虚高；AUPRC通常比AUROC更能暴露稀有正标签上的错误。
- 阈值调优必须与指标实现分离，并且只能在验证集上完成。
```

---

## 六、研究者视角观察

> 在长尾多标签任务中，AUROC 较高不必然意味着稀有正标签的预测可靠。必须结合 AUPRC、macro F1、per-label F1 以及每个标签的有效正样本数共同判断。

当前所有指标来自合成数据，用途是验证评估函数的正确性，不能将其解读为真实模型的 baseline 性能。
